"""Pinned Qwen interface, assistant-only targets and in-assembly intervention."""
import hashlib
import json
from contextlib import contextmanager

from .protocol import DEFAULTS, LABELS, SINGLE_LABEL, generation_limit, task_prompt, target


def load_model(revision, cache_dir=None):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check device access outside the sandbox and install CUDA PyTorch.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This protocol requires BF16-capable CUDA hardware')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Supply a full immutable model commit SHA with --revision')
    processor = AutoProcessor.from_pretrained(DEFAULTS['model'], revision=revision,
                    min_pixels=DEFAULTS['min_pixels'], max_pixels=DEFAULTS['max_pixels'],
                    use_fast=False, cache_dir=cache_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(DEFAULTS['model'],
                revision=revision, torch_dtype=torch.bfloat16, attn_implementation='sdpa', cache_dir=cache_dir)
    model.requires_grad_(False)
    return model.cuda(), processor


def adapt(model, scope, seed):
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    if scope not in ('L', 'M', 'LM'):
        raise ValueError('Unknown adaptation scope')
    model.requires_grad_(False)
    if 'L' in scope:
        targets = [name for name, _ in model.named_modules()
                   if name.startswith('model.language_model.layers.') and
                   name.rsplit('.', 1)[-1] in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
        if not targets:
            raise RuntimeError('Pinned Qwen LLM module paths did not match')
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                               target_modules=targets, bias='none'))
    for name, param in model.named_parameters():
        if '.visual.merger.' in name:
            param.requires_grad_('M' in scope)
    # FP32 trainable/master weights prevent BF16 rounding from swallowing small updates.
    # Forward callers use BF16 autocast; the frozen backbone stays BF16.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not ('lora_' in n or '.visual.merger.' in n) for n, _ in trainable):
        raise RuntimeError('Unexpected trainable parameter scope')
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model, trainable


class Encoder:
    def __init__(self, processor, dataset):
        self.processor, self.dataset = processor, dataset
        tok = processor.tokenizer
        longest = [target(dataset, [label]) for label in LABELS[dataset]] if dataset in SINGLE_LABEL else [target(dataset, LABELS[dataset])]
        max_generation = generation_limit(dataset)
        self.target_budget = max(len(tok.encode(t + tok.eos_token, add_special_tokens=False)) for t in longest)
        if self.target_budget > max_generation:
            raise ValueError('Longest canonical target exceeds fixed generation budget; revise protocol before screening')

    def __call__(self, row, training=False):
        import torch
        from PIL import Image
        p, tok = self.processor, self.processor.tokenizer
        system = task_prompt(self.dataset)
        tokens = tok.encode(row['text'], add_special_tokens=False)
        if not tokens or any(i in set(tok.all_special_ids) for i in tokens):
            raise ValueError(f'Empty body or reserved special token: {row["id"]}')
        limit = 128 if self.dataset in SINGLE_LABEL else 384
        with Image.open(row['image']) as im:
            image = im.convert('RGB')
        if row.get('image_hash') and hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest() != row['image_hash']:
            raise ValueError(f'Image changed since manifest creation: {row["id"]}')
        messages = [dict(role='system', content=system), dict(role='user', content=[
            dict(type='image'), dict(type='text', text='\nBody:\n')])]
        prompt = p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = p(text=[prompt], images=[image], return_tensors='pt')
        ids = inputs['input_ids'][0].tolist()
        boundary = tok.encode('<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False)
        if ids[-len(boundary):] != boundary:
            raise RuntimeError('Unexpected Qwen assistant boundary')
        left = len(ids) - len(boundary)
        # Assemble token spans explicitly: BPE merges cannot consume the field marker.
        # Reserve the longest legal target, independent of this example's true labels.
        reserve = self.target_budget
        budget = min(limit, len(tokens), DEFAULTS['max_length']-len(ids)-reserve)
        if budget <= 0:
            raise ValueError('Protected prompt/image leaves no room for body and target')
        body_ids = tokens[:budget]
        positions = list(range(left, left+budget))
        ids = ids[:left] + body_ids + ids[left:]
        inputs['input_ids'] = torch.tensor([ids], dtype=torch.long)
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        if training:
            answer = tok.encode(target(self.dataset, row['labels']) + tok.eos_token, add_special_tokens=False)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor([answer])], dim=1)
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            inputs['labels'] = torch.tensor([[-100]*len(ids)+answer])
        return inputs, positions, budget < len(tokens)



class Intervention:
    """Hook the language decoder input AFTER Qwen assembles merged visual embeddings.

    Decoder cache positions distinguish prefill from subsequent generation tokens.
    No external inputs_embeds generation shortcut is used.
    """
    def __init__(self, model, stats=None):
        self.stats, self.state, self.observed, self.logs = stats, None, {}, []
        self.image_token_id = getattr(getattr(model, 'config', None), 'image_token_id', None)
        # Resolve exactly one decoder, including PEFT's transparent wrapper.
        decoders = [m for n, m in model.named_modules() if n.endswith('language_model')]
        if len(decoders) != 1:
            raise RuntimeError('Expected one Qwen language decoder')
        self.handle = decoders[0].register_forward_pre_hook(self._hook, with_kwargs=True)

    @contextmanager
    def use(self, row, inputs, positions, condition='full', seed=2026, scale=0.05, capture=False):
        if condition not in ('full', 'mask_image', 'mask_text') or scale < 0:
            raise ValueError('Invalid intervention condition or negative noise scale')
        if inputs['input_ids'].shape[0] != 1:
            raise ValueError('The fixed validation protocol requires batch size 1')
        if len(set(positions)) != len(positions) or any(i < 0 or i >= inputs['input_ids'].shape[1] for i in positions):
            raise ValueError('Invalid body token positions')
        if self.state is not None:
            raise RuntimeError('Intervention context cannot be nested')
        self.state = dict(id=row['id'], ids=inputs['input_ids'], positions=positions,
                          condition=condition, seed=seed, scale=scale, capture=capture)
        self.observed = {}
        try:
            yield self
        finally:
            self.state = None

    def _hook(self, module, args, kwargs):
        import torch
        s = self.state
        if s is None:
            return
        x = kwargs.get('inputs_embeds')
        if x is None:
            raise RuntimeError('Qwen decoder no longer receives assembled inputs_embeds')
        cache_position = kwargs.get('cache_position')
        if cache_position is not None and cache_position.numel() and cache_position[0].item() > 0:
            return
        if x.shape[1] != s['ids'].shape[1]:
            raise RuntimeError('Unexpected prefill reconstruction / missing cache positions')
        image_id = self.image_token_id or getattr(module.config, 'image_token_id', 151655)
        spans = dict(image=(s['ids'][0] == image_id).nonzero().flatten().tolist(), text=s['positions'])
        if not spans['image']:
            raise RuntimeError('No expanded image content tokens')
        if set(spans['image']) & set(spans['text']):
            raise ValueError('Body positions overlap protected image positions')
        if s['capture']:
            self.observed = {k:x[0, v].detach().float().cpu() for k,v in spans.items()}
        if s['condition']=='full':
            return
        modality = {'mask_image':'image', 'mask_text':'text'}[s['condition']]
        positions = spans[modality]
        if not positions:
            raise RuntimeError('Cannot mask an empty body')
        stat = self.stats[modality]
        mean = stat['mean'].float().cpu()
        std = stat['std'].float().cpu()
        if mean.shape != (x.shape[-1],) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError('Malformed frozen masking statistics')
        replacements = []
        for pos in positions:
            key = f'{s["seed"]}:{s["id"]}:{modality}:{pos}'.encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % (2**63-1)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            noise = torch.randn(mean.shape, generator=generator)
            replacements.append(mean + s['scale']*std*noise)
        changed = x.clone()
        changed[0, positions] = torch.stack(replacements).to(device=x.device, dtype=x.dtype)
        self.logs.append(dict(id=s['id'], modality=modality, condition=s['condition'], noise_seed=s['seed'], noise_scale=s['scale'], shape=list(x.shape),
                              positions=positions, original_norm=x[0, positions].float().norm(dim=-1).mean().item(),
                              replacement_norm=changed[0, positions].float().norm(dim=-1).mean().item(),
                              location='model.language_model prefill inputs_embeds'))
        kwargs['inputs_embeds'] = changed
        return args, kwargs

    def close(self):
        self.handle.remove()

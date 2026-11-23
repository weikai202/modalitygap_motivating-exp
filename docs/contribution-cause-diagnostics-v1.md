# CrisisMMD 负 text contribution / Hateful Memes 零贡献诊断

使用原64条训练样本得到的同一 epoch8 断点；没有训练或改动生产 masking。官方开发子集从原32条扩大到128条（保留原32＋新增96，每类新增48，seed2026，排除训练ID、归一化正文和RGB重复），没有使用test。新结果独立保存于 `artifacts/contribution-cause-diagnostics-v1/`。

当前证据未发现图文导出错位、标签反转或被遮内容进入decoder的实现错误。CrisisMMD 的负值来自该模型在部分样本上的图文融合干扰，且原32条子集放大了幅度；Hateful Memes 的0首先是两条预测变化抵消，图片里的正文又使单独文本输入具有冗余。这里的贡献是条件消融F1差，保留原序列长度、图像网格与位置。

## 统一复算

| 数据集 | 子集 | full F1 | 遮图F1 | 遮文F1 | image贡献(pp) | text贡献(pp) |
|---|---|---|---|---|---|---|
| crisismmd | original32 | 0.8745 | 0.8118 | 1.0000 | 6.27 | -12.55 |
| crisismmd | additional96 | 0.7387 | 0.7643 | 0.7871 | -2.56 | -4.85 |
| crisismmd | combined128 | 0.7746 | 0.7768 | 0.8419 | -0.22 | -6.73 |
| hatefulmemes | original32 | 0.6235 | 0.5836 | 0.6235 | 3.99 | 0.00 |
| hatefulmemes | additional96 | 0.6354 | 0.5833 | 0.6457 | 5.20 | -1.03 |
| hatefulmemes | combined128 | 0.6328 | 0.5853 | 0.6406 | 4.75 | -0.78 |

## 数据与实现核对

扩展开发数据的标签逐条对照原发布记录；CrisisMMD 逐条断言 label == label_text == label_image。Hateful Memes 沿用原challenge二分类标签，已知ID85761与后续细粒度版本的分歧仍保留，并单独重评分。原32条×3条件×2任务共192条生成token逐条完全复现。

32个真实epoch8内容隔离probe（每任务8条×遮文/遮图）：替换全部被遮token或同形状像素，prefill及后续3个cache step的原始/处理后logits最大差均为0；full条件的相同替换均引起非零变化。每个probe都核验实际attention层隐藏keys不可达。说明在已测样本、原结构条件下内容屏蔽生效；正文长度和图像网格保留是协议定义的一部分。

新增joint blocking控制通过独立小Qwen测试，prefill/cache/repetition penalty均同时隔离两路内容；控制只用于本次诊断，不改变正式训练实现。

## CrisisMMD：四条错误与文本干扰

原32条full为28/32正确，遮文为32/32：4条均为错→对，无对→错。两条在遮图、遮文时都正确，full却错误；另外两条full延续遮图条件的错误。这是观察到的融合行为，不能进一步据此定位某个神经元或断言一般性的文本有害。

| ID | 参考 | full | 遮图（保留文本） | 遮文（保留图像） | 原图人工查看 |
|---|---|---|---|---|---|
| 910111529290301445_0 | informative | not_informative | informative | informative | 飓风路径/强度预报图 |
| 920320104839237632_0 | not_informative | informative | not_informative | not_informative | 人像照片 |
| 922558317049864192_0 | not_informative | informative | informative | not_informative | Tesla产品/住宅照片 |
| 930072830032666626_0 | not_informative | informative | informative | not_informative | CNN标志画面 |

将被遮正文长度统一为1或64 token，遮文F1均为 1.0000/1.0000；同时遮图遮文F1为 0.3333。原32条的满分不需要正文长度差异，且去掉图像内容后消失。

原32条有 6 条含明显乱码标记，原发布记录本身已有这些字符。移除URL/@mention（未修复正文乱码）后full F1为 0.9062，改善2条、损害1条，仍未达到遮文F1。该控制同时改变正文内容和长度，不能把变化完全归因为URL语义。部分错误正文为正常ASCII，所以乱码也不能独立解释负贡献。

扩大开发集后遮文不再满分，但额外96条及合计128条上text contribution仍为负；这是该checkpoint/该开发子集上的文本干扰证据，不能视为全量benchmark结论。

## Hateful Memes：精确抵消与图片OCR

原32条：full与遮文都是20/32正确，且混淆矩阵均为 TN=11、FP=5、FN=7、TP=9。30条类别预测一致；ID63987 对→错，ID67208 错→对，两次翻转都发生在 not_hateful 类，所以两个条件的macro-F1精确相同。0不是四舍五入，也不表示模型忽略了独立文本。

原图单独OCR（没有提供正文或标签）8条，正文词召回平均 95.00%，归一化词序列完全匹配 6/8。证明被遮的独立正文仍有一条可从图片恢复的通道；这支持冗余解释，但不是分类内部一定依赖OCR的因果证明。

固定被遮正文长度1/64 token后的F1为 0.6235/0.5934；联合遮蔽F1为 0.3333。固定64 token使F1比原结构下降约3.02个百分点，显示长度/位置会影响该诊断数值；这属于保留结构的消融定义，不是隐藏正文语义仍可被读取的证据。

仅把ID85761按后续细粒度版本重评分，原32条text contribution仍为 0.00 个百分点；这一个标签版本差异不能解释原先的零值。

## 复现与边界

准备数据：`.venv/bin/python scripts/prepare_contribution_diagnostics.py`（目录已存在时拒绝覆盖）。GPU运行：`OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/diagnose_contribution_causes.py --dataset crisismmd`，另一个任务将参数改为hatefulmemes；已完成目录拒绝覆盖。报告复算：`OMP_NUM_THREADS=1 .venv/bin/python scripts/report_contribution_causes.py`。

所有新分类预测再次从生成token重解码、复算分组F1与贡献。完整混淆矩阵、错对转移、配对bootstrap探索区间和敏感性结果见新目录diagnosis.json；判定依据和哈希见validation.json。生产源码与先前汇总报告SHA不变，204份更早结果文件SHA不变。

结论的范围是同一单seed、小规模训练得到的checkpoint及选中开发样本；没有把开发子集扩充当作新的训练或正式test，也没有通过调mask、改标签或换样本使贡献变正。

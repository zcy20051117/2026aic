# 2026 AIC: DINOv3 Mask2Former Best-Model Pipeline

This repository contains the source code and tests for the best-result training lineage:

1. Prepare and verify the `trainadd` hard-sample inputs.
2. Train the V4 clean-hard base model from frozen DINOv3 ViT-L/16 features.
3. Initialize V5 from V4 `best.pth` and train with LoveDA plus vehicle-rich AIC images.
4. Continue V5 through `resume15`, restoring model, optimizer, scheduler, and RNG state from `last.pth`.
5. Select the final checkpoint by clean AIC validation mIoU; run prediction, evaluation, and submission-format validation.

Datasets, generated manifests, model weights, logs, outputs, and submissions are intentionally excluded. Set the variables in `.env.example` for your environment before use. See `docs/reproduction.md` and `docs/checkpoint-lineage.md`.

这算是我第3次从头到位正儿八经参加的算法比赛了，不过结果也确实不尽人意，最开始我使用的是yolo26l-sem去处理，结果发现泛性很差，val上0.81的miou，线上提交只有66，后来使用了sam3和dinov3进行尝试，sam3训练开销太大，租的4080s还是带不动，就采用dinov3进行尝试，零样本在val上有0.36的样子，基线可以达到0.684，我是冻结了backbone，全参训了mask2former解码器，feature confusion方法我放后面了

不过这明显是gpt给我写的，我在最后才知道有特征融合这个东西，会的还是太少了。

后面我上到了68.911，这是因为我发现线上test集中有很多烂图，画质非常低，于是我就挑选了train中相同场景的图片，做了模糊化处理，然后真的可以上分。不过可惜之处在于我没有考虑到权重的问题，这些烂图和原始训练集的训练权重相同，这些我没有考虑到，还有就是训练的时候，我只看到荒地类别的训练效果差，就只将这个类别的训练权重改成1.5，其余训练权重都设置为1。这个比赛5月初就开始了，我9月初才参加，1个月的时间上分对我来说是有点紧张了，很多方案没有去尝试参数没有调，数据增强也没有去试，loss是weightce+dice，也没有尝试其他权重，mask2former其实也可以换，特征融合我就更没有考虑到了，经验确实不足吧，比赛题目拿到手我感觉连一个工作链都没有，就跑完零样本和基线以后开始瞎搞。



# 1. 两条特征提取路径

一张 1024×1024 图像经过归一化后，同时进入两条路径：

* **冻结的 DINOv3 ViT-L/16**：每个 patch 为 16×16，因此 patch 网格是 64×64，特征通道数为 1024。代码取配置索引 **[4, 11, 17, 23]** 对应的四份中间输出，记为 X₄,X₁₁,X₁₇,X₂₃。它们的空间尺寸**都是 64×64**，区别是网络深度，而不是分辨率；较深层的输出已经包含此前各层处理的信息。
* **可训练的空间先验模块 SPM**：直接从原图用卷积提取局部纹理、边缘等信息，产生四种分辨率的特征。


| SPM 特征 | 相对原图步长 | 1024×1024 输入时的尺寸 |
| -------- | ------------ | ----------------------- |
| C₁      | 1/4          | 256×256                |
| C₂      | 1/8          | 128×128                |
| C₃      | 1/16         | 64×64                  |
| C₄      | 1/32         | 32×32                  |

所以，**多尺度主要由 SPM 建立**，并不是 DINOv3 的四层天然输出了四种分辨率。模型配置 (loveda_mix_v5_staging/configs/baseline_v1.yaml:19) · Adapter 前向代码 (loveda_mix_v5_staging/aic_dino_m2f/models/adapter.py:97)

## 2. Adapter 怎样让两条路径交互

代码先给 C₂,C₃,C₄ 加上可学习的尺度标识，再把它们拼成一组空间特征。接下来依次进行四次交互：

`空间特征 → 读取 X₄ → 更新空间特征 → 读取 X₁₁ → 更新 → 读取 X₁₇ → 更新 → 读取 X₂₃`

每次交互中，**空间特征作为 query，当前 DINO 层的 patch 特征作为被读取的特征**。采用的是可训练的可变形注意力：当前配置为 16 个注意力头，模块会学习采样位置及其注意力权重，而不是在整张 64×64 特征图上做一次固定平均。注意力结果以残差形式更新空间特征，后面还有卷积前馈网络；最后一次交互还带有两次额外提取。

这里有一个容易误解的实现细节：**四份 DINO 输出预先在 `no_grad` 下提取，交互模块更新的是 SPM 那条空间特征流，并没有把更新后的 token 写回 DINOv3 主干。** 因而 DINO 主干仍冻结，Adapter 学习怎样利用它的特征。Adapter 前向代码 (loveda_mix_v5_staging/aic_dino_m2f/models/adapter.py:97)

## 3. 怎样得到送往解码器的四尺度融合特征

四次交互完成后，代码把更新过的空间特征拆回 C₂,C₃,C₄，再将上采样的 C₂ 与原来的 C₁ 相加，补出高分辨率分支。

随后，四份 DINO 特征分别插值到对应尺寸并**逐元素相加**：

* 1/4 尺度空间特征 ＋ 调整尺寸后的 X₄；
* 1/8 尺度空间特征 ＋ 调整尺寸后的 X₁₁；
* 1/16 尺度空间特征 ＋ 调整尺寸后的 X₁₇；
* 1/32 尺度空间特征 ＋ 调整尺寸后的 X₂₃。

相加后各自经过归一化，形成送往分割头的四尺度特征。这里的“相加”**没有显式设置**诸如 `0.2×第4层＋0.8×第23层` 的层级系数；但相加之前的可变形注意力、SPM，以及后续解码器的参数都是学习得到的，所以也不能把整个系统理解成简单平均。四尺度输出代码 (loveda_mix_v5_staging/aic_dino_m2f/models/adapter.py:132)

## 4. Mask2Former 内部再次融合多尺度

像素解码器先把 1/8、1/16、1/32 三个尺度投影到 **256 通道**，送进 **6 层多尺度可变形注意力编码器**。然后，它将 1/8 尺度的编码结果上采样，与 1/4 尺度分支融合，生成较高分辨率的 `mask_features`。这一步进一步结合了大范围语义和小目标、边界所需的细节。像素解码器 (loveda_mix_v5_staging/aic_dino_m2f/models/pixel_decoder.py:85)

Mask2Former 接着用 **100 个查询、9 层 Transformer 解码器**，轮流读取三个尺度的特征。每个查询最终预测两样东西：它属于哪个类别（8 类另加 no-object），以及它覆盖图像的哪片区域。最后对每个像素、每个类别，把各查询的“类别得分 × 区域 mask 得分”累加，选择得分最高的类别。分割头 (loveda_mix_v5_staging/aic_dino_m2f/models/mask2former.py:59) · 最终语义得分计算 (loveda_mix_v5_staging/aic_dino_m2f/models/postprocess.py:10)

一句话概括：**冻结 DINO 提供四个深度的语义特征；可训练 SPM 提供四个分辨率的空间细节；Adapter 用注意力读取 DINO 信息并做对应尺度相加；Mask2Former 再学习跨尺度融合并生成 mask。** Barren 的 `1.5` 是训练时的类别损失权重，**不是**上述任何一步的特征融合系数。

1. 两条特征提取路径

一张 1024×1024 图像经过归一化后，同时进入两条路径：

* **冻结的 DINOv3 ViT-L/16**：每个 patch 为 16×16，因此 patch 网格是 64×64，特征通道数为 1024。代码取配置索引 **[4, 11, 17, 23]** 对应的四份中间输出，记为 X₄,X₁₁,X₁₇,X₂₃。它们的空间尺寸**都是 64×64**，区别是网络深度，而不是分辨率；较深层的输出已经包含此前各层处理的信息。
* **可训练的空间先验模块 SPM**：直接从原图用卷积提取局部纹理、边缘等信息，产生四种分辨率的特征。


| SPM 特征 | 相对原图步长 | 1024×1024 输入时的尺寸 |
| -------- | ------------ | ----------------------- |
| C₁      | 1/4          | 256×256                |
| C₂      | 1/8          | 128×128                |
| C₃      | 1/16         | 64×64                  |
| C₄      | 1/32         | 32×32                  |

所以，**多尺度主要由 SPM 建立**，并不是 DINOv3 的四层天然输出了四种分辨率。模型配置 (loveda_mix_v5_staging/configs/baseline_v1.yaml:19) · Adapter 前向代码 (loveda_mix_v5_staging/aic_dino_m2f/models/adapter.py:97)

## 2. Adapter 怎样让两条路径交互

代码先给 C₂,C₃,C₄ 加上可学习的尺度标识，再把它们拼成一组空间特征。接下来依次进行四次交互：

`空间特征 → 读取 X₄ → 更新空间特征 → 读取 X₁₁ → 更新 → 读取 X₁₇ → 更新 → 读取 X₂₃`

每次交互中，**空间特征作为 query，当前 DINO 层的 patch 特征作为被读取的特征**。采用的是可训练的可变形注意力：当前配置为 16 个注意力头，模块会学习采样位置及其注意力权重，而不是在整张 64×64 特征图上做一次固定平均。注意力结果以残差形式更新空间特征，后面还有卷积前馈网络；最后一次交互还带有两次额外提取。

这里有一个容易误解的实现细节：**四份 DINO 输出预先在 `no_grad` 下提取，交互模块更新的是 SPM 那条空间特征流，并没有把更新后的 token 写回 DINOv3 主干。** 因而 DINO 主干仍冻结，Adapter 学习怎样利用它的特征。Adapter 前向代码 (loveda_mix_v5_staging/aic_dino_m2f/models/adapter.py:97)

## 3. 怎样得到送往解码器的四尺度融合特征

四次交互完成后，代码把更新过的空间特征拆回 C₂,C₃,C₄，再将上采样的 C₂ 与原来的 C₁ 相加，补出高分辨率分支。

随后，四份 DINO 特征分别插值到对应尺寸并**逐元素相加**：

* 1/4 尺度空间特征 ＋ 调整尺寸后的 X₄；
* 1/8 尺度空间特征 ＋ 调整尺寸后的 X₁₁；
* 1/16 尺度空间特征 ＋ 调整尺寸后的 X₁₇；
* 1/32 尺度空间特征 ＋ 调整尺寸后的 X₂₃。

相加后各自经过归一化，形成送往分割头的四尺度特征。这里的“相加”**没有显式设置**诸如 `0.2×第4层＋0.8×第23层` 的层级系数；但相加之前的可变形注意力、SPM，以及后续解码器的参数都是学习得到的，所以也不能把整个系统理解成简单平均。四尺度输出代码 (loveda_mix_v5_staging/aic_dino_m2f/models/adapter.py:132)

## 4. Mask2Former 内部再次融合多尺度

像素解码器先把 1/8、1/16、1/32 三个尺度投影到 **256 通道**，送进 **6 层多尺度可变形注意力编码器**。然后，它将 1/8 尺度的编码结果上采样，与 1/4 尺度分支融合，生成较高分辨率的 `mask_features`。这一步进一步结合了大范围语义和小目标、边界所需的细节。像素解码器 (loveda_mix_v5_staging/aic_dino_m2f/models/pixel_decoder.py:85)

Mask2Former 接着用 **100 个查询、9 层 Transformer 解码器**，轮流读取三个尺度的特征。每个查询最终预测两样东西：它属于哪个类别（8 类另加 no-object），以及它覆盖图像的哪片区域。最后对每个像素、每个类别，把各查询的“类别得分 × 区域 mask 得分”累加，选择得分最高的类别。分割头 (loveda_mix_v5_staging/aic_dino_m2f/models/mask2former.py:59) · 最终语义得分计算 (loveda_mix_v5_staging/aic_dino_m2f/models/postprocess.py:10)

一句话概括：**冻结 DINO 提供四个深度的语义特征；可训练 SPM 提供四个分辨率的空间细节；Adapter 用注意力读取 DINO 信息并做对应尺度相加；Mask2Former 再学习跨尺度融合并生成 mask。** Barren 的 `1.5` 是训练时的类别损失权重，**不是**上述任何一步的特征融合系数。

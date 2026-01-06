# Zou_MindEye
#### MindEye 在 NSD 数据集上的复现报告

本文档描述了我在 Natural Scenes Dataset (NSD) 上对 MindEye 方法的复现实验，包括已完成实验、部分完成实验以及由于计算资源限制未能完全复现的部分。

##### 1. 数据集与实验设置

所有实验均基于 Natural Scenes Dataset（NSD）。该数据集记录了被试在观看自然场景图像（来源于 MS-COCO）时的 fMRI 脑活动信号，是当前脑编码与脑解码研究中使用最广泛的数据集之一。

受时间与计算资源限制，本次复现实验仅选取 被试 1（Subject 1） 的数据进行训练与评估。原论文中使用了 Subject 1、2、5、7 四名被试，并报告四组实验结果的平均性能。
Subject 1 的数据包含 15,724 个体素（voxels），已能充分验证模型在单被试条件下的性能表现。

实验中使用的数据直接下载自作者在 HuggingFace 上公开的官方版本，因此数据的预处理流程、训练/测试划分方式均与 MindEye 原文及其相关工作保持一致。

##### 2. 模型结构概述

MindEye 模型主要由以下三个核心组件构成：

###### 2.1 Voxel-to-CLIP 主干网络

Voxel-to-CLIP（Voxel2CLIP）网络负责将 fMRI 体素信号映射至 CLIP 表征空间，并支持两种不同形式的输出：
1. CLIP 隐藏层输出（257 × 768）：用于后续的图像重建任务。
2. CLIP CLS token（768）：用于图像检索与脑检索任务。

Voxel2CLIP 网络采用大规模多层 MLP 架构。当输出为 257 × 768 的 CLIP 隐藏层时，模型参数量约为 9.48 亿。由于该规模的模型从头训练成本极高，本实验直接使用作者在 HuggingFace 上公开的、基于 Subject 1 预训练 的模型权重。

###### 2.2 Diffusion Prior

Diffusion Prior 基于开源的 DALL·E 2 扩散先验模型，并针对脑信号条件生成任务进行了适配与修改。该模块负责将 Voxel2CLIP 预测得到的 CLIP 表征映射至扩散模型可用的潜在条件空间，并支持后续与 Versatile Diffusion 的集成。

由于网络环境限制，本实验采用手动方式下载作者在 HuggingFace 上公开的 Diffusion Prior 及 Versatile Diffusion 权重。
在具体实现中，Diffusion Prior 与 Voxel2CLIP 被封装为 BrainDiffusionPrior 模块，并分别支持使用 CLIP 隐藏层输出（Hidden Layer）或最终层输出（Final Layer）作为条件输入。

###### 2.3 Stable Diffusion

Stable Diffusion 主要用于 MindEye 的低级管道（Low-Level Pipeline），以 img2img 方式为图像重建提供结构与低频视觉信息。本实验同样直接使用作者在 HuggingFace 上开源的官方模型权重。

##### 3 已完成的复现实验
###### 3.1 图像检索与脑检索

我成功复现了 MindEye 在 图像检索（Image Retrieval） 与 脑检索（Brain Retrieval） 任务上的实验结果。
下图展示了原论文中的实验结果：

![alt text](image-7.png)

下图为我在 Subject 1 上复现得到的实验结果：

![alt text](image-1.png)

![alt text](image-2.png)

![alt text](image-3.png)
其中包括
1. 前向检索（Forward Retrieval）
2. 后向检索（Backward Retrieval）

从 Top-1 准确率 可以看出，本实验结果与原论文报告的性能基本一致。在视觉高度相似的候选图像集合中，模型仍能够稳定地检索出对应的目标图像，验证了 Voxel2CLIP 映射的有效性。

###### 3.2 fMRI → 图像重建

本实验完整复现了 MindEye 的 fMRI 到图像重建流程。
下图为原论文中的重建结果示例：

![alt text](image-8.png)

下图为我在 Subject 1 上得到的重建结果：

![alt text](image-4.png)

可以观察到，生成图像在整体结构、语义类别及主要视觉特征上与真实图像高度一致，整体重建质量达到了原论文所展示的实验水平。
受时间限制，本次实验未进一步复现论文中其他被试或其他对比方法的重建结果。

###### 3.3 消融实验

我复现了原论文中的消融实验第一部分，原论文的表格如下：

![alt text](image-9.png)

受时间限制，本实验仅在 Subject 1 上进行了三组实验。其中：
1. MindEye（Low-Level）表示图像重建从低级管道生成的模糊结构图像开始。
2. MindEye（High-Level）表示图像重建仅基于高级语义管道进行。
下图分别展示了对应的重建结果：
![alt text](image-5.png)

![alt text](image-6.png)

根据原论文使用的评价指标，我得到的定量实验结果如下：

![alt text](image.png)

整体趋势与原论文保持一致，低级管道在结构与低级视觉指标上具有明显优势，而高级管道更偏向语义一致性。

##### 4 未完全复现的实验

为完整复现原论文其余消融实验，需要从头训练多种不同结构与配置的模型，包括：

![alt text](image-10.png)

![alt text](image-11.png)

![alt text](image-12.png)

然而，MindEye 的主干网络参数规模约为 9.4 亿，且映射至 CLIP 隐藏层（257 × 768）后，显著增加了计算量与显存占用。

本实验主要在 8 卡 × 24GB 显存 的学校服务器上进行。实际测试发现，多 GPU 数据并行并不会降低单卡显存占用，仅能提升吞吐率；即使启用 fp16 训练，整体训练过程依然极其耗时。

我也尝试在 AutoDL 算力云平台租用 单卡 48GB 显存 的服务器进行实验，但单个模型配置的完整训练时间仍超过 10 小时。在课程/项目截止日期前，难以完成全部消融实验的复现。

#### 总结
本次复现实验在 单被试（Subject 1）条件下 成功复现了 MindEye 的核心实验结果，包括检索任务、图像重建任务以及部分消融实验。实验结果在定性与定量层面均与原论文保持一致，验证了 MindEye 方法的有效性。
由于模型规模与计算资源限制，其余消融实验未能在限定时间内完成。

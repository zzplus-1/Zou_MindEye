# Zou_MindEye
### MindEye 在 NSD 数据集上的复现报告

本文档描述了我在 Natural Scenes Dataset (NSD) 上对 MindEye 方法的复现实验，包括已完成实验、部分完成实验以及由于计算资源限制未能完全复现的部分。

<img width="711" height="450" alt="image" src="https://github.com/user-attachments/assets/7adcdca0-69a0-4045-914d-610ee421e89e" />


#### 1. 数据集与实验设置

所有实验均基于 Natural Scenes Dataset（NSD）。该数据集记录了被试在观看自然场景图像（来源于 MS-COCO）时的 fMRI 脑活动信号，是当前脑编码与脑解码研究中使用最广泛的数据集之一。

受时间与计算资源限制，本次复现实验仅选取 被试 1（Subject 1） 的数据进行训练与评估。原论文中使用了 Subject 1、2、5、7 四名被试，并报告四组实验结果的平均性能。
Subject 1 的数据包含 15,724 个体素（voxels），已能充分验证模型在单被试条件下的性能表现。

实验中使用的数据直接下载自作者在 HuggingFace 上公开的官方版本，因此数据的预处理流程、训练/测试划分方式均与 MindEye 原文及其相关工作保持一致。
https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main/webdataset_avg_split

#### 2. 模型结构概述

MindEye 模型主要由以下三个核心组件构成：

##### 2.1 Voxel-to-CLIP 主干网络

Voxel-to-CLIP（Voxel2CLIP）网络负责将 fMRI 体素信号映射至 CLIP 表征空间，并支持两种不同形式的输出：
1. CLIP 隐藏层输出（257 × 768）：用于后续的图像重建任务。
2. CLIP CLS token（768）：用于图像检索与脑检索任务。

Voxel2CLIP 网络采用大规模多层 MLP 架构。当输出为 257 × 768 的 CLIP 隐藏层时，模型参数量约为 9.48 亿。由于该规模的模型从头训练成本极高，本实验直接使用在 HuggingFace 上公开的预训练的模型权重。
https://huggingface.co/openai/clip-vit-large-patch14/tree/main

##### 2.2 Diffusion Prior

Diffusion Prior 基于开源的 DALL·E 2 扩散先验模型，并针对脑信号条件生成任务进行了适配与修改。该模块负责将 Voxel2CLIP 预测得到的 CLIP 表征映射至扩散模型可用的潜在条件空间，并支持后续与 Versatile Diffusion 的集成。

由于网络环境限制，本实验采用手动方式下载作者在 HuggingFace 上公开的 Versatile Diffusion 权重。
https://hf-mirror.com/shi-labs/versatile-diffusion/tree/main

在具体实现中，Diffusion Prior 与 Voxel2CLIP 被封装为 BrainDiffusionPrior 模块，并分别支持使用 CLIP 隐藏层输出（Hidden Layer）或最终层输出（Final Layer）作为条件输入。可从 HuggingFace 上下载作者开源的，在受试 1 上训练得到的模型参数。
https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main/mindeye_models

##### 2.3 Stable Diffusion

Stable Diffusion 主要用于 MindEye 的低级管道（Low-Level Pipeline），以 img2img 方式为图像重建提供结构与低频视觉信息。本实验同样直接使用作者在 HuggingFace 上开源的官方模型权重。
https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main/mindeye_models

#### 3 已完成的复现实验
##### 3.1 图像检索与脑检索

我成功复现了 MindEye 在 图像检索（Image Retrieval） 与 脑检索（Brain Retrieval） 任务上的实验结果。
下图展示了原论文中的实验结果：

<img width="711" height="500" alt="image" src="https://github.com/user-attachments/assets/5f5d336c-ee98-49b4-ab82-c60d0c8d75bf" />

下图为我在 Subject 1 上复现得到的实验结果，其中包括

<img width="711" height="250" alt="image-1" src="https://github.com/user-attachments/assets/6e8273b8-ea54-4804-a376-b064f43a030a" />

1. 前向检索（Forward Retrieval）
<img width="711" height="800" alt="image-2" src="https://github.com/user-attachments/assets/0870d0ab-6cc9-471f-b17e-a392b70dd232" />

2. 后向检索（Backward Retrieval）
<img width="711" height="800" alt="image-3" src="https://github.com/user-attachments/assets/225aa7ae-7cb4-4b0a-87c6-1287372350e5" />

从 Top-1 准确率 可以看出，本实验结果与原论文报告的性能基本一致。在视觉高度相似的候选图像集合中，模型仍能够稳定地检索出对应的目标图像，验证了 Voxel2CLIP 映射的有效性。

##### 3.2 fMRI → 图像重建

本实验完整复现了 MindEye 的 fMRI 到图像重建流程。
下图为原论文中的重建结果示例：

<img width="711" height="300" alt="image-8" src="https://github.com/user-attachments/assets/fa826ca7-0419-4872-a764-d05d6dea9535" />

下图为我在 Subject 1 上得到的重建结果：

<img width="711" height="600" alt="image-4" src="https://github.com/user-attachments/assets/868a6cde-0500-426f-ac3f-c5b2d611dcdb" />

可以观察到，生成图像在整体结构、语义类别及主要视觉特征上与真实图像高度一致，整体重建质量达到了原论文所展示的实验水平。
受时间限制，本次实验未进一步复现论文中其他被试或其他对比方法的重建结果。

##### 3.3 消融实验

我复现了原论文中的消融实验第一部分，原论文的表格如下：

<img width="711" height="300" alt="image-9" src="https://github.com/user-attachments/assets/ae5aa6b1-06e6-4c69-93a4-ec97cb301e29" />


受时间限制，本实验仅在 Subject 1 上进行了三组实验。其中：
1. MindEye（Low-Level）表示图像重建从低级管道生成的模糊结构图像开始。

<img width="711" height="600" alt="image-5" src="https://github.com/user-attachments/assets/a9877ae2-ceb8-4fda-9f2d-7d76de2a6bda" />

2. MindEye（High-Level）表示图像重建仅基于高级语义管道进行。

<img width="711" height="600" alt="image-6" src="https://github.com/user-attachments/assets/1208fec7-08e5-494b-a24b-0b83bfcf2816" />


根据原论文使用的评价指标，我得到的定量实验结果如下：

<img width="711" height="150" alt="image" src="https://github.com/user-attachments/assets/82092a6a-bb69-4ab4-aa82-dee71831ac5a" />

整体趋势与原论文保持一致，低级管道在结构与低级视觉指标上具有明显优势，而高级管道更偏向语义一致性。

##### 4 未完全复现的实验

为完整复现原论文其余消融实验，需要从头训练多种不同结构与配置的模型，包括：

<img width="711" height="180" alt="image-10" src="https://github.com/user-attachments/assets/42c57696-753b-4018-996a-a6e1c78e6dc5" />

<img width="711" height="210" alt="image-11" src="https://github.com/user-attachments/assets/15c585b3-7620-4e04-9b19-2dc74549c668" />

<img width="711" height="170" alt="image-12" src="https://github.com/user-attachments/assets/0eafada3-5f9e-47ff-bb33-205a943304dd" />

然而，MindEye 的主干网络参数规模约为 9.4 亿，且映射至 CLIP 隐藏层（257 × 768）后，显著增加了计算量与显存占用。

本实验主要在 8 卡 × 24GB 显存 的学校服务器上进行。实际测试发现，多 GPU 数据并行并不会降低单卡显存占用，仅能提升吞吐率；即使启用 fp16 训练，整体训练过程依然极其耗时。

我也尝试在 AutoDL 算力云平台租用 单卡 48GB 显存 的服务器进行实验，但单个模型配置的完整训练时间仍超过 10 小时。在课程/项目截止日期前，难以完成全部消融实验的复现。

#### 总结
本次复现实验在 单被试（Subject 1）条件下 成功复现了 MindEye 的核心实验结果，包括检索任务、图像重建任务以及部分消融实验。实验结果在定性与定量层面均与原论文保持一致，验证了 MindEye 方法的有效性。
由于模型规模与计算资源限制，其余消融实验未能在限定时间内完成。


# UniMRSeg 实验流水线（Pipeline）

本文档基于 `Paper/UniMRSeg.pdf` 与 `medical/` 目录下的实现，按论文中的“输入级补偿–特征级对齐–输出级一致性”（对应论文框架图）展开，逐阶段给出可复现的训练 / 推理流水线。内容涵盖预处理、三阶段训练策略、模型结构、损失建模（含公式）、以及推理与后处理，便于对照论文逐步落地。

## 总览：三阶段层次化补偿（与论文对应）
- **Stage 1 完整模态重建 (main-ssl1.py，UNet3D_3DASPP_ssl1；论文输入级补偿)**  
  输入完整 4 通道 MRI，通过重建自身获得模态鲁棒的底层表征，为后续缺失模态提供教师信号。
- **Stage 2 不完整模态对比 + 分割预热 (main-ssl2.py，UNet3D_3DASPP_ssl2_new；论文特征级补偿)**  
  同一输入生成两路增强特征，使用 NT-Xent 拉近模态特征分布，并用 Dice 对输出进行分割预热，强化模态不变性。
- **Stage 3 任意模态组合自适应分割 (main-ssl3.py，UNet3D_3DASPP_ssl3 / UniMRSeg；论文输出级补偿)**  
  显式枚举 15 种输入（1 个完整 + 14 个不完整组合；2⁴−1 种缺失方式，跳过“全部缺失”无效情形），以完整分支为教师，对不完整分支进行特征与输出蒸馏，实现任意模态的统一分割。

> 对应论文层次：Stage1（输入级重建）、Stage2（特征级对比对齐）、Stage3（输出级一致性/蒸馏）。

## 数据预处理
### Stage 1（`utils/data_utils_ssl1.py`）
1. `LoadImaged` 读取 `image`/`label`。
2. `ConvertToMultiChannelBasedOnBratsClassesD` 将标签转为 (TC, WT, ET) 三通道。
3. `CropForegroundd` & `RandSpatialCropd` 裁剪至 `[128,128,128]`（保持体素对齐）。
4. 直接 `ToTensord`（不做归一化/增强，聚焦重建任务，符合论文“输入级”对原始模态的保持）。

### Stage 2 & 3（`utils/data_utils.py`）
1. 与 Stage 1 相同的读取与裁剪。
2. 数据增强：三轴随机翻转、强度随机缩放/平移。
3. `NormalizeIntensityd`（非零、按通道）确保模态分布一致。
4. `ToTensord`。

## 网络结构概要
- Stage 1/2 采用 **3D UNet + 3D ASPP** 主干。
- Stage 3 使用 **UniMRSeg**：包含
  - **Encoder_Complete**：完整模态路径，提取多尺度特征 `e1..e5`。
  - **SSL_Adaptor**：对缺失模态输入进行补偿编码，与完整分支特征对齐。
  - **Decoder**：共享解码器，实现跨模态统一输出。
  - 模型内部使用分块注意力与多尺度上下文（见 `medical/model/UniMRSeg.py`），保持 3D 体素一致性。

> 对照论文：输入级（保持原始模态）、特征级（跨模态特征对齐）、输出级（统一解码与一致性约束）。

## 损失与数学建模
### Stage 1：重建自监督
1. 归一化目标（逐体素最小-最大）  
   \[
   x^{*} = \frac{x - \min(x)}{\max(x)-\min(x)+\epsilon}
   \]
2. 总损失（SSIM + L1，对应 `loss_func1/2`）：  
   \[
   \mathcal{L}_{\text{ssl1}} = \underbrace{\text{SSIM}(f_\theta(x), x^{*})}_{\mathcal{L}_1} + \underbrace{\|f_\theta(x)-x^{*}\|_1}_{\mathcal{L}_2}
   \]

### Stage 2：模态不变对比 + 分割
1. **NT-Xent 对比损失**（`nt_xent_loss`，温度 \(\tau\)）：代码中先将两路增强输出 `(out_1, out_2)` 沿 batch 维拼成四个向量 \([z_0,z_1,z_2,z_3]\)，正对为 (0,1) 与 (2,3)，对角与正对以外均为负样本，特征经 `F.normalize`：  
   \[
   \mathcal{L}_{\text{NTX}} = -\frac{1}{N}\!\sum_{i\in\{0,1,2,3\}}\! \log \frac{\exp(\langle z_i,z_{p(i)}\rangle / \tau)}{\sum_{k\neq i}\exp(\langle z_i,z_k\rangle / \tau)},\;
   p(0)=1,p(1)=0,p(2)=3,p(3)=2,\; N=4
   \]
   这与实现中的 `loss = -log(pos/neg.sum).mean()` 等价，仅由样本数求平均。
2. **Dice 分割损失**（针对双路拼接的标签）：  
   \[
   \mathcal{L}_{\text{dice}} = 1 - \frac{2\sum p y}{\sum p + \sum y + \epsilon}
   \]
3. 总损失（代码中按尺度平均）：  
   \[
   \mathcal{L}_{\text{ssl2}} = \frac{1}{K}\sum_{k=1}^{K}\mathcal{L}_{\text{NTX}}^{(k)} + \mathcal{L}_{\text{dice}}
   \]

### Stage 3：任意模态补偿分割
对 1 个完整 + 14 个不完整输入组合 \(x^{(m)}\)（通过将缺失模态置零得到，跳过全零输入）：
1. **完整路径监督**（仅完整输入 \(x^{(0)}\)）：  
   \[
   \mathcal{L}_{\text{seg}}^{c} = \mathcal{L}_{\text{dice}}(\hat{y}^{c}, y)
   \]
2. **不完整路径蒸馏**（对每个组合 \(m>0\)）：  
   - 预测蒸馏（与冻结的完整输出对齐）：  
     \[
     \mathcal{L}_{\text{logit}}^{(m)} = \mathcal{L}_{\text{dice}}\!\left(\hat{y}^{(m)}, \sigma(\hat{y}^{c})\right)
     \]
   - 特征对齐（5 个层级 L1，代码系数 0.2）：  
     \[
     \mathcal{L}_{\text{feat}}^{(m)} = \lambda \sum_{l=1}^{5} \| e_l^{(m)} - e_l^{c} \|_1,\quad \lambda=0.2
     \]
3. **总损失**：  
   \[
   \mathcal{L}_{\text{ssl3}} = \mathcal{L}_{\text{seg}}^{c} + \sum_{m=1}^{14}\left(\mathcal{L}_{\text{logit}}^{(m)} + \mathcal{L}_{\text{feat}}^{(m)}\right)
   \]

## 训练与推理流程
1. **Stage 1（论文输入级）**：训练重建网络，保存 `model_final_ssl1.pt`，输出为后续教师特征。
2. **Stage 2（论文特征级）**：加载 Stage 1 权重，启用对比 + Dice 监督，得到 `model_final_ssl2.pt`，实现模态不变特征。
3. **Stage 3（论文输出级）**：加载 Stage 2 权重，对 1 个完整 + 14 个不完整输入进行蒸馏训练，生成统一模型 `model_final_ssl3.pt`，保证任意模态输出一致。
4. **推理**（`test.py` / `tester_all_input.py`）：  
   - 预处理同 Stage 2/3。  
   - **完整模态**调用 `forward_complete`，不完整模态调用 `forward_uncomplete`（枚举或按实际缺失置零）。  
   - `SlidingWindowInferer`（默认 `overlap=0.5`/`0.25`）分块推理。

### 逐阶段实操（命令行与论文对应）
- Stage 1（输入级重建）：`python main-ssl1.py ...`（保持默认数据增强为关闭，契合输入级重建假设）。
- Stage 2（特征级对齐）：`python main-ssl2.py --checkpoint model_final_ssl1.pt ...`（加载 Stage1，开启对比 + Dice）。
- Stage 3（输出级一致性）：`python main-ssl3.py --checkpoint model_final_ssl2.pt ...`（完整 / 不完整双分支蒸馏）。
- 测试：`python test.py --checkpoint model_final_ssl3.pt ...`；如需显式评估各不完整组合，可使用 `tester_all_input.py`。

## 后处理
1. `sigmoid` 概率映射后按阈值 0.5 二值化（`post_pred_func`）。
2. 输出为 (TC, WT, ET) 三类掩膜，可按需重采样回原始空间。

## 复现提示
- 模态顺序为 4 通道 MRI（如 T1/T1ce/T2/FLAIR），缺失模态以零填充实现。
- 训练超参以脚本默认值为准（AdamW, warmup-cosine, batch size 2，最大 300 epoch）。
- 保持与 JSON 中的 fold 划分一致以匹配论文结果。

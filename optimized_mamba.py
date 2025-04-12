import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mamba_ssm import Mamba
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from einops import rearrange
from astropy.io import fits
import glob
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings("ignore")

# 检查CUDA可用性
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# 优化1: 针对FITS文件的Aurora极光图像数据集
class AuroraFITSDataset(Dataset):
    def __init__(self, fits_dir, patch_size=128, overlap=32, transform=None, limit_files=None):
        """
        fits_dir: FITS文件目录
        patch_size: 图像块大小
        overlap: 图像块重叠大小
        transform: 数据变换
        limit_files: 限制文件数量(测试时使用)
        """
        self.fits_dir = fits_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform

        # 获取所有FITS文件路径
        self.fits_files = sorted(glob.glob(os.path.join(fits_dir, "*.fits")))
        if limit_files is not None:
            self.fits_files = self.fits_files[:limit_files]

        print(f"找到 {len(self.fits_files)} 个FITS文件")

        # 预先计算每个文件的可用块数量
        self.file_patches = []
        for fits_file in tqdm(self.fits_files, desc="分析FITS文件"):
            try:
                with fits.open(fits_file) as hdul:
                    # 检查FITS文件结构
                    # 假设主数据在第一个扩展或主HDU
                    data = None
                    for hdu in hdul:
                        if hasattr(hdu, 'data') and hdu.data is not None:
                            data = hdu.data
                            break

                    if data is None:
                        print(f"警告: 无法从 {fits_file} 读取数据")
                        continue

                    # 确定数据维度 - 处理不同的FITS格式
                    shape = data.shape
                    if len(shape) == 2:  # 单波段图像
                        height, width = shape
                        bands = 1
                    elif len(shape) == 3:  # 多波段图像
                        if shape[0] <= 10:  # 通常波段数较少
                            bands, height, width = shape
                        else:  # 也可能是时间序列数据
                            height, width, bands = shape
                    else:
                        print(f"警告: 不支持的数据维度 {shape} 在文件 {fits_file}")
                        continue

                    # 计算此文件的图像块数量
                    h_patches = max(1, (height - self.patch_size) // (self.patch_size - self.overlap) + 1)
                    w_patches = max(1, (width - self.patch_size) // (self.patch_size - self.overlap) + 1)
                    total_patches = h_patches * w_patches

                    self.file_patches.append({
                        'file': fits_file,
                        'shape': (bands, height, width),
                        'h_patches': h_patches,
                        'w_patches': w_patches,
                        'total_patches': total_patches
                    })
            except Exception as e:
                print(f"处理文件 {fits_file} 时出错: {str(e)}")

        # 计算总图像块数量
        self.total_patches = sum(fp['total_patches'] for fp in self.file_patches)
        print(f"总共提取了 {self.total_patches} 个图像块")

        # 创建索引映射，用于快速定位图像块
        self.index_map = []
        offset = 0
        for file_info in self.file_patches:
            for h in range(file_info['h_patches']):
                for w in range(file_info['w_patches']):
                    self.index_map.append({
                        'file_idx': self.file_patches.index(file_info),
                        'h_idx': h,
                        'w_idx': w,
                        'global_idx': offset
                    })
                    offset += 1

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        # 找到对应的文件和位置
        map_info = self.index_map[idx]
        file_info = self.file_patches[map_info['file_idx']]
        fits_file = file_info['file']
        h_idx = map_info['h_idx']
        w_idx = map_info['w_idx']

        # 计算图像块位置
        h_step = self.patch_size - self.overlap
        w_step = self.patch_size - self.overlap

        h_start = h_idx * h_step
        w_start = w_idx * w_step

        # 打开FITS文件并读取数据
        with fits.open(fits_file) as hdul:
            # 找到数据所在的HDU
            data = None
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    data = hdu.data
                    break

            if data is None:
                raise ValueError(f"无法从 {fits_file} 读取数据")

            # 根据数据维度提取图像块
            shape = data.shape
            if len(shape) == 2:  # 单波段图像
                patch = data[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
                patch = np.expand_dims(patch, axis=0)  # 添加波段维度
            elif len(shape) == 3:
                if shape[0] <= 10:  # 通常波段在第一维
                    patch = data[:, h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
                else:  # 波段可能在最后一维
                    patch = data[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size, :]
                    patch = np.transpose(patch, (2, 0, 1))  # 调整为 [bands, height, width]

            # 归一化数据到 [0, 1]
            patch_min = np.min(patch)
            patch_max = np.max(patch)
            if patch_max > patch_min:
                patch = (patch - patch_min) / (patch_max - patch_min)
            else:
                patch = np.zeros_like(patch)

            # 处理NaN和Inf值
            patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)

            # 转为torch张量
            patch_tensor = torch.from_numpy(patch).float()

            if self.transform:
                patch_tensor = self.transform(patch_tensor)

            # 返回元数据用于可能的重建
            metadata = {
                'file': fits_file,
                'h_start': h_start,
                'w_start': w_start,
                'original_shape': shape,
                'min_val': patch_min,
                'max_val': patch_max
            }

            return patch_tensor, metadata


# 优化2: 自适应选择性状态空间模型
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state, seq_len=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # SSM核心参数
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state))
        self.C = nn.Parameter(torch.randn(d_state))

        # 动态参数生成网络
        self.delta_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )

        self.B_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_state),
            nn.Tanh()
        )

        self.C_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_state),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        返回: [batch_size, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # 动态生成Delta、B和C参数
        delta = self.delta_net(x)  # [batch, seq_len, 1]
        B_dyn = self.B_net(x)  # [batch, seq_len, d_state]
        C_dyn = self.C_net(x)  # [batch, seq_len, d_state]

        # 初始化状态
        h = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []

        # 实现并行扫描算法
        for t in range(seq_len):
            # 选择性状态更新 - 基于当前输入调整状态传播
            delta_t = delta[:, t, :]  # [batch, 1]
            B_t = B_dyn[:, t, :] * self.B  # [batch, d_state]
            C_t = C_dyn[:, t, :] * self.C  # [batch, d_state]

            # 状态更新: h_t = exp(A*Δ)*h_{t-1} + B_t*x_t
            h = h * torch.exp(delta_t * torch.diag(self.A)) + B_t * x[:, t, :].unsqueeze(-1)

            # 输出: y_t = C_t*h_t
            y = (C_t * h).sum(dim=1)  # [batch, d_model]
            outputs.append(y)

        return torch.stack(outputs, dim=1)  # [batch, seq_len, d_model]


# 优化3: 自适应双阈值残差编码器
class AdaptiveResidualEncoder(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()

        # 阈值生成网络
        self.threshold_net = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),  # 生成两个阈值
            nn.Softplus()  # 确保阈值为正
        )

    def forward(self, residual, prediction):
        """
        实现自适应双阈值编码
        residual: 原始残差
        prediction: 预测值
        """
        # 基于预测值动态生成阈值
        thresholds = self.threshold_net(prediction)
        lower_th = thresholds[:, 0:1, :, :]  # 低阈值，小于此值视为背景/噪声
        upper_th = thresholds[:, 1:2, :, :]  # 高阈值，大于此值视为显著信号

        # 构建掩码
        background_mask = (torch.abs(residual) < lower_th).float()
        signal_mask = (torch.abs(residual) > upper_th).float()
        medium_mask = 1.0 - background_mask - signal_mask

        # 自适应量化
        quantized_residual = torch.zeros_like(residual)

        # 背景区域直接置零
        # 信号区域保持原值
        quantized_residual = signal_mask * residual

        # 中等区域进行非线性量化
        medium_values = medium_mask * residual
        # 使用非线性函数进行平滑量化
        medium_quantized = torch.sign(medium_values) * torch.round(torch.abs(medium_values) * 127) / 127.0
        quantized_residual = quantized_residual + medium_mask * medium_quantized

        # 计算量化误差
        quant_error = residual - quantized_residual

        return quantized_residual, quant_error, thresholds


# 优化4: Mamba极光图像预测模型
class MambaAuroraPredictor(nn.Module):
    def __init__(self, spectral_bands=1, hidden_dim=128, d_state=64, depth=3):
        super().__init__()
        self.spectral_bands = spectral_bands
        self.hidden_dim = hidden_dim

        # 空间特征提取
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(spectral_bands, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

        # 使用Mamba处理序列
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=4,
                expand=2
            ) for _ in range(depth)
        ])

        # 融合层
        self.fusion_layer = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # 预测层
        self.predictor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, spectral_bands, kernel_size=3, padding=1)
        )

        # 自适应残差编码器
        self.residual_encoder = AdaptiveResidualEncoder(spectral_bands)

    def forward(self, x):
        """
        x: [batch_size, spectral_bands, height, width]
        """
        batch_size, c, h, w = x.shape

        # 1. 空间特征提取
        spatial_features = self.spatial_encoder(x)  # [B, hidden_dim, H, W]

        # 2. 调整为序列，进行处理
        # 先沿着行方向处理
        seq_features_h = rearrange(spatial_features, 'b c h w -> (b w) h c')

        # 通过Mamba层处理行序列
        for mamba_layer in self.mamba_layers:
            seq_features_h = mamba_layer(seq_features_h)

        # 重新整形
        features_h = rearrange(seq_features_h, '(b w) h c -> b c h w', b=batch_size)

        # 再沿着列方向处理
        seq_features_w = rearrange(spatial_features, 'b c h w -> (b h) w c')

        # 通过Mamba层处理列序列
        for mamba_layer in self.mamba_layers:
            seq_features_w = mamba_layer(seq_features_w)

        # 重新整形
        features_w = rearrange(seq_features_w, '(b h) w c -> b c h w', b=batch_size)

        # 融合行列特征
        fused_features = self.fusion_layer(features_h + features_w)

        # 预测
        prediction = self.predictor(fused_features)

        return prediction

    def compress(self, x):
        """
        压缩过程: 预测并编码残差
        返回: 量化后残差，阈值信息
        """
        # 生成预测
        prediction = self(x)

        # 计算残差
        residual = x - prediction

        # 自适应残差处理
        quantized_residual, quant_error, thresholds = self.residual_encoder(residual, prediction)

        return prediction, quantized_residual, thresholds

    def decompress(self, prediction, quantized_residual):
        """
        解压过程: 将预测值与残差相加
        """
        # 重建原始图像
        reconstructed = prediction + quantized_residual
        return reconstructed


# 优化5: GPU加速训练过程
def train_mamba_aurora(model, dataloader, epochs=10, lr=1e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler()  # 混合精度训练

    # 损失函数组合
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_time = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (data, _) in enumerate(progress_bar):
            start_time = time.time()
            data = data.to(device)

            # 内存优化：定期清理GPU缓存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # 混合精度训练
            with autocast():
                # 前向传播，获取预测值和压缩表示
                prediction, quantized_residual, thresholds = model.compress(data)

                # 重建图像
                reconstructed = model.decompress(prediction, quantized_residual)

                # 计算损失 - 多目标
                recon_loss = mse_loss(reconstructed, data)  # 重建损失
                prediction_loss = l1_loss(prediction, data)  # 预测损失
                sparsity_loss = 0.01 * torch.mean(torch.abs(quantized_residual))  # 稀疏性损失

                # 总损失
                loss = recon_loss + 0.1 * prediction_loss + sparsity_loss

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_time = time.time() - start_time
            total_time += batch_time
            total_loss += loss.item()

            # 计算PSNR和零元素比例
            with torch.no_grad():
                mse = torch.mean((reconstructed - data) ** 2)
                psnr = 10 * torch.log10(1.0 / mse).item()
                zero_ratio = (quantized_residual == 0).float().mean().item() * 100

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'PSNR': f"{psnr:.2f}dB",
                'zeros': f"{zero_ratio:.1f}%",
                'time': f"{batch_time:.3f}s"
            })

        scheduler.step()

        # 每个epoch结束打印统计信息
        avg_loss = total_loss / len(dataloader)
        avg_time = total_time / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.6f}, "
              f"Avg. Batch Time: {avg_time:.4f}s, LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"mamba_aurora_epoch_{epoch + 1}.pth")


# 优化6: FITS文件压缩函数
def compress_aurora_fits(model, input_file, output_file, device=None, patch_size=128, overlap=64):
    """压缩单个FITS文件"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    start_total = time.time()

    # 读取FITS文件
    with fits.open(input_file) as hdul:
        # 找到数据所在的HDU
        data = None
        header = None
        for hdu in hdul:
            if hasattr(hdu, 'data') and hdu.data is not None:
                data = hdu.data
                header = hdu.header
                break

        if data is None:
            raise ValueError(f"无法从 {input_file} 读取数据")

        # 确定数据维度
        shape = data.shape
        if len(shape) == 2:  # 单波段图像
            height, width = shape
            data = np.expand_dims(data, axis=0)  # [1, H, W]
            bands = 1
        elif len(shape) == 3:
            if shape[0] <= 10:  # 通常波段在第一维
                bands, height, width = shape
            else:  # 波段可能在最后一维
                height, width, bands = shape
                data = np.transpose(data, (2, 0, 1))  # 调整为 [bands, height, width]

        # 原始数据范围
        data_min = np.min(data)
        data_max = np.max(data)

        # 归一化数据
        if data_max > data_min:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = np.zeros_like(data)

        # 处理NaN和Inf
        normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=1.0, neginf=0.0)

        # 创建保存压缩结果的数组
        compressed_predictions = np.zeros_like(normalized_data)
        compressed_residuals = np.zeros_like(normalized_data)
        compressed_thresholds = np.zeros((2, height, width))  # 保存两个阈值

        # 分块处理
        h_steps = max(1, (height - patch_size) // (patch_size - overlap) + 1)
        w_steps = max(1, (width - patch_size) // (patch_size - overlap) + 1)

        # 权重图用于重叠区域的合并
        weights = np.zeros((bands, height, width))

        print(f"处理文件: {input_file}, 形状: {shape}, 分为 {h_steps}x{w_steps} 个块")

        with torch.no_grad():
            # 逐块处理
            for h_idx in tqdm(range(h_steps), desc="压缩FITS文件"):
                for w_idx in range(w_steps):
                    # 计算当前块的位置
                    h_start = min(h_idx * (patch_size - overlap), height - patch_size)
                    w_start = min(w_idx * (patch_size - overlap), width - patch_size)
                    h_end = h_start + patch_size
                    w_end = w_start + patch_size

                    # 提取块
                    block = normalized_data[:, h_start:h_end, w_start:w_end]

                    # 转为张量
                    block_tensor = torch.from_numpy(block).float().unsqueeze(0).to(device)

                    # 压缩
                    prediction, quantized_residual, thresholds = model.compress(block_tensor)

                    # 转回numpy
                    prediction_np = prediction.squeeze(0).cpu().numpy()
                    residual_np = quantized_residual.squeeze(0).cpu().numpy()
                    thresholds_np = thresholds.squeeze(0).cpu().numpy()

                    # 创建权重 - 使用汉宁窗减少边缘效应
                    h_window = np.hanning(patch_size)
                    w_window = np.hanning(patch_size)
                    hw_window = h_window[:, np.newaxis] * w_window[np.newaxis, :]
                    block_weight = np.ones((bands, patch_size, patch_size))
                    for b in range(bands):
                        block_weight[b] = hw_window

                    # 累加结果和权重
                    compressed_predictions[:, h_start:h_end, w_start:w_end] += prediction_np * block_weight
                    compressed_residuals[:, h_start:h_end, w_start:w_end] += residual_np * block_weight
                    compressed_thresholds[:, h_start:h_end, w_start:w_end] += thresholds_np * hw_window
                    weights[:, h_start:h_end, w_start:w_end] += block_weight

        # 加权平均
        nonzero_weights = (weights > 0)
        compressed_predictions[nonzero_weights] /= weights[nonzero_weights]
        compressed_residuals[nonzero_weights] /= weights[nonzero_weights]
        nonzero_weights_th = (weights[0] > 0)  # 只需第一个通道的权重
        compressed_thresholds[:, nonzero_weights_th] /= weights[0, nonzero_weights_th]

        # 重建图像
        reconstructed = compressed_predictions + compressed_residuals

        # 反归一化
        reconstructed = reconstructed * (data_max - data_min) + data_min
        compressed_predictions = compressed_predictions * (data_max - data_min) + data_min

        # 计算压缩统计信息
        compression_time = time.time() - start_total

        # 计算压缩比 - 假设原始和压缩都是浮点
        non_zero_residuals = np.count_nonzero(compressed_residuals)
        original_size = np.prod(data.shape) * 4  # 原始数据 (32位浮点)
        metadata_size = (compressed_thresholds.size + 10) * 4  # 元数据和阈值 (32位浮点)
        compressed_size = non_zero_residuals * 1 + bands * 4 + metadata_size  # 残差用8位表示
        compression_ratio = original_size / compressed_size

        # 计算PSNR
        mse = np.mean((data - reconstructed) ** 2)
        if mse > 0:
            psnr = 20 * np.log10((data_max - data_min) / np.sqrt(mse))
        else:
            psnr = float('inf')

        print(f"压缩比: {compression_ratio:.2f}x, PSNR: {psnr:.2f} dB, 时间: {compression_time:.2f}s")

        # 保存压缩结果
        with fits.open(input_file) as hdul:
            # 创建新的FITS文件
            compressed_hdul = fits.HDUList()

            # 添加主HDU
            primary_hdu = fits.PrimaryHDU(header=hdul[0].header)
            compressed_hdul.append(primary_hdu)

            # 添加压缩数据
            # 1. 非零残差位置和值
            residual_mask = (compressed_residuals != 0)
            residual_positions = np.where(residual_mask)
            residual_values = compressed_residuals[residual_mask]

            # 量化残差值到8位
            quantized_values = np.round(residual_values * 127).astype(np.int8)

            # 创建数据表HDU
            col1 = fits.Column(name='band_idx', format='I', array=residual_positions[0])
            col2 = fits.Column(name='row_idx', format='I', array=residual_positions[1])
            col3 = fits.Column(name='col_idx', format='I', array=residual_positions[2])
            col4 = fits.Column(name='residual', format='B', array=quantized_values)

            residual_table = fits.BinTableHDU.from_columns([col1, col2, col3, col4])
            residual_table.header['EXTNAME'] = 'RESIDUALS'
            compressed_hdul.append(residual_table)

            # 2. 预测模型参数 - 这里假设模型参数会单独保存
            # 但我们需要记录一些用于重建的元数据
            metadata_hdu = fits.ImageHDU(np.array([
                data_min, data_max,
                compression_ratio, psnr,
                bands, height, width,
                patch_size, overlap
            ]))
            metadata_hdu.header['EXTNAME'] = 'METADATA'
            compressed_hdul.append(metadata_hdu)

            # 3. 阈值图 - 用于可能的调试和分析
            threshold_hdu = fits.ImageHDU(compressed_thresholds)
            threshold_hdu.header['EXTNAME'] = 'THRESHOLDS'
            compressed_hdul.append(threshold_hdu)

            # 4. 保存预测图像 - 用于可能的调试
            prediction_hdu = fits.ImageHDU(compressed_predictions)
            prediction_hdu.header['EXTNAME'] = 'PREDICTION'
            compressed_hdul.append(prediction_hdu)

            # 5. 添加重建的图像 - 用于可能的调试或快速加载
            reconstructed_hdu = fits.ImageHDU(reconstructed)
            reconstructed_hdu.header['EXTNAME'] = 'RECONSTRUCTED'
            compressed_hdul.append(reconstructed_hdu)

            # 写入FITS文件
            compressed_hdul.writeto(output_file, overwrite=True)

    results = {
        'file': input_file,
        'shape': shape,
        'original_size': original_size / (1024 * 1024),  # MB
        'compressed_size': compressed_size / (1024 * 1024),  # MB
        'compression_ratio': compression_ratio,
        'psnr': psnr,
        'time': compression_time,
        'zero_ratio': 100 - (non_zero_residuals / np.prod(data.shape) * 100)  # 零元素百分比
    }

    return results


# 优化7: 解压缩FITS文件
def decompress_aurora_fits(model, compressed_file, output_file=None, device=None):
    """解压缩FITS文件"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    start_time = time.time()

    # 读取压缩文件
    with fits.open(compressed_file) as hdul:
        # 读取元数据
        metadata = hdul['METADATA'].data
        data_min, data_max = metadata[0], metadata[1]
        bands, height, width = int(metadata[4]), int(metadata[5]), int(metadata[6])

        # 读取预测图
        prediction = hdul['PREDICTION'].data

        # 读取残差表
        residual_table = hdul['RESIDUALS'].data
        band_indices = residual_table['band_idx']
        row_indices = residual_table['row_idx']
        col_indices = residual_table['col_idx']
        residual_values = residual_table['residual'].astype(np.float32) / 127.0

        # 重建残差图
        residual = np.zeros((bands, height, width), dtype=np.float32)
        for i in range(len(band_indices)):
            residual[band_indices[i], row_indices[i], col_indices[i]] = residual_values[i]

        # 重建原始图像
        reconstructed = prediction + residual

        # 反归一化
        if output_file:
            # 将重建结果保存为新FITS文件
            with fits.open(compressed_file) as template_hdul:
                # 复制主HDU的头信息
                primary_hdu = fits.PrimaryHDU(header=template_hdul[0].header)

                # 创建图像HDU
                image_hdu = fits.ImageHDU(reconstructed)
                for key in template_hdul[0].header:
                    if key not in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'EXTEND']:
                        image_hdu.header[key] = template_hdul[0].header[key]

                # 创建新FITS文件
                hdul_out = fits.HDUList([primary_hdu, image_hdu])
                hdul_out.writeto(output_file, overwrite=True)

    decompress_time = time.time() - start_time
    print(f"解压时间: {decompress_time:.4f}s")

    return reconstructed, decompress_time


# 优化8: 批量处理FITS文件
def batch_process_fits(model, input_dir, output_dir, device=None, limit=None):
    """批量处理目录中的FITS文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取所有FITS文件
    fits_files = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
    if limit:
        fits_files = fits_files[:limit]

    results = []

    for fits_file in tqdm(fits_files, desc="批量处理FITS文件"):
        file_name = os.path.basename(fits_file)
        output_file = os.path.join(output_dir, f"compressed_{file_name}")

        try:
            result = compress_aurora_fits(model, fits_file, output_file, device)
            results.append(result)
        except Exception as e:
            print(f"处理 {fits_file} 时出错: {str(e)}")

    # 输出汇总结果
    if results:
        avg_ratio = np.mean([r['compression_ratio'] for r in results])
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_time = np.mean([r['time'] for r in results])
        avg_zero = np.mean([r['zero_ratio'] for r in results])

        print("\n批处理结果汇总:")
        print(f"处理文件数: {len(results)}")
        print(f"平均压缩比: {avg_ratio:.2f}x")
        print(f"平均PSNR: {avg_psnr:.2f} dB")
        print(f"平均处理时间: {avg_time:.2f}s")
        print(f"平均零元素比例: {avg_zero:.2f}%")

        # 保存结果到CSV
        with open(os.path.join(output_dir, "compression_results.csv"), "w") as f:
            f.write("文件,原始大小(MB),压缩大小(MB),压缩比,PSNR(dB),处理时间(s),零元素比例(%)\n")
            for r in results:
                f.write(f"{r['file']},{r['original_size']:.2f},{r['compressed_size']:.2f},"
                        f"{r['compression_ratio']:.2f},{r['psnr']:.2f},{r['time']:.2f},{r['zero_ratio']:.2f}\n")

    return results


# 可视化FITS文件处理结果
def visualize_fits_compression(original_file, compressed_file, bands_to_show=None):
    """可视化FITS文件压缩效果"""
    # 读取原始FITS文件
    with fits.open(original_file) as hdul:
        # 找到数据所在的HDU
        original_data = None
        for hdu in hdul:
            if hasattr(hdu, 'data') and hdu.data is not None:
                original_data = hdu.data
                break

        if original_data is None:
            raise ValueError(f"无法从 {original_file} 读取数据")

        # 标准化数据形状
        if len(original_data.shape) == 2:
            original_data = np.expand_dims(original_data, axis=0)  # 添加波段维度

    # 读取压缩文件
    with fits.open(compressed_file) as hdul:
        # 读取元数据
        metadata = hdul['METADATA'].data
        compression_ratio, psnr = metadata[2], metadata[3]

        # 读取预测和重建数据
        prediction = hdul['PREDICTION'].data
        reconstructed = hdul['RECONSTRUCTED'].data

        # 读取阈值数据
        thresholds = hdul['THRESHOLDS'].data

    # 确定要显示的波段
    num_bands = original_data.shape[0]
    if bands_to_show is None:
        if num_bands == 1:
            bands_to_show = [0]
        elif num_bands <= 3:
            bands_to_show = list(range(num_bands))
        else:
            # 选择前中后三个波段
            bands_to_show = [0, num_bands // 2, num_bands - 1]

    # 创建可视化图形
    num_cols = len(bands_to_show)
    fig, axes = plt.subplots(5, num_cols, figsize=(num_cols * 4, 20))

    # 如果只有一个波段，确保axes是二维的
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    # 标题
    fig.suptitle(f'极光FITS图像压缩结果 (压缩比: {compression_ratio:.2f}x, PSNR: {psnr:.2f}dB)', fontsize=16)

    # 行标题
    row_titles = ['原始图像', '预测图像', '重建图像', '残差(x5)', '阈值图']

    for col, band in enumerate(bands_to_show):
        # 原始图像
        im0 = axes[0, col].imshow(original_data[band], cmap='viridis')
        axes[0, col].set_title(f'波段 {band}')
        fig.colorbar(im0, ax=axes[0, col])

        # 预测图像
        im1 = axes[1, col].imshow(prediction[band], cmap='viridis')
        fig.colorbar(im1, ax=axes[1, col])

        # 重建图像
        im2 = axes[2, col].imshow(reconstructed[band], cmap='viridis')
        fig.colorbar(im2, ax=axes[2, col])

        # 计算并显示残差 (放大5倍)
        residual = (original_data[band] - reconstructed[band]) * 5
        im3 = axes[3, col].imshow(residual, cmap='coolwarm')
        fig.colorbar(im3, ax=axes[3, col])

        # 显示阈值图 (第一个通道为低阈值，第二个为高阈值)
        if col == 0:
            threshold_low = thresholds[0]
            im4 = axes[4, col].imshow(threshold_low, cmap='hot')
            axes[4, col].set_title('低阈值')
            fig.colorbar(im4, ax=axes[4, col])
        else:
            threshold_high = thresholds[1]
            im4 = axes[4, col].imshow(threshold_high, cmap='hot')
            axes[4, col].set_title('高阈值')
            fig.colorbar(im4, ax=axes[4, col])

    # 设置行标题
    for i, title in enumerate(row_titles):
        axes[i, 0].set_ylabel(title, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.splitext(os.path.basename(original_file))[0] + '_compression_result.png', dpi=300)
    plt.show()


# 主函数
def main():
    # 设置参数
    fits_dir = "aurora_fits"  # FITS文件目录
    output_dir = "compressed_fits"  # 压缩输出目录
    model_file = "mamba_aurora_model.pth"  # 模型保存路径

    batch_size = 4
    epochs = 10
    patch_size = 128
    overlap = 32

    # 检测FITS文件中的波段数
    sample_fits = glob.glob(os.path.join(fits_dir, "*.fits"))[0]
    with fits.open(sample_fits) as hdul:
        for hdu in hdul:
            if hasattr(hdu, 'data') and hdu.data is not None:
                data = hdu.data
                break

        # 确定波段数
        if len(data.shape) == 2:
            spectral_bands = 1
        elif len(data.shape) == 3:
            if data.shape[0] <= 10:
                spectral_bands = data.shape[0]
            else:
                spectral_bands = data.shape[2]

    print(f"检测到波段数: {spectral_bands}")

    # 创建数据集
    dataset = AuroraFITSDataset(
        fits_dir=fits_dir,
        patch_size=patch_size,
        overlap=overlap
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型
    model = MambaAuroraPredictor(
        spectral_bands=spectral_bands,
        hidden_dim=128,
        d_state=64,
        depth=3
    )

    # 检查是否已有保存的模型
    if os.path.exists(model_file):
        print(f"加载已有模型: {model_file}")
        checkpoint = torch.load(model_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 训练模型
        print("开始训练模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_mamba_aurora(
            model=model,
            dataloader=dataloader,
            epochs=epochs,
            lr=1e-4,
            device=device
        )

        # 保存最终模型
        torch.save({
            'model_state_dict': model.state_dict(),
        }, model_file)

    # 批量处理FITS文件
    print("开始批量处理FITS文件...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = batch_process_fits(
        model=model,
        input_dir=fits_dir,
        output_dir=output_dir,
        device=device,
        limit=10  # 限制处理文件数量，可根据需要调整
    )

    # 可视化一个样例
    if len(results) > 0:
        sample_file = results[0]['file']
        compressed_file = os.path.join(output_dir, f"compressed_{os.path.basename(sample_file)}")
        visualize_fits_compression(sample_file, compressed_file)


if __name__ == "__main__":
    main()

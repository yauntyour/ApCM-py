from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import zipfile
from net import ApCM
import sys
import logging
from datetime import datetime, timezone

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(
            f"log/{datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),  # 同时输出到终端
    ],
)


def set_seed(seed=42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_data(filepath):
    """
    加载真实数据（用于测试）
    """
    images = []
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        file_list = zip_ref.namelist()
        for file_name in file_list:
            if "png" in file_name:
                # 读取图像
                img = Image.open(BytesIO(zip_ref.read(file_name)))
                img = img.convert("L")
                img_tensor = torch.tensor(np.array(img), dtype=torch.float32)
                images.append(img_tensor)
    return images


def generate_training_images(n=1000, size=(32, 32)):
    """
    生成模拟训练数据
    """
    images = []
    for _ in range(n):
        # 创建带结构的模拟图像（模拟真实图像的局部相关性）
        img = torch.zeros(size)
        # 随机添加3-8个高斯blob，模拟真实图像特征
        num_blobs = torch.randint(3, 9, (1,)).item()
        for _ in range(num_blobs):
            cx = torch.randint(5, size[0] - 5, (1,)).item()
            cy = torch.randint(5, size[1] - 5, (1,)).item()
            x, y = torch.meshgrid(torch.arange(size[0]), torch.arange(size[1]))
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            sigma = torch.randint(2, 5, (1,)).item()  # 随机标准差
            blob = torch.exp(-d2 / (2 * sigma**2))
            img += blob * torch.rand(1) * 0.8  # 随机强度
        # 添加一些随机噪声
        img += torch.randn_like(img) * 0.05
        img = torch.clamp(img, 0, 1)
        images.append(img)
    return images


def preprocess_images(images, target_size=(32, 32), is_flat=False):
    """
    预处理图像：调整大小、展平、归一化

    Args:
        images: 图像列表，每个元素是2D张量或展平的张量
        target_size: 目标尺寸
        is_flat: 输入图像是否已经展平
    """
    processed_images = []
    for img in images:
        if is_flat:
            # 如果已经是展平的，重塑为图像格式
            img_2d = img.view(target_size)
            img_pil = Image.fromarray((img_2d.numpy() * 255).astype(np.uint8), mode="L")
        else:
            # 如果是2D图像，转换为PIL格式
            img_pil = Image.fromarray((img.numpy() * 255).astype(np.uint8), mode="L")

        # 调整大小（如果需要）
        if img_pil.size != target_size:
            img_resized = img_pil.resize(target_size)
        else:
            img_resized = img_pil

        img_tensor = torch.tensor(np.array(img_resized), dtype=torch.float32)

        # 归一化到 [0, 1]
        img_normalized = img_tensor / 255.0

        # 展平
        img_flat = img_normalized.view(-1)
        processed_images.append(img_flat)

    return torch.stack(processed_images)


def psnr(original, reconstructed):
    """
    计算峰值信噪比
    """
    mse = F.mse_loss(original, reconstructed)
    if mse == 0:
        return float("inf")
    max_val = 1.0  # 归一化后的最大值
    psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_val


def visualize_comparison(
    originals,
    reconstructions,
    titles=["Original", "Reconstructed"],
    num_samples=4,
    save_path=None,
):
    """
    可视化原始图像和重建图像的对比
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))

    for i in range(min(num_samples, len(originals))):
        # 重塑为图像格式 (32, 32)
        orig_img = originals[i].view(32, 32).detach().cpu().numpy()
        recon_img = reconstructions[i].view(32, 32).detach().cpu().numpy()

        # 显示原始图像
        axes[0, i].imshow(orig_img, cmap="gray")
        axes[0, i].set_title(f"{titles[0]} {i+1}")
        axes[0, i].axis("off")

        # 显示重建图像
        axes[1, i].imshow(recon_img, cmap="gray")
        axes[1, i].set_title(f"{titles[1]} {i+1}")
        axes[1, i].axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Loss curves saved to {save_path}")

    plt.tight_layout()
    plt.show()


def plot_loss_curves(train_loss_history, train_recon_loss_history, save_path=None):
    """
    绘制训练损失曲线

    Args:
        train_loss_history: 总损失历史
        train_recon_loss_history: 重构损失历史
        save_path: 保存图像路径（可选）
    """
    plt.figure(figsize=(12, 5))

    # 绘制总损失
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Total Loss", color="blue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 绘制重构损失
    plt.subplot(1, 2, 2)
    plt.plot(
        train_recon_loss_history, label="Reconstruction Loss", color="red", linewidth=2
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Loss curves saved to {save_path}")

    plt.show()


if __name__ == "__main__":

    save_path = "models/best_model.pth"

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    logging.info("=" * 60)
    logging.info("🎯 训练策略：使用生成数据训练，真实数据测试")
    logging.info("=" * 60)

    # ==================== 生成训练数据 ====================
    logging.info("\n🔧 生成训练数据...")
    train_images = generate_training_images(n=2000, size=(32, 32))  # 生成2000张训练图像
    logging.info(f"生成了 {len(train_images)} 张训练图像")

    # 预处理训练数据
    train_data = preprocess_images(train_images, target_size=(32, 32), is_flat=False)
    logging.info(f"训练数据形状: {train_data.shape}")

    # ==================== 加载真实测试数据 ====================
    logging.info("\n📂 加载真实测试数据...")
    try:
        test_images_raw = load_data("dataset/row_roket.zip")
        logging.info(f"加载了 {len(test_images_raw)} 张真实测试图像")
    except FileNotFoundError:
        logging.info("❌ 错误: 未找到 dataset/row_roket.zip 文件")
        logging.info("请确保真实数据文件存在，用于测试模型性能")
        exit(1)

    # 预处理测试数据
    test_data = preprocess_images(test_images_raw, target_size=(32, 32), is_flat=False)
    logging.info(f"测试数据形状: {test_data.shape}")

    # ==================== 参数设置 ====================
    input_dim = train_data.shape[1]  # 32*32 = 1024
    m_dim = 128  # 压缩维度
    batch_size = 32  # 训练批次大小
    epochs = 2000
    lr = 1e-5
    use_lr_scheduler = True

    n_layers = 12
    hidden_dim = m_dim * 6
    aux_predictor_hidden = m_dim * 6

    logging.info(f"\n📊 参数设置:")
    logging.info(f"  输入维度: {input_dim}")
    logging.info(f"  压缩维度: {m_dim}")
    logging.info(f"  压缩率: {m_dim/input_dim:.3f} ({input_dim//m_dim}:1)")
    logging.info(f"  批大小: {batch_size}")
    logging.info(f"  训练轮数: {epochs}")
    logging.info(f"  学习率: {lr}")
    logging.info(f"  学习率调整: {use_lr_scheduler}")
    logging.info(f"  网络层数: {n_layers}")
    logging.info(f"  隐藏层维度: {hidden_dim}")
    logging.info(f"  预测层维度: {aux_predictor_hidden}")

    # ==================== 初始化模型 ====================
    model = ApCM(
        L=32,
        D=32,
        m_dim=m_dim,
        max_mem=16,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        aux_predictor_hidden=aux_predictor_hidden,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_lr_scheduler:
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=50, verbose=True
            )
        except TypeError:
            logging.info("检测到旧版本 PyTorch，使用无 verbose 参数的学习率调度器")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=50
            )

    def loss_fn(x, x_recon, z_comp, log_det):
        recon_loss = F.mse_loss(x_recon, x)
        prior_loss = 0.5 * torch.mean(z_comp.pow(2))
        total_loss = recon_loss + 0.01 * prior_loss
        return total_loss, recon_loss

    # 初始化损失记录
    train_loss_history = []
    train_recon_loss_history = []
    best_loss = float("inf")
    lr_update_counter = 0  # 记录学习率调整次数

    # ==================== 训练阶段 ====================
    logging.info("\n🚀 开始训练（使用生成数据）...")
    for epoch in range(epochs):
        # 随机采样一个批次
        indices = torch.randperm(len(train_data))[:batch_size]
        x_batch = train_data[indices].to(device)

        # 前向传播
        z_comp, z_aux_true, log_det = model.vectorEncoder(x_batch)
        x_recon = model.vectorDecoder(z_comp)

        # 计算损失
        loss, recon_loss = loss_fn(x_batch, x_recon, z_comp, log_det)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 学习率调整
        if use_lr_scheduler:
            scheduler.step(loss)

        # 记录学习率调整（手动实现 verbose 功能）
        current_lr = optimizer.param_groups[0]["lr"]
        if (
            use_lr_scheduler
            and epoch > 0
            and current_lr != optimizer.param_groups[0].get("prev_lr", current_lr)
        ):
            lr_update_counter += 1
            logging.info(f"Epoch {epoch+1}: 学习率调整为 {current_lr:.6f}")
        optimizer.param_groups[0]["prev_lr"] = current_lr

        # 记录损失
        train_loss_history.append(loss.item())
        train_recon_loss_history.append(recon_loss.item())

        # 保存最佳模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 50 == 0:
            logging.info(
                f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, Recon Loss: {recon_loss.item():.6f}, LR: {current_lr:.6f}"
            )

    logging.info(f"\n训练完成，学习率共调整了 {lr_update_counter} 次")
    logging.info(f"最佳损失: {best_loss:.6f}")

    # 加载最佳模型用于测试
    logging.info("加载最佳模型进行测试...")
    model.load_state_dict(torch.load(save_path))

    logging.info("\n📈 绘制训练损失曲线...")
    plot_loss_curves(
        train_loss_history,
        train_recon_loss_history,
        save_path="assets/loss_curves_res_GR"
        + str(input_dim // m_dim)
        + "Lr["
        + str(use_lr_scheduler)
        + "]n"
        + str(n_layers)
        + "h"
        + str(hidden_dim)
        + "ph"
        + str(aux_predictor_hidden)
        + "E"
        + str(epochs)
        + ".png",
    )

    # ==================== 测试阶段 ====================
    logging.info("\n" + "=" * 60)
    logging.info("🧪 开始测试（使用真实数据）...")
    logging.info("=" * 60)

    # 使用所有真实测试图像
    test_data_device = test_data.to(device)

    # 压缩
    with torch.no_grad():
        z_comp, _, _ = model.vectorEncoder(test_data_device)

    logging.info(f"测试数据维度: {test_data_device.shape}")
    logging.info(f"压缩后维度: {z_comp.shape}")
    logging.info(f"压缩率: {z_comp.shape[1] / test_data_device.shape[1]:.3f}")

    # 解压
    with torch.no_grad():
        x_recon = model.vectorDecoder(z_comp)

    # 评估
    mae = F.l1_loss(test_data_device, x_recon).item()
    mse = F.mse_loss(test_data_device, x_recon).item()
    psnr_val = psnr(test_data_device, x_recon).item()

    logging.info(f"\n📊 整体测试结果:")
    logging.info(f"  重建 MAE: {mae:.6f}")
    logging.info(f"  重建 MSE: {mse:.6f}")
    logging.info(f"  PSNR: {psnr_val:.2f} dB")

    # 可视化对比
    logging.info("\n📸 生成可视化对比图（真实数据 vs 重建结果）...")
    visualize_comparison(
        test_data_device[:4],  # 只展示前4张
        x_recon[:4],
        titles=[
            f"Real Image {z_comp.shape[1] / test_data_device.shape[1]:.3f}",
            f"Reconstructed {z_comp.shape[1] / test_data_device.shape[1]:.3f}",
        ],
        num_samples=min(4, len(test_data_device)),
        save_path="assets/Example_res_GR_"
        + str(input_dim // m_dim)
        + "Lr["
        + str(use_lr_scheduler)
        + "]n"
        + str(n_layers)
        + "h"
        + str(hidden_dim)
        + "ph"
        + str(aux_predictor_hidden)
        + "E"
        + str(epochs)
        + ".png",
    )

    # 计算每张图像的详细指标
    logging.info("\n📈 每张测试图像的详细指标:")
    for i in range(min(5, len(test_data_device))):
        psnr_single = psnr(test_data_device[i], x_recon[i]).item()
        mae_single = F.l1_loss(test_data_device[i], x_recon[i]).item()
        mse_single = F.mse_loss(test_data_device[i], x_recon[i]).item()
        logging.info(
            f"  图像 {i+1}: PSNR = {psnr_single:.2f} dB, MAE = {mae_single:.6f}, MSE = {mse_single:.6f}"
        )

    logging.info("\n" + "=" * 60)
    logging.info("✅ 测试完成！")
    logging.info(f"  训练数据: {len(train_data)} 张生成图像")
    logging.info(f"  测试数据: {len(test_data)} 张真实图像")
    logging.info(f"  压缩效果: {input_dim} 维 → {m_dim} 维")
    logging.info(f"  最佳模型已保存到: {save_path}")
    logging.info("=" * 60)

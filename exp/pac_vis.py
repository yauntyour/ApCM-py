# train_pca_baseline.py
from io import BytesIO
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import zipfile
from sklearn.decomposition import PCA
import sys
import logging
from datetime import datetime, timezone

# 配置日志（完全仿照 vis.py）
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(
            f"log/pca_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(filepath):
    """加载真实数据（用于测试）"""
    images = []
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        file_list = zip_ref.namelist()
        for file_name in file_list:
            if "png" in file_name:
                img = Image.open(BytesIO(zip_ref.read(file_name)))
                img = img.convert("L")
                img_tensor = torch.tensor(np.array(img), dtype=torch.float32)
                images.append(img_tensor)
    return images


def generate_training_images(n=1000, size=(32, 32)):
    """生成模拟训练数据（与 vis.py 完全一致）"""
    images = []
    for _ in range(n):
        img = torch.zeros(size)
        num_blobs = torch.randint(3, 9, (1,)).item()
        for _ in range(num_blobs):
            cx = torch.randint(5, size[0] - 5, (1,)).item()
            cy = torch.randint(5, size[1] - 5, (1,)).item()
            x, y = torch.meshgrid(torch.arange(size[0]), torch.arange(size[1]))
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            sigma = torch.randint(2, 5, (1,)).item()
            blob = torch.exp(-d2 / (2 * sigma**2))
            img += blob * torch.rand(1) * 0.8
        img += torch.randn_like(img) * 0.05
        img = torch.clamp(img, 0, 1)
        images.append(img)
    return images


def preprocess_images(images, target_size=(32, 32), is_flat=False):
    """预处理图像（与 vis.py 完全一致）"""
    processed_images = []
    for img in images:
        if is_flat:
            img_2d = img.view(target_size)
            img_pil = Image.fromarray((img_2d.numpy() * 255).astype(np.uint8), mode="L")
        else:
            img_pil = Image.fromarray((img.numpy() * 255).astype(np.uint8), mode="L")
        if img_pil.size != target_size:
            img_resized = img_pil.resize(target_size)
        else:
            img_resized = img_pil
        img_tensor = torch.tensor(np.array(img_resized), dtype=torch.float32)
        img_normalized = img_tensor / 255.0
        img_flat = img_normalized.view(-1)
        processed_images.append(img_flat)
    return torch.stack(processed_images)


def psnr(original, reconstructed):
    """计算峰值信噪比（与 vis.py 完全一致）"""
    mse = F.mse_loss(original, reconstructed)
    if mse == 0:
        return float("inf")
    max_val = 1.0
    psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_val


def visualize_comparison(
    originals,
    reconstructions,
    titles=["Original", "Reconstructed"],
    num_samples=4,
    save_path=None,
):
    """可视化对比（与 vis.py 完全一致）"""
    fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))
    for i in range(min(num_samples, len(originals))):
        orig_img = originals[i].view(32, 32).detach().cpu().numpy()
        recon_img = reconstructions[i].view(32, 32).detach().cpu().numpy()
        axes[0, i].imshow(orig_img, cmap="gray")
        axes[0, i].set_title(f"{titles[0]} {i+1}")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_img, cmap="gray")
        axes[1, i].set_title(f"{titles[1]} {i+1}")
        axes[1, i].axis("off")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Visualization saved to {save_path}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    logging.info("=" * 60)
    logging.info("🎯 Baseline: PCA Compression vs IDRP (for comparison)")
    logging.info("=" * 60)

    # ==================== 生成训练数据 ====================
    logging.info("\n🔧 生成训练数据...")
    train_images = generate_training_images(n=2000, size=(32, 32))
    logging.info(f"生成了 {len(train_images)} 张训练图像")

    # 预处理训练数据（扁平化）
    train_data_flat = preprocess_images(
        train_images, target_size=(32, 32), is_flat=False
    )
    logging.info(f"训练数据形状: {train_data_flat.shape}")  # (2000, 1024)

    # ==================== 加载真实测试数据 ====================
    logging.info("\n📂 加载真实测试数据...")
    try:
        test_images_raw = load_data("dataset/row_roket.zip")
        logging.info(f"加载了 {len(test_images_raw)} 张真实测试图像")
    except FileNotFoundError:
        logging.error("❌ 错误: 未找到 dataset/row_roket.zip 文件")
        exit(1)

    test_data_flat = preprocess_images(
        test_images_raw, target_size=(32, 32), is_flat=False
    )
    logging.info(f"测试数据形状: {test_data_flat.shape}")

    # ==================== 参数设置 ====================
    input_dim = train_data_flat.shape[1]  # 1024
    m_dim = 128
    compression_ratio = m_dim / input_dim
    logging.info(f"\n📊 PCA 参数:")
    logging.info(f" 输入维度: {input_dim}")
    logging.info(f" 压缩维度: {m_dim}")
    logging.info(f" 压缩率: {compression_ratio:.3f} ({input_dim//m_dim}:1)")

    # ==================== 训练 PCA（拟合） ====================
    logging.info("\n🚀 拟合 PCA 模型（在生成数据上）...")
    pca = PCA(n_components=m_dim, svd_solver="full", random_state=42)

    # 将训练数据转为 NumPy 并标准化（可选，但通常 PCA 对 scale 敏感）
    X_train_np = train_data_flat.cpu().numpy()
    # 注意：这里不归一化均值，因为图像已归一化到 [0,1]
    pca.fit(X_train_np)
    logging.info(
        f"PCA 拟合完成。解释方差比例: {pca.explained_variance_ratio_.sum():.4f}"
    )

    # ==================== 测试阶段 ====================
    logging.info("\n" + "=" * 60)
    logging.info("🧪 开始测试（使用真实数据）...")
    logging.info("=" * 60)

    X_test_np = test_data_flat.cpu().numpy()
    logging.info(f"测试数据维度: {X_test_np.shape}")

    # 压缩
    z_comp = pca.transform(X_test_np)  # (N, m_dim)
    logging.info(f"压缩后维度: {z_comp.shape}")
    logging.info(f"压缩率: {z_comp.shape[1] / X_test_np.shape[1]:.3f}")

    # 解压（重建）
    X_recon_np = pca.inverse_transform(z_comp)  # (N, 1024)
    X_recon = torch.from_numpy(X_recon_np).float().to(device)
    test_data_device = test_data_flat.to(device)

    # 评估
    mae = F.l1_loss(test_data_device, X_recon).item()
    mse = F.mse_loss(test_data_device, X_recon).item()
    psnr_val = psnr(test_data_device, X_recon).item()

    logging.info(f"\n📊 PCA 测试结果:")
    logging.info(f" 重建 MAE: {mae:.6f}")
    logging.info(f" 重建 MSE: {mse:.6f}")
    logging.info(f" PSNR: {psnr_val:.2f} dB")

    # 可视化对比
    logging.info("\n📸 生成可视化对比图（真实数据 vs PCA 重建）...")
    visualize_comparison(
        test_data_device[:4],
        X_recon[:4],
        titles=[
            f"Real Image {compression_ratio:.3f}",
            f"PCA Reconstructed {compression_ratio:.3f}",
        ],
        num_samples=min(4, len(test_data_device)),
        save_path=f"assets/PCA_Example_{input_dim//m_dim}to1.png",
    )

    # 每张图像的详细指标
    logging.info("\n📈 每张测试图像的详细指标:")
    for i in range(min(5, len(test_data_device))):
        psnr_single = psnr(test_data_device[i], X_recon[i]).item()
        mae_single = F.l1_loss(test_data_device[i], X_recon[i]).item()
        mse_single = F.mse_loss(test_data_device[i], X_recon[i]).item()
        logging.info(
            f" 图像 {i+1}: PSNR = {psnr_single:.2f} dB, MAE = {mae_single:.6f}, MSE = {mse_single:.6f}"
        )

    logging.info("\n" + "=" * 60)
    logging.info("✅ PCA 基线测试完成！")
    logging.info(f" 训练数据: {len(train_data_flat)} 张生成图像")
    logging.info(f" 测试数据: {len(test_data_flat)} 张真实图像")
    logging.info(f" 压缩效果: {input_dim} 维 → {m_dim} 维")
    logging.info("=" * 60)

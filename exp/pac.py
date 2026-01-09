# pca_model.py
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np


class PCACompressor:
    def __init__(self, n_components: int):
        """
        Args:
            n_components (int): 压缩后的维度 m_dim
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, X: torch.Tensor):
        """
        X: (B, L, D) -> flatten to (B, L*D)
        """
        B, L, D = X.shape
        X_flat = X.view(B, -1).cpu().numpy()  # (B, L*D)
        self.pca.fit(X_flat)
        self.is_fitted = True

    def compress(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, L, D) -> z: (B, m)"""
        assert self.is_fitted, "Must call fit() before compress"
        B, L, D = X.shape
        X_flat = X.view(B, -1).cpu().numpy()
        z = self.pca.transform(X_flat)  # (B, m)
        return torch.from_numpy(z).to(X.device)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, m) -> X_rec: (B, L, D)"""
        assert self.is_fitted, "Must call fit() before reconstruct"
        z_np = z.cpu().numpy()
        X_rec_flat = self.pca.inverse_transform(z_np)  # (B, L*D)
        # 注意：原始 L, D 需要外部传入！这里我们假设已知
        # 为简化，返回扁平形式，由外部 reshape
        return torch.from_numpy(X_rec_flat).to(z.device)

    @property
    def compression_ratio(self):
        # 假设原始维度为 fitted 时的特征数
        if not self.is_fitted:
            return None
        orig_dim = self.pca.n_features_in_
        return self.n_components / orig_dim

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


def set_seed(seed=42):
    """
    设置所有相关随机种子以确保结果可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU


set_seed(42)


class SwiGLULayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x):
        x = self.w(x)
        gate, val = x.chunk(2, dim=-1)
        return F.silu(gate) * val  # silu = swish


class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            SwiGLULayer(hidden_dim, hidden_dim),  # 门控
            nn.Linear(hidden_dim, dim),
        )
        self.layer_norm = nn.LayerNorm(dim)  # 可选，提升稳定性

    def forward(self, x):
        return x + self.block(self.layer_norm(x))  # Pre-LN 风格


class InvertibleCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=None, affine=True):
        super().__init__()
        self.affine = affine
        half_dim = dim // 2
        if hidden_dim is None:
            hidden_dim = half_dim * 2
        out_dim = half_dim * 2 if affine else half_dim
        self.net = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            SwiGLULayer(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim),  # 通常为1-2个。强非线性为2个
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        st = self.net(x1)
        if self.affine:
            s, t = st.chunk(2, dim=-1)
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            log_det = torch.sum(s, dim=-1)
        else:
            t = st
            y1 = x1
            y2 = x2 + t
            log_det = torch.zeros(x.shape[0], device=x.device)
        return torch.cat([y1, y2], dim=-1), log_det

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=-1)
        st = self.net(y1)
        if self.affine:
            s, t = st.chunk(2, dim=-1)
            x1 = y1
            x2 = (y2 - t) * torch.exp(-s)
        else:
            t = st
            x1 = y1
            x2 = y2 - t
        return torch.cat([x1, x2], dim=-1)


class PermuteLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        perm = torch.randperm(dim)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(dim)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x):
        return x[..., self.perm]

    def inverse(self, x):
        return x[..., self.inv_perm]


class InvertibleDimReductionWithPredictor(nn.Module):
    """
    IDRP神经网络
    使用神经网络预测 z_aux,只需存储 z_comp（有损压缩）
    """

    def __init__(
        self,
        input_dim,
        m_dim,
        n_layers=6,
        hidden_dim=None,
        affine=True,
        aux_predictor_hidden=64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.m_dim = m_dim
        self.aux_dim = input_dim - m_dim
        self.network = nn.ModuleList()
        for i in range(n_layers):
            self.network.append(InvertibleCouplingLayer(input_dim, hidden_dim, affine))
            self.network.append(PermuteLayer(input_dim))

        # 额外的网络用于从 z_comp 预测 z_aux
        self.aux_predictor = nn.Sequential(
            nn.Linear(m_dim, aux_predictor_hidden),
            SwiGLULayer(aux_predictor_hidden, aux_predictor_hidden),
            nn.Linear(aux_predictor_hidden, self.aux_dim),
        )

    def forward(self, x):
        h = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for layer in self.network:
            if isinstance(layer, InvertibleCouplingLayer):
                h, log_det = layer(h)
                log_det_total += log_det
            else:
                h = layer(h)
        z_comp = h[:, : self.m_dim]
        z_aux_true = h[:, self.m_dim :]  # 真实的 z_aux（用于训练）
        return z_comp, z_aux_true, log_det_total

    def inverse(self, z_comp):
        """
        从 z_comp 预测 z_aux，然后重建 x
        """
        z_aux_pred = self.aux_predictor(z_comp)
        h = torch.cat([z_comp, z_aux_pred], dim=-1)
        for layer in reversed(self.network):
            h = layer.inverse(h)
        return h

    def compress(self, x):
        """
        压缩函数：返回 z_comp（压缩表示）
        """
        z_comp, _, _ = self.forward(x)
        return z_comp

    def decompress(self, z_comp):
        """
        解压函数：从 z_comp 重建 x
        """
        return self.inverse(z_comp)


class ApCM(nn.Module):

    def __init__(
        self,
        L,
        D,
        m_dim,
        max_mem,
        n_layers: int = 4,
        hidden_dim: int = 256,
        aux_predictor_hidden: int = 128,
    ):
        """
        Args:
            n_layers: Invertible Dim Compressor layers
            hidden_dim: Invertible Dim Compressor hidden_dim
        Returns:
            selected_memory: (batch_size, m_dim)
        """
        super(ApCM, self).__init__()
        self.L = L
        self.D = D
        self.max_mem = max_mem
        self.m_dim = m_dim  # event vector dim
        # vectorEncoder
        self.vectorizer = InvertibleDimReductionWithPredictor(
            input_dim=L * D,
            m_dim=m_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            aux_predictor_hidden=aux_predictor_hidden,
        )
        self.register_buffer("AFF_ctrl", torch.zeros((1, max_mem)))

    def vectorDecoder(self, x):
        return self.vectorizer.inverse(x)

    def vectorEncoder(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (B, L*D)
        return self.vectorizer(x)

    def read(self, x, memory_block, eps=1e-8):
        """
        Args:
            x: (batch_size, L, D)
            memory_block: (max_mem, m_dim) —— 全局共享 memory bank
        Returns:
            selected_memory: (batch_size, m_dim)
        """
        batch_size = x.size(0)
        assert memory_block.shape == (
            self.max_mem,
            self.m_dim,
        ), f"Expected memory_block shape ({self.max_mem}, {self.m_dim}), got {memory_block.shape}"

        # Encode input to vector
        x = x.view(batch_size, -1)  # (B, L*D)
        z_comp, _, _ = self.vectorizer(x)  # (B, m_dim)

        # L2 normalize for cosine similarity
        z_norm = F.normalize(z_comp, p=2, dim=1, eps=eps)  # (B, m_dim)
        mem_norm = F.normalize(memory_block, p=2, dim=1, eps=eps)  # (max_mem, m_dim)

        # Compute cosine similarity: (B, m_dim) @ (max_mem, m_dim)^T -> (B, max_mem)
        sim = z_norm @ mem_norm.t()

        # Softmax over memory slots
        alpha = F.softmax(sim, dim=1)  # (B, max_mem)

        # Get most similar memory index
        index = torch.argmax(alpha, dim=1)  # (B,)

        # Retrieve selected memories: (B, m_dim)
        raw_data = self.vectorizer.inverse(memory_block[index])  # advanced indexing

        # update access frequency
        index_2d = index.unsqueeze(0)  #  (1, batch_size)
        ones_2d = torch.ones_like(index_2d, dtype=torch.float)  # ones
        self.AFF_ctrl.scatter_add_(1, index_2d, ones_2d)

        return raw_data, index

    def write(self, x, memory_block):
        """
        Args:
            x: (batch_size, L, D)
            memory_block: (max_mem, m_dim) —— 全局共享 memory bank
        Returns:
            indices: (batch_size,) —— 每个样本写入的位置
        """
        batch_size = x.size(0)
        assert memory_block.shape == (self.max_mem, self.m_dim)

        # 编码输入得到压缩向量
        vectors = self.vectorEncoder(x)  # (B, m_dim)

        access_freq = self.AFF_ctrl.squeeze(0)  # (max_mem,)
        if (access_freq == 0).any():
            unused_indices = torch.nonzero(access_freq == 0, as_tuple=False).squeeze()
            if unused_indices.numel() == 1:
                index = unused_indices.item()
            else:
                index = unused_indices[0].item()
        else:
            index = torch.argmin(access_freq).item()

        self.AFF_ctrl[:, index] = 0

        avg_vector = vectors.mean(dim=0)  # (m_dim,)
        memory_block[index] = avg_vector

        return memory_block, torch.tensor([index] * batch_size, device=x.device)


if __name__ == "__main__":
    B = 2  # batch size
    L = 4  # sequence length
    D = 4  # d_model
    m_dim = 4  # event vector dim
    max_mem = 4  # max memory nums
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ApCM(L=L, D=D, m_dim=m_dim, max_mem=max_mem).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # (batch_size, L, D)
    x = torch.tensor(
        [
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            # [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]],
        ],
        dtype=torch.float,
    ).to(device)

    memory_block = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [-0.2224, -0.2450, 0.1069, -0.2975],
            [0.7210, 2.3575, 3.0989, 3.0838],
        ],
        dtype=torch.float,
    ).to(device)

    output, index = model.read(x, memory_block)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    print(f"Index:{index}")

    print(f"Old:{memory_block}")
    memory_block, index = model.write(x, memory_block)
    print(f"New:{memory_block}")
    print(f"Index:{index}")

import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath


class GeometricCosineAttention(nn.Module):
    """基于几何约束的方向余弦空间注意力模块 - 延后归一化版本"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # QKV的映射矩阵
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gamma = nn.Parameter(torch.zeros(1))

        # 是否使用softmax (可配置参数)
        self.use_softmax = False

        # 输出归一化
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x, adj_mask):
        """
        Args:
            x: 输入特征 [B, E, C] B是批次，E是边数，C是特征维度
            adj_mask: 邻接矩阵 [E, E]
        """
        identity = x
        B, E, C = x.shape

        # 生成QKV
        q = (
            self.q_proj(x)
            .reshape(B, E, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, E, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, E, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # 不再限制使用前3个通道
        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, num_heads, E, E]

        # 应用邻接矩阵掩码
        mask = adj_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, E, E]

        if self.use_softmax:
            # 传统方式：使用softmax时需要将不连接的边设为-inf
            attn = attn.masked_fill(mask == 0, float("-inf"))
            attn = attn.softmax(dim=-1)
        else:
            # 几何方式：直接用掩码过滤不连接的边
            attn = attn * mask  # 不连接的边乘以0，消除其影响
            # 确保值在[0,1]范围内（将[-1,1]映射到[0,1]）
            attn = (attn + 1) / 2
            # 移除行归一化步骤，保留原始几何相似度强度

        attn = self.attn_drop(attn)

        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, E, C)

        # 投影后应用归一化以保持几何约束
        x = self.proj(x)
        x = self.output_norm(x)  # 归一化延后到这里，处理聚合后的特征
        x = self.proj_drop(x)

        # 添加残差连接
        return identity + self.gamma * x


# 修改 TemporalConvolution 类


class TemporalConvolution(nn.Module):
    """时间域卷积模块，替代原有的时间注意力"""

    def __init__(self, dim, kernel_size=3, expansion_factor=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        # 扩张的隐藏层维度
        hidden_dim = dim * expansion_factor

        # 预定义重排对象
        self.rearrange_to_conv = Rearrange("b t e c -> (b e) c t")

        # 构建时间卷积网络主体
        self.conv_network = nn.Sequential(
            # 第一层卷积 (升维)
            nn.Conv1d(
                dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2, bias=False
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # 第二层卷积 (降维)
            nn.Conv1d(hidden_dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
        )

        # 层归一化（在卷积前应用）
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.zeros(1))  # 与注意力模块一致，使用gamma参数

    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, T, E, C]
        """
        B, T, E, C = x.shape
        identity = x  # 保存输入用于残差连接

        # 应用归一化
        x = self.norm(x)

        # 重排为卷积格式
        x = self.rearrange_to_conv(x)

        # 应用卷积网络
        x = self.conv_network(x)

        # 重排回原始格式
        x = rearrange(x, "(b e) c t -> b t e c", b=B, e=E)

        # 添加残差连接，使用gamma参数控制
        return identity + self.gamma * x


class ST_TR_Block(nn.Module):
    """时空注意力块"""

    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
    ):
        super().__init__()

        # 空间注意力保持不变
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_attn = GeometricCosineAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # 替换时间注意力为时间卷积
        self.norm2 = nn.Identity()  # 卷积模块内部已包含LayerNorm
        self.temporal_attn = TemporalConvolution(
            dim,
            kernel_size=3,  # 可以调整卷积核大小
            expansion_factor=2,  # 可以调整扩张因子
            dropout=drop,  # 使用相同的dropout率
        )

        # MLP保持不变
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, adj_mask):
        """
        Args:
            x: 输入特征 [B, T, E, C]
            adj_mask: 邻接矩阵 [E, E]
        """
        B, T, E, C = x.shape  # 获取时间步长T

        # 空间注意力
        x_spatial = rearrange(x, "b t e c -> (b t) e c")
        x_spatial = self.norm1(x_spatial)
        x_spatial = self.spatial_attn(x_spatial, adj_mask)
        x_spatial = rearrange(x_spatial, "(b t) e c -> b t e c", t=T)
        x = x + self.drop_path(x_spatial)

        # 时间卷积 - 使用正确的名称 temporal_attn
        x = x + self.drop_path(self.temporal_attn(self.norm2(x)))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x


def get_sinusoid_encoding_table(length, dim):
    """生成正弦余弦位置编码表"""
    position = torch.arange(length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

    pos_table = torch.zeros(length, dim)
    pos_table[:, 0::2] = torch.sin(position * div_term)
    pos_table[:, 1::2] = torch.cos(position * div_term)

    return pos_table


class LGLT(nn.Module):
    """Line Graph Linear Transformer for Skeleton-based Action Recognition"""

    def __init__(
        self,
        num_class,
        num_frames,
        num_edges,
        in_channels=3,
        embed_dim=64,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        use_learnable_pos_emb=True,
    ):
        super().__init__()

        # 基本参数
        self.num_class = num_class
        self.num_frames = num_frames
        self.num_edges = num_edges
        self.in_channels = in_channels
        self.use_learnable_pos_emb = use_learnable_pos_emb

        # 特征嵌入
        self.embed = nn.Sequential(
            nn.Linear(in_channels, embed_dim), norm_layer(embed_dim)
        )

        # 位置编码
        if use_learnable_pos_emb:
            # 可学习的位置编码
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, num_frames, embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            # 固定的正弦余弦位置编码
            pos_table = get_sinusoid_encoding_table(num_frames, embed_dim)
            self.register_buffer(
                "pos_embed_temporal", pos_table.unsqueeze(0)
            )  # [1, T, embed_dim]

        # Drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                ST_TR_Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(num_layers)
            ]
        )

        # 输出层
        self.norm = norm_layer(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_class),
        )

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, adj_mask, pos_encodings=None):
        """
        Args:
            x: 输入特征 [B, C, T, E]
            adj_mask: 邻接矩阵 [E, E]
            pos_encodings: 预计算的位置编码 [B, T, embed_dim]
        """
        B, C, T, E = x.shape

        # 形状检查
        assert E == self.num_edges, f"边数不匹配：期望 {self.num_edges}，实际得到 {E}"
        assert T <= self.num_frames, (
            f"时间步数超出限制：期望 <= {self.num_frames}，实际得到 {T}"
        )
        assert C == self.in_channels, (
            f"输入通道数不匹配：期望 {self.in_channels}，实际得到 {C}"
        )

        # 重排并嵌入
        x = rearrange(x, "b c t e -> (b t) e c")
        x = self.embed(x)
        x = rearrange(x, "(b t) e c -> b t e c", b=B)

        # 使用预计算的位置编码或模型内置的位置编码
        if pos_encodings is not None:
            pos_embed = pos_encodings.unsqueeze(2)  # [B, T, 1, embed_dim]
        else:
            # 使用模型内置的位置编码（可学习或固定）
            pos_embed = self.pos_embed_temporal[:, :T, :].unsqueeze(
                2
            )  # [1, T, 1, embed_dim]

        x = x + pos_embed

        # 通过Transformer块
        for blk in self.blocks:
            x = blk(x, adj_mask)

        # 全局池化
        x = x.mean(dim=2)  # 空间池化
        x = x.mean(dim=1)  # 时间池化

        # 分类
        x = self.norm(x)
        x = self.head(x)

        return x

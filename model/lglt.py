import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath


class SpatialAttention(nn.Module):
    """空间域注意力模块"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # QKV的映射矩阵
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 添加一个gamma参数用于控制残差强度
        self.gamma = nn.Parameter(torch.zeros(1))

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
            .reshape(B, E, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, E, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, E, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, E, E]

        # 使用邻接矩阵过滤，引入先验拓扑信息
        mask = adj_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, E, E]
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力输出
        x = (attn @ v).transpose(1, 2).reshape(B, E, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 添加残差连接，使用gamma参数控制
        return identity + self.gamma * x


class TemporalAttention(nn.Module):
    """时间域注意力模块"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # 分离QKV映射矩阵
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gamma = nn.Parameter(torch.zeros(1))  # 添加gamma参数

    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, T, E, C]
        """
        identity = x  # 保存输入用于残差连接
        B, T, E, C = x.shape

        # 重排为标准Transformer输入格式
        x = rearrange(x, "b t e c -> (b e) t c")

        # 分别生成Q,K,V
        q = (
            self.q_proj(x)
            .reshape(-1, T, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(-1, T, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(-1, T, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力输出
        x = (attn @ v).transpose(1, 2).reshape(-1, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 恢复原始形状
        x = rearrange(x, "(b e) t c -> b t e c", b=B)

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

        # 空间注意力
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_attn = SpatialAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # 时间注意力
        self.norm2 = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # MLP
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

        # 时间注意力
        x = x + self.drop_path(self.temporal_attn(self.norm2(x)))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x


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
    ):
        super().__init__()

        # 基本参数
        self.num_class = num_class
        self.num_frames = num_frames
        self.num_edges = num_edges
        self.in_channels = in_channels

        # 特征嵌入
        self.embed = nn.Sequential(
            nn.Linear(in_channels, embed_dim), norm_layer(embed_dim)
        )

        # 只保留时间位置编码
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        # 移除空间位置编码
        # self.pos_embed_spatial = nn.Parameter(torch.zeros(1, num_edges, embed_dim))

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
        nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, adj_mask):
        """
        Args:
            x: 输入特征 [B, C, T, E]
            adj_mask: 邻接矩阵 [E, E]
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

        # 获取实际序列对应的位置编码
        pos_embed = self.pos_embed_temporal[:, :T, :]  # [1, T, embed_dim]
        # 调整位置编码维度以匹配特征维度
        pos_embed = pos_embed.unsqueeze(2)  # [1, T, 1, embed_dim]
        x = x + pos_embed  # 广播机制会处理批次和边的维度

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

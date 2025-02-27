import math

import torch
import torch.nn as nn
from einops import rearrange


class PositionalEncoding(nn.Module):
    """标准的正弦余弦位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, : x.size(1), :]


class TransformerTimeSeries(nn.Module):
    """标准Transformer Encoder用于时间序列分类"""

    def __init__(
        self,
        num_class,
        num_frames,
        num_edges,
        in_channels=3,
        d_model=64,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        activation="gelu",
        use_cls_token=True,
        pool_mode="cls",  # 'cls', 'mean' 或 'attention'
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        # 基本参数
        self.num_class = num_class
        self.num_frames = num_frames
        self.num_edges = num_edges
        self.in_channels = in_channels
        self.use_cls_token = use_cls_token
        self.pool_mode = pool_mode
        self.d_model = d_model

        # 特征嵌入 - 将原始特征映射到transformer维度
        self.embed = nn.Sequential(nn.Linear(in_channels, d_model), norm_layer(d_model))

        # 位置编码
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=num_frames + 1
        )  # +1 是为了CLS token

        # CLS token (如果使用)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 标准Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # 使用Pre-LN架构，更稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 特别时间池化注意力权重 (如果使用'attention'池化)
        if pool_mode == "attention":
            self.pool_attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1)
            )

        # 分类头
        self.norm = norm_layer(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_class),
        )

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, adj_mask=None):
        """
        Args:
            x: 输入特征 [B, C, T, E]
               B: 批次大小, C: 通道数, T: 时间帧数, E: 骨架边数
            adj_mask: 邻接矩阵 (不使用，仅为保持API一致)
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

        # 重排并嵌入 - 先合并空间
        x = rearrange(x, "b c t e -> (b e) t c")  # [B*E, T, C]
        x = self.embed(x)  # [B*E, T, d_model]

        # 处理每个关节/边的时间序列
        all_outputs = []
        for i in range(B):
            # 提取当前批次的所有边
            start_idx = i * E
            end_idx = (i + 1) * E
            curr_edges = x[start_idx:end_idx]  # [E, T, d_model]

            # 对边级别进行时间序列处理
            edge_outputs = []
            for j in range(E):
                curr_edge_seq = curr_edges[j]  # [T, d_model]

                # 添加CLS token (如果启用)
                if self.use_cls_token:
                    curr_edge_seq = torch.cat(
                        [self.cls_token.expand(1, -1, -1), curr_edge_seq], dim=1
                    )  # [1, T+1, d_model]
                else:
                    curr_edge_seq = curr_edge_seq.unsqueeze(0)  # [1, T, d_model]

                # 应用位置编码
                curr_edge_seq = self.pos_encoder(curr_edge_seq)  # [1, T(+1), d_model]

                # 通过Transformer Encoder
                edge_output = self.transformer_encoder(
                    curr_edge_seq
                )  # [1, T(+1), d_model]

                # 池化以获得序列表示
                if self.use_cls_token and self.pool_mode == "cls":
                    # 使用CLS token作为序列表示
                    edge_feat = edge_output[:, 0]  # [1, d_model]
                elif self.pool_mode == "mean":
                    # 平均池化
                    start_idx = 1 if self.use_cls_token else 0
                    edge_feat = edge_output[:, start_idx:].mean(dim=1)  # [1, d_model]
                elif self.pool_mode == "attention":
                    # 注意力加权池化
                    start_idx = 1 if self.use_cls_token else 0
                    attn_weights = self.pool_attention(
                        edge_output[:, start_idx:]
                    )  # [1, T, 1]
                    attn_weights = torch.softmax(attn_weights, dim=1)
                    edge_feat = (edge_output[:, start_idx:] * attn_weights).sum(
                        dim=1
                    )  # [1, d_model]

                edge_outputs.append(edge_feat)

            # 合并所有边的特征
            edge_outputs = torch.cat(edge_outputs, dim=0)  # [E, d_model]
            edge_outputs = edge_outputs.mean(
                dim=0, keepdim=True
            )  # [1, d_model] - 空间平均池化
            all_outputs.append(edge_outputs)

        # 合并所有批次
        x = torch.cat(all_outputs, dim=0)  # [B, d_model]

        # 分类
        x = self.norm(x)
        x = self.head(x)

        return x


class TransformerTimeSeriesJoint(nn.Module):
    """联合处理所有时间和关节的Transformer Encoder"""

    def __init__(
        self,
        num_class,
        num_frames,
        num_edges,
        in_channels=3,
        d_model=64,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        activation="gelu",
        use_cls_token=True,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        # 基本参数
        self.num_class = num_class
        self.num_frames = num_frames
        self.num_edges = num_edges
        self.in_channels = in_channels
        self.use_cls_token = use_cls_token
        self.d_model = d_model

        # 特征嵌入
        self.embed = nn.Sequential(nn.Linear(in_channels, d_model), norm_layer(d_model))

        # 位置编码
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=num_frames * num_edges + 1
        )

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 标准Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 分类头
        self.norm = norm_layer(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_class),
        )

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, adj_mask=None):
        """
        Args:
            x: 输入特征 [B, C, T, E]
            adj_mask: 邻接矩阵 (不使用)
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

        # 重排并嵌入 - 将时间和边展平为一个序列
        x = rearrange(x, "b c t e -> b (t e) c")  # [B, T*E, C]
        x = self.embed(x)  # [B, T*E, d_model]

        # 添加CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+T*E, d_model]

        # 应用位置编码
        x = self.pos_encoder(x)  # [B, (1+)T*E, d_model]

        # 通过Transformer Encoder
        x = self.transformer_encoder(x)  # [B, (1+)T*E, d_model]

        # 使用CLS token作为序列表示
        if self.use_cls_token:
            x = x[:, 0]  # [B, d_model]
        else:
            # 如果没有CLS token，则使用平均池化
            x = x.mean(dim=1)  # [B, d_model]

        # 分类
        x = self.norm(x)
        x = self.head(x)

        return x

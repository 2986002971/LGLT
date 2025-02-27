import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_hop_distance(num_node, edge, max_hop=2):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


class Graph:
    """The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self, layout="coco", strategy="spatial", max_hop=2, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == "openpose":
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (4, 3),
                (3, 2),
                (7, 6),
                (6, 5),
                (13, 12),
                (12, 11),
                (10, 9),
                (9, 8),
                (11, 5),
                (8, 2),
                (5, 1),
                (2, 1),
                (0, 1),
                (15, 0),
                (14, 0),
                (17, 15),
                (16, 14),
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == "ntu-rgb+d":
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (1, 2),
                (2, 21),
                (3, 21),
                (4, 3),
                (5, 21),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 21),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 1),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 1),
                (18, 17),
                (19, 18),
                (20, 19),
                (22, 23),
                (23, 8),
                (24, 25),
                (25, 12),
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == "ntu_edge":
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (1, 2),
                (3, 2),
                (4, 3),
                (5, 2),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 2),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 1),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 1),
                (18, 17),
                (19, 18),
                (20, 19),
                (21, 22),
                (22, 8),
                (23, 24),
                (24, 12),
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == "dual_coco":
            self.num_node = 14
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                # 鼻子
                (4, 5),  # 鼻子
                # 肩线
                (0, 2),  # 左肩
                (0, 3),  # 右肩
                (0, 4),  # 左肩
                (0, 5),  # 右肩
                (0, 6),  # 左肩
                (0, 8),  # 右肩
                (2, 4),  # 左肩
                (2, 6),  # 左肩
                (3, 5),  # 右肩
                (3, 8),  # 右肩
                (4, 6),  # 左肩
                (5, 8),  # 右肩
                # 肘
                (6, 7),  # 左肘
                (8, 9),  # 右肘
                # 胯
                (1, 2),  # 左胯
                (1, 3),  # 右胯
                (1, 10),  # 左胯
                (1, 12),  # 右胯
                (2, 10),  # 左胯
                (3, 12),  # 右胯
                # 膝
                (10, 11),  # 左膝
                (12, 13),  # 右膝
                # 额外连接
                (7, 9),  # 左右前臂
                (11, 13),  # 左右小腿
            ]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == "coco":
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [16, 14],
                [14, 12],
                [17, 15],
                [15, 13],
                [12, 13],
                [6, 12],
                [7, 13],
                [6, 7],
                [8, 6],
                [9, 7],
                [10, 8],
                [11, 9],
                [2, 3],
                [2, 1],
                [3, 1],
                [4, 2],
                [5, 3],
                [4, 6],
                [5, 7],
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        # elif layout=='customer settings'
        #     pass
        elif layout == "ucla":
            self.num_node = 19  # UCLA数据集的节点数
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (4, 5),
                (4, 3),
                (4, 6),
                (9, 8),
                (12, 11),
                (15, 14),
                (18, 17),
                (1, 13),
                (1, 2),
                (1, 0),
                (13, 14),
                (2, 16),
                (2, 0),
                (16, 17),
                (5, 7),
                (5, 6),
                (5, 3),
                (7, 8),
                (6, 3),
                (6, 10),
                (11, 10),
                (0, 3),
            ]
            self.edge = self_link + neighbor_link
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == "uniform":
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == "distance":
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == "spatial":
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (
                                self.hop_dis[j, self.center]
                                == self.hop_dis[i, self.center]
                            ):
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif (
                                self.hop_dis[j, self.center]
                                > self.hop_dis[i, self.center]
                            ):
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))

        return x.contiguous(), A


def zero(x):
    return 0


def iden(x):
    return x


class ST_GCN_18(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes.
    """

    def __init__(
        self,
        in_channels=4,  # 改为4通道输入（学生序列2通道+参考序列2通道）
        graph_cfg={"layout": "dual_coco", "strategy": "spatial", "max_hop": 2},
        edge_importance_weighting=True,
        data_bn=True,
        **kwargs,
    ):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else iden

        # 主干网络
        kwargs0 = {k: v for k, v in kwargs.items() if k != "dropout"}
        self.st_gcn_networks = nn.ModuleList(
            (
                st_gcn_block(
                    in_channels, 64, kernel_size, 1, residual=False, **kwargs0
                ),
                # st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                # st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                # st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                st_gcn_block(64, 128, kernel_size, 2, **kwargs),
                # st_gcn_block(128, 128, kernel_size, 1, **kwargs),
                # st_gcn_block(128, 128, kernel_size, 1, **kwargs),
                st_gcn_block(128, 256, kernel_size, 2, **kwargs),
                # st_gcn_block(256, 256, kernel_size, 1, **kwargs),
                st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            )
        )

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # 输出头：输出单个相似度分数
        self.fcn = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        # data normalization
        N, C, T, V = x.size()  # 移除 M 维度
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, C, T, V)

        # forward backbone
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)

        # 输出单个相似度分数
        x = self.fcn(x)
        return x.view(N)  # [N]

    def extract_feature(self, x):
        # data normalization
        N, C, T, V = x.size()  # 移除 M 维度
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, c, t, v)

        # prediction
        x = self.fcn(x)
        output = x.view(N, -1, t, v)

        return output, feature


class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class STGCN_UCLA(nn.Module):
    """针对UCLA数据集优化的ST-GCN模型"""

    def __init__(
        self,
        in_channels=3,  # UCLA使用3维方向向量
        num_class=10,  # UCLA有10个动作类别
        graph_cfg={"layout": "ucla", "strategy": "distance", "max_hop": 2},
        edge_importance_weighting=True,
        data_bn=True,
        dropout=0.2,
        window_size=9,
        tcn_dropout=0.2,
        channels=None,
        strides=None,
        **kwargs,
    ):
        super().__init__()

        # 使用配置中提供的通道数和步长
        if channels is None:
            channels = [3, 64, 128, 128, 256, 256]
        if strides is None:
            strides = [1, 1, 2, 1, 2, 1]

        # UCLA边关系的Graph对象
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        # 构建网络
        spatial_kernel_size = A.size(0)
        kernel_size = (window_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else iden

        # 主干网络 - 可以根据需要调整层数和通道数
        self.st_gcn_networks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.st_gcn_networks.append(
                st_gcn_block(
                    channels[i],
                    channels[i + 1],
                    kernel_size,
                    stride=strides[i],
                    dropout=tcn_dropout if i > 0 else 0,  # 保留这个参数
                    residual=(i != 0),  # 第一层不使用残差连接
                    **kwargs,
                )
            )

        # 边重要性权重
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # 分类头
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # 数据归一化
        N, C, T, V = x.size()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, C, T, V)

        # 前向传播
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # 全局池化
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)

        # 输出分类结果
        x = self.fcn(x)
        return x.view(N, -1)  # [N, num_class]

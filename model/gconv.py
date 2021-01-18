import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tensor import dot


class GConv(nn.Module):
    """Simple GCN layer

    Similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj_mat, bias=True):
        super(GConv, self).__init__()
        # 输入特征维度
        self.in_features = in_features
        #  输入特征维度
        self.out_features = out_features
        # 邻接矩阵 节点数*节点数
        self.adj_mat = nn.Parameter(adj_mat, requires_grad=False)
        # 权重矩阵 in_features*out_features
        self.weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        # Following https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch/blob/a0ae88c4a42eef6f8f253417b97df978db842708/model/gcn_layers.py#L45
        # This seems to be different from the original implementation of P2M
        # LOOP权重 in_features*out_features
        self.loop_weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        if bias:
            # 输出偏置项
            self.bias = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 重置参数
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.xavier_uniform_(self.loop_weight.data)

    def forward(self, inputs):
        # inputs : bsx点数x输入特征维度(无任何先验的情况下,可以是点的坐标)
        # self.weight : in_features*out_features
        support = torch.matmul(inputs, self.weight)
        # support: bsx点数x输出特征维度
        # self.loop_weight : in_features*out_features
        support_loop = torch.matmul(inputs, self.loop_weight)
        
        # self.adj_mat : 点数x点数
        # dot(self.adj_mat, support, True) 
        # 矩阵乘法,得到bsx点数x输出特征维度
        output = dot(self.adj_mat, support, True) + support_loop
        if self.bias is not None:
            # 是否加输出偏置(所有点加一样的值)
            ret = output + self.bias
        else:
            ret = output
        return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Threshold

# 正交投影。。。。。。
# 坐标*外参，将坐标转变到相机坐标系下，然后用正交投影+双线性插值，获取每个点的所有的特征（遮挡问题怎么处理？？？，这里都没有深度测试，没有考虑遮挡）
class GraphProjection(nn.Module):
    """Graph Projection layer, which pool 2D features to mesh
    The layer projects a vertex of the mesh to the 2D image and use 
    bilinear interpolation to get the corresponding feature.
    """

    def __init__(self):
        super(GraphProjection, self).__init__()


    def forward(self, img_features, input):

        self.img_feats = img_features 
        # 决定图像宽高
        # h = 248 * x/z + 111.5
        # w = 248 * y/z + 111.5
        # 248和111.5怎么来的？？经验值?
        h = 248 * torch.div(input[:, 1], input[:, 2]) + 111.5
        w = 248 * torch.div(input[:, 0], -input[:, 2]) + 111.5
        # 裁剪图像，最大值为223 （即图像为<=224）
        h = torch.clamp(h, min = 0, max = 223)
        w = torch.clamp(w, min = 0, max = 223)
        # 特征图尺寸
        img_sizes = [56, 28, 14, 7]
        out_dims = [64, 128, 256, 512]
        feats = [input]
        # 四次投影
        for i in range(4):
            out = self.project(i, h, w, img_sizes[i], out_dims[i])
            feats.append(out)
        # 四次投影的特征直接cat
        output = torch.cat(feats, 1)
        
        return output

    def project(self, index, h, w, img_size, out_dim):
        # 第index次投影， 图像尺寸h*w , 图像尺寸img_size（xy方向相同）
        
        # 取出本次特征
        img_feat = self.img_feats[index]
        # 计算出图像尺寸大小和224原图的相对百分比，由此得出输出特征图尺寸相对于当前特征图大小
        x = h / (224. / img_size)
        y = w / (224. / img_size)
        # torch.floor(x) ： 小于等于x的最大整数
        # torch.ceil(x) ： 大于等于x的最小整数
        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()
        # 按图像尺寸阶段最大值
        x2 = torch.clamp(x2, max = img_size - 1)
        y2 = torch.clamp(y2, max = img_size - 1)

        #Q11 = torch.index_select(torch.index_select(img_feat, 1, x1), 1, y1)
        #Q12 = torch.index_select(torch.index_select(img_feat, 1, x1), 1, y2)
        #Q21 = torch.index_select(torch.index_select(img_feat, 1, x2), 1, y1)
        #Q22 = torch.index_select(torch.index_select(img_feat, 1, x2), 1, y2)

        # Q11为
        Q11 = img_feat[:, x1, y1].clone()
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()

        x, y = x.long(), y.long()
        # 双线性插值
        weights = torch.mul(x2 - x, y2 - y)
        
        Q11 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q11, 0, 1))

        weights = torch.mul(x2 - x, y - y1)
        Q12 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q12, 0 ,1))

        weights = torch.mul(x - x1, y2 - y)
        Q21 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q21, 0, 1))

        weights = torch.mul(x - x1, y - y1)
        Q22 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q22, 0, 1))

        output = Q11 + Q21 + Q12 + Q22

        return output
    
gp = GraphProjection()
bs = 4
channels = 16
h = [56, 28, 14, 7]
w = [56, 28, 14, 7]
img_features = []
for i in range(4):
    img_features.append (torch.rand((bs,h[i],w[i])))
N = 500
dim = 3
input = torch.rand((N,dim))
gp(img_features,input)


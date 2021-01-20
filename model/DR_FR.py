'''
Differential Renderer based Finger Recognition
'''
from numpy.matrixlib.defmatrix import matrix
from torch import embedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import math
import imageio
import kaolin as kal
from kaolin.graphics import DIBRenderer
from kaolin.graphics.dib_renderer.utils.sphericalcoord import get_spherical_coords_x
from kaolin.rep import TriangleMesh
import torchvision
import numpy as np
import os
import math

# pointnet
from kaolin.models.PointNet import PointNetClassifier

class DR_FR_Model(nn.Module):
    r"""Differential Renderer based Finger Recognition
        """
    def __init__(self,f_dim=512, point_num = 962 , num_classes=347):
        super(DR_FR_Model, self).__init__()
        '''
        初始化参数:
        '''
        self.f_dim = f_dim
        self.point_num = point_num
        self.num_classes = num_classes
        self.model = PointNetClassifier(num_classes=self.num_classes,classifier_layer_dims = [1024, 512], feat_layer_dims = [32,128,512])
        
    def forward(self, point):
        '''
        输入: 点云 N*3
        输出: logit
        '''
        feature,logit = self.model(point)
        return feature,logit

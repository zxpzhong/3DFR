'''
Verificaiotn_DataLoader++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
from torchvision import datasets, transforms
from base import BaseDataLoader
import os
import torch
import pandas as pd
from torchvision import transforms as T
from PIL import Image
import numpy as np
import kaolin as kal
from torchvision import datasets, transforms
from base import BaseDataLoader
import random
import trimesh
import math

'''
数据扩增
'''
def eulerAnglesToRotationMatrix(angles1) :
    theta = np.zeros((3, 1), dtype=np.float64)
    theta[0] = angles1[0]*3.14159265/180.0
    theta[1] = angles1[1]*3.14159265/180.0
    theta[2] = angles1[2]*3.14159265/180.0
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R
# R = eulerAnglesToRotationMatrix([35,20,-30])

def rotate_point (point, rotation_angle):
    point = np.array(point)
    rotation_matrix = eulerAnglesToRotationMatrix(rotation_angle)
    rotated_point = point.reshape(-1, 3)@rotation_matrix
    return rotated_point
 
###########
# 在XYZ上加高斯噪声 #
###########
def jitter_point(point, sigma=0.01, clip=0.05):
    assert(clip > 0)
    point = np.array(point)
    point = point.reshape(-1,3)
    Row, Col = point.shape
    jittered_point = sigma * np.random.randn(Row, Col)
    # jittered_point = np.clip(sigma * np.random.randn(Row, Col), -1*clip, clip)
    jittered_point += point
    return jittered_point

def augment_data(point, rotation_angle = 0, sigma = 0):
    rotation_angle = list(np.random.rand(3))*360
    sigma = (np.random.random()-0.5)/10
    return jitter_point(rotate_point(point, rotation_angle), sigma)

def img_open(path):
    img = Image.open(path)
    array = np.array(img)
    # array[:,:20,:] = 0
    # array[:,-20:,:] = 0
    img = Image.fromarray(array, mode='RGB')
    return img
class LFMB_3DFB_Pictures_Seged_Rectified_Train(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset = Train_Dataset("/home/zf/vscode/3d/DR_3DFM/saved/obj/pytorch3duvmap/0118_210807/",self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class LFMB_3DFB_Pictures_Seged_Rectified_Test(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset = Test_Dataset("/home/zf/vscode/3d/DR_3DFM/saved/obj/pytorch3duvmap/0118_210807/",self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

transform = T.Compose([
    # T.Resize([256, 256]),
    # T.RandomCrop(224),
    T.Grayscale(),
    T.RandomRotation(10),
    T.RandomAffine(10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    T.RandomGrayscale(),
    T.RandomPerspective(0.2,0.2),
    T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize([0.5], [0.5]),  # 标准化至[-1, 1]，规定均值和标准差
])
'''
PIL读取出来的图像默认就已经是0-1范围了！！！！！！！！，不用再归一化
'''
transform_notrans = T.Compose([
    # T.Grayscale(),
    T.Resize([256,256]), # 缩放图片(Image)
    T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
    # T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
    # T.Normalize([0.5], [0.5]),  # 标准化至[-1, 1]，规定均值和标准差
])

class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self,root,train_file):
        self.root = root
        handle = open(train_file)
        lines = handle.readlines()
        self.prefixs = [item.split(',')[2] for item in lines[1:]]
        self.labels = [item.split(',')[3] for item in lines[1:]]
        
        # self.prefixs.sort()
        # self.prefixs = ['0001_2_01']
        # 构建label
        pass

    def __len__(self):
        return len(self.prefixs)
        # return 1

    def __getitem__(self, index):
        path = os.path.join(self.root,"{}.obj.obj".format(self.prefixs[index]))
        # 加载三维模型
        # mesh = trimesh.load_mesh(path)
        mesh = kal.rep.TriangleMesh.from_obj(path, enable_adjacency=True)
        vertices = mesh.vertices
        vertices = augment_data(vertices)
        # mesh.face_textures
        return np.float32(vertices),int(self.labels[index])


class Test_Dataset(torch.utils.data.Dataset):
    def __init__(self,root,train_file):
        self.root = root
        handle = open(train_file)
        lines = handle.readlines()
        
        self.prefixs = list(set([item.split(',')[2] for item in lines[1:]]+[item.split(',')[3] for item in lines[1:]]))
        self.labels = [item.split(',')[4] for item in lines[1:]]
        
        # 构建查询表
        self.query = [[self.prefixs.index(item.split(',')[2]),self.prefixs.index(item.split(',')[3]),int(item.split(',')[4])] for item in lines[1:]]
        
        # self.prefixs.sort()
        # self.prefixs = ['0001_2_01']
        # 构建label
        pass

    def __len__(self):
        return len(self.prefixs)
        # return 1

    def __getitem__(self, index):
        path = os.path.join(self.root,"{}.obj.obj".format(self.prefixs[index]))
        # 加载三维模型
        # mesh = trimesh.load_mesh(path)
        mesh = kal.rep.TriangleMesh.from_obj(path, enable_adjacency=True)
        vertices = mesh.vertices
        # mesh.face_textures
        return vertices
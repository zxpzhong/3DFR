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

import trimesh
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
        # mesh.face_textures
        return vertices,int(self.labels[index])


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
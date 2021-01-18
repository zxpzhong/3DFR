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

from torchvision import datasets, transforms
from base import BaseDataLoader

def img_open(path):
    img = Image.open(path)
    array = np.array(img)
    # array[:,:20,:] = 0
    # array[:,-20:,:] = 0
    img = Image.fromarray(array, mode='RGB')
    return img

class LFMB_3DFB_Pictures_Seged_Rectified(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset = Train_Dataset('/home/data/finger_vein/LFMB-3DFB_Pictures_Seged_Rectified_640_400/',self.data_dir)
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
        self.prefixs = list(set([item.split(',')[2] for item in lines[1:]]))
        self.prefixs.sort()
        # self.prefixs = ['0001_2_01']
        pass

    def __len__(self):
        return len(self.prefixs)
        # return 1

    def __getitem__(self, index):
        path = self.root+self.prefixs[index]
        img1 = img_open(path+"_A.bmp")
        img1 = transform_notrans(img1)
        # img1 = torch.cat([img1, img1, img1], 0)
        img2 = img_open(path+"_B.bmp")
        img2 = transform_notrans(img2)
        # img2 = torch.cat([img2, img2, img2], 0)
        img3 = img_open(path+"_C.bmp")
        img3 = transform_notrans(img3)
        # img3 = torch.cat([img3, img3, img3], 0)
        img4 = img_open(path+"_D.bmp")
        img4 = transform_notrans(img4)
        # img4 = torch.cat([img4, img4, img4], 0)
        img5 = img_open(path+"_E.bmp")
        img5 = transform_notrans(img5)
        # img5 = torch.cat([img5, img5, img5], 0)
        img6 = img_open(path+"_F.bmp")
        img6 = transform_notrans(img6)
        # img6 = torch.cat([img6, img6, img6], 0)

        # pil_img = torch.cat([pil_img, pil_img, pil_img], 0)
        # label = int(index)
        img = [img1,img2,img3,img4,img5,img6]
        mask1 = torch.where(torch.sum(img1,dim=0) > 0,torch.ones_like(torch.sum(img1,dim=0)) ,torch.zeros_like(torch.sum(img1,dim=0))).unsqueeze(-1)
        mask2 = torch.where(torch.sum(img2,dim=0) > 0,torch.ones_like(torch.sum(img2,dim=0)) ,torch.zeros_like(torch.sum(img2,dim=0))).unsqueeze(-1)
        mask3 = torch.where(torch.sum(img3,dim=0) > 0,torch.ones_like(torch.sum(img3,dim=0)) ,torch.zeros_like(torch.sum(img3,dim=0))).unsqueeze(-1)
        mask4 = torch.where(torch.sum(img4,dim=0) > 0,torch.ones_like(torch.sum(img4,dim=0)) ,torch.zeros_like(torch.sum(img4,dim=0))).unsqueeze(-1)
        mask5 = torch.where(torch.sum(img5,dim=0) > 0,torch.ones_like(torch.sum(img5,dim=0)) ,torch.zeros_like(torch.sum(img5,dim=0))).unsqueeze(-1)
        mask6 = torch.where(torch.sum(img6,dim=0) > 0,torch.ones_like(torch.sum(img6,dim=0)) ,torch.zeros_like(torch.sum(img6,dim=0))).unsqueeze(-1)
        mask = [mask1,mask2,mask3,mask4,mask5,mask6]
        for i in range(len(img)):
            img[i] = torch.cat([img[i],mask[i].permute(2,0,1)],dim=0)
        # print(path)
        return img,self.prefixs[index],mask
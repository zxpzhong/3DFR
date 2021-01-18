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
# uv贴图接口
from utils.uvmap import uv_map
import numpy as np
import os
import math
import trimesh

# 图卷积部分
from model.gbottleneck import GBottleneck
from model.gconv import GConv
from model.gpooling import GUnpooling
from model.gprojection import GProjection

# convmesh部分
from model.reconstruction import ReconstructionNetwork
from .mesh_template import MeshTemplate,MyMeshTemplate

# pytorch3d
from pytorch3d.structures import Meshes

# unet部分
from .unet import UNet

# 相机变换
import copy
def eulerAnglesToRotationMatrix(angles1) :
    theta = np.zeros((3, 1), dtype=np.float64)
    theta[0] = angles1[0]*3.141592653589793/180.0
    theta[1] = angles1[1]*3.141592653589793/180.0
    theta[2] = angles1[2]*3.141592653589793/180.0
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
R = eulerAnglesToRotationMatrix([35,20,-30])
# R = eulerAnglesToRotationMatrix([0,0,0])
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta) :
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

class Img_Embedding_Model(nn.Module):
    '''
    N视角的特征提取器
    '''

    def __init__(self, n_classes_input=24, pretrained=False):
        super(Img_Embedding_Model, self).__init__()

        self.features_dim = 960

        self.conv0_1 = nn.Conv2d(n_classes_input, 24, 3, stride=1, padding=1)
        self.conv0_2 = nn.Conv2d(24, 24, 3, stride=1, padding=1)

        self.conv1_1 = nn.Conv2d(24, 32, 3, stride=2, padding=1)  # 224 -> 112
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 112 -> 56
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 56 -> 28
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(128, 256, 5, stride=2, padding=2)  # 28 -> 14
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(256, 512, 5, stride=2, padding=2)  # 14 -> 7
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)


        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        img = F.relu(self.conv0_1(img))
        img = F.relu(self.conv0_2(img))
        # img0 = torch.squeeze(img) # 224

        img = F.relu(self.conv1_1(img))
        img = F.relu(self.conv1_2(img))
        img = F.relu(self.conv1_3(img))
        # img1 = torch.squeeze(img) # 112

        img = F.relu(self.conv2_1(img))
        img = F.relu(self.conv2_2(img))
        img = F.relu(self.conv2_3(img))
        img2 = img

        img = F.relu(self.conv3_1(img))
        img = F.relu(self.conv3_2(img))
        img = F.relu(self.conv3_3(img))
        img3 = img

        img = F.relu(self.conv4_1(img))
        img = F.relu(self.conv4_2(img))
        img = F.relu(self.conv4_3(img))
        img4 = img

        img = F.relu(self.conv5_1(img))
        img = F.relu(self.conv5_2(img))
        img = F.relu(self.conv5_3(img))
        img = F.relu(self.conv5_4(img))
        img5 = img

        return img5

class Renderer(nn.Module):
    '''
    上纹理+渲染
    '''
    def __init__(self,N = 6,f_dim=256, point_num = 1024):
        super(Renderer, self).__init__()
        '''
        初始化参数:
        '''
        self.N = N
        self.f_dim = f_dim
        self.point_num = point_num
        
        # DIB渲染器
        # self.renderer = DIBRenderer(height=640, width=400, mode='Lambertian',camera_fov_y=66.96 * np.pi / 180.0)
        self.renderer = DIBRenderer(height=256, width=256, mode='Lambertian',camera_fov_y=66.96 * np.pi / 180.0)
        self.renderer.set_look_at_parameters([0],[0],[0],fovx=57.77316 * np.pi / 180.0, fovy=44.95887 * np.pi / 180.0, near=0.01, far=10.0)
        # 设置相机参数
        # self.renderer.set_camera_parameters()
        # 预定义相机外参
        # 真实参数
        # 相机全部往中心移动
        # x = 0.597105
        # y = 1.062068
        # z = 1.111316
        # x = -0.3
        # y = 1.2
        # z = 0.1
        x = 0
        y = 0
        z = 0
        self.cam_mat = np.array([
            [[ 0.57432211  ,0.77105488  ,0.27500633],
 [-0.56542476,  0.13069975 , 0.81437854],
 [ 0.59198729 ,-0.62321099,  0.51103728]],
[[ 0.45602357 , 0.72700674,  0.51332611],
 [ 0.14605884 ,-0.63010819  ,0.76264702],
 [ 0.87790052 ,-0.2728092 , -0.39352995]],
[[ 0.60918383,  0.52822546 , 0.59150057],
 [ 0.73834933 ,-0.64995523, -0.17999572],
 [ 0.28937056 , 0.54638454 ,-0.78595714]],
[[ 7.71746128e-01 , 4.78767298e-01,  4.18556793e-01],
 [ 4.76878378e-01 ,-2.72559500e-04 ,-8.78969248e-01],
 [-4.20707651e-01,  8.77941797e-01, -2.28524119e-01]],
[[ 0.78888283,  0.55521065 , 0.2634483 ],
 [-0.15905217 , 0.5985437 , -0.78514193],
 [-0.59360448 , 0.57748297 , 0.56048831]],
[[ 0.71232121 , 0.68900052 , 0.13370407],
 [-0.69422699 , 0.71968522, -0.01010355],
 [-0.10318619 ,-0.085624  ,  0.99096979]]
        ])
        for i in range(6):
            self.cam_mat[i] = self.cam_mat[i]@R
        self.cam_mat = torch.FloatTensor(self.cam_mat)
        self.cameras_coordinate = [[2.50436065+x, -3.75589484+y, 1.88800446+z],
                            [4.02581981+x, -2.56894275+y, -3.29281609+z],
                            [1.01348544+x, 1.88043939+y, -5.4273143+z],
                            [-2.45261002+x, 3.5962286+y, -1.87506165+z],
                            [-3.12155638+x, 2.09254542+y, 2.21770186+z],
                            [-1.07692383+x, -1.37631717+y, 4.3081322+z]]
        xx = -0.3
        yy = 1.2
        zz = 0.1
        for i in range(6):
            self.cameras_coordinate[i] = np.array(self.cameras_coordinate[i])@R
            self.cameras_coordinate[i][0]+=xx
            self.cameras_coordinate[i][1]+=yy
            self.cameras_coordinate[i][2]+=zz
        self.cameras_coordinate = torch.FloatTensor(self.cameras_coordinate)
        if torch.cuda.is_available():
            self.cam_mat = torch.FloatTensor(self.cam_mat).cuda()
            self.cameras_coordinate = torch.FloatTensor(self.cameras_coordinate).cuda()
        # self.gif_writer = imageio.get_writer('example.gif', mode='I')
        pass

    def forward(self,vertex_positions,mesh_faces,input_uvs,input_texture,mesh_face_textures):
        '''
        输入: 
            vertices : 顶点数*3
            faces : 面数*3
            uv : uv坐标值 : 顶点数x2
            texture : NxCxHxW 图像
        输出:
            N个视角下的渲染图像:NxCxHxW
        '''
        images = []
        img_probs = []
        for i in range(self.N):
            # N个视角下渲染,每次渲染一个视角下的batchsize张图片
            # 设置相机参数
            camera_view_mtx = self.cam_mat[i].repeat(vertex_positions.shape[0],1,1)
            camera_view_shift = self.cameras_coordinate[i].repeat(vertex_positions.shape[0],1)
            self.renderer.camera_params = [camera_view_mtx, camera_view_shift, self.renderer.camera_params[2]]
            predictions, img_prob, _ = self.renderer(points=[vertex_positions, mesh_faces],
                                   uv_bxpx2=input_uvs,
                                   texture_bx3xthxtw=input_texture,
                                   ft_fx3=mesh_face_textures)
            predictions = torch.cat((predictions, img_prob), dim=3).permute(0, 3, 1, 2)
            temp = predictions.detach().cpu().numpy()[0]
            # self.gif_writer.append_data((temp * 255).astype(np.uint8))
            images.append(predictions)
            img_probs.append(img_prob)
        return images,img_probs
    def get_camera_info(self):
        return self.cam_mat,self.cameras_coordinate

class DR_3D_Model(nn.Module):
    r"""Differential Render based 3D Finger Reconstruction Model
        """
    def __init__(self,N = 6,f_dim=512, point_num = 962 , num_classes=1,ref_path = '/home/zf/vscode/3d/DR_3DFM/data/cylinder_template_mesh/uvsphere_31rings.obj'):
        super(DR_3D_Model, self).__init__()
        '''
        初始化参数:
        '''
        self.N = N
        self.f_dim = f_dim
        self.point_num = point_num
        
        # 参考mesh
        # 添加参考mesh信息
        self.meshtemp = MeshTemplate(ref_path, is_symmetric=False)
        
        # 图片特征提取网络
        self.img_embedding_model = Img_Embedding_Model()
        
        # 三维形变网络
        self.displace_net = ReconstructionNetwork(symmetric=False,
                                  texture_res=256,
                                  mesh_res=32,
                                  interpolation_mode = 'bilinear',
                                 )
        self.uvmap_net = ReconstructionNetwork(symmetric=False,
                                  texture_res=256,
                                  mesh_res=32,
                                  interpolation_mode = 'bilinear',
                                 )
        # self.displace_net = UNet(24,3)
        # self.uvmap_net = UNet(24,1)
        
        # 可微渲染器
        self.renderer = Renderer(N=self.N,f_dim=self.f_dim,point_num = self.point_num)
        
        # GAP
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        
        
    def forward(self, images):
        '''
        输入: N张图片 N list C*H*W
        输出: 对应N个视角的图片 : N list C*H*W
        '''
        # 为每张图像提取特征
        # embeddings = []
        # for i in range(len(images)):
        #     embeddings.append(self.img_embedding_model(images[i]))
        # features_cat = torch.zeros([embeddings[0].shape[0],self.f_dim*self.N]).cuda()
        # for i in range(len(embeddings)):
        #     x = self.GAP(embeddings[i])
        #     x = x.view(x.size(0), -1)
        #     features_cat[:,i*self.f_dim:(i+1)*self.f_dim] = x
        
        # 图像通道混合后直接unet
        features_cat = images[0]
        for i in range(len(images)-1):
            features_cat = torch.cat((features_cat,images[i+1]),dim=1)
        # 特征提取
        # features_cat = self.img_embedding_model(features_cat)
        # features_cat = self.GAP(features_cat)
        # features_cat = features_cat.view(features_cat.size(0), -1)
        # 将特征输入解码器,得到displacement map和uv map
        pred_tex,_ = self.displace_net(features_cat)
        _,mesh_map = self.uvmap_net(features_cat)
        
        raw_vtx = self.meshtemp.get_vertex_positions(mesh_map)

        vertex_positions,mesh_faces,input_uvs,input_texture,mesh_face_textures = self.meshtemp.forward_renderer(raw_vtx, pred_tex)
        
        repro_imgs,img_probs = self.renderer(vertex_positions,mesh_faces,input_uvs,input_texture,mesh_face_textures)

        new_mesh = Meshes(verts=raw_vtx, faces=self.meshtemp.mesh.faces.unsqueeze(0).repeat(raw_vtx.shape[0],1,1))
        return repro_imgs,raw_vtx,img_probs,self.meshtemp.mesh.faces,new_mesh,input_texture

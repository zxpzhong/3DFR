import os
from utils.Finger.tool import tools as tl
from utils.Finger.process import process_finger_data as pfd, faces_texture_mapping as ftm
import numpy as np
from utils.Finger.process import points_texture_mapping as tm


def init():
    # 六张bmp图片的像素信息，读取后放在全局变量中，避免每次都去重新读取
    tl.bmp_pixel = [[], [], [], [], [], []]
    # 哈希表，存储顶点对应的像素uv信息
    tl.map_vertex_to_texture = dict()

    # 哈希表,存储三角面片顶点对应的vt的index(行数)
    tl.map_vertex_to_vt_index = dict()
    # 每个相机对应的三角面片 如faces_belong_camera_A=[[1,3,5],[2,3,5]...]
    # faces_belong_camera_A = []
    # faces_belong_camera_B = []
    # faces_belong_camera_C = []
    # faces_belong_camera_D = []
    # faces_belong_camera_E = []
    # faces_belong_camera_F = []

    # 所有相机对应的三角面片，A相机放在0索引，以此类推
    tl.faces_belong_camera = [[], [], [], [], [], []]

    # 所有相机对应的bmp应该crop出的范围，[Umin,Vmin,Umax,Vmax],初始化时给相反的最大最小值,这里取的10000和-100，因为不可能有超过这个范围的了
    tl.bmp_crop_ranges = [[10000, 10000, -100, -100], [10000, 10000, -100, -100],
                    [10000, 10000, -100, -100], [10000, 10000, -100, -100],
                    [10000, 10000, -100, -100], [10000, 10000, -100, -100]]
    # 提前计算出crop的宽度u_width和高度v_height,先初始化为0
    tl.crops_width_and_height = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    # 在得到crops_width_and_height后，提前计算出各个相机crop出的图在png中v所占的范围比重（0-1），例如A：0-0.25，B：0.25-0.4...F：0.8-1
    tl.crops_v_scale_in_png = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    # uvmap_png的长度和宽度
    tl.uv_map_size = [0, 0]

    # face的索引 寻找bug时使用
    tl.face_index = 1


'''
入口参数：
    data_points : 数据点 , 二维numpy数组 , 第一维点数,第二维xyz坐标
    faces: 三角面片的组成索引 , 二维numpy数组, 第一维三角面片数目 , 第二维三角面片的三个索引
出口参数:
    uvmap : 1280*1600 二维numpy数组
    uv_val : uv值 , 二维numpy数组 ,第一维uv值数目, 第二维uv值
    vt_list : uv索引 , 二维numpy数组, 第一维三角面片数目, 第二维uv值的序号索引
'''
def uv_map(data_points,faces_point,imgs):
    '''
    输入:
        data_points : list : points number x 3
        file_path : 图片路径
    输出:
        uv map 图片 uv_map_png: CxHxW
        uv 值 uv_val_in_obj : uv数目 * 2
        uv索引 : 三角面片数目 * 3
    '''
    init()
    imgs = [np.array(item[0][0].cpu())*255 for item in imgs]
    obj_suffix = '.obj'
    # 拿到mesh所有顶点数据
    # data_points, face_start_index = pfd.read_mesh_points(file_path + obj_suffix)
    # 求出所有顶点对应的中心点O
    center_point = pfd.get_center_point(data_points)
    # 获取相机平面的参数ax+by+cz+d=0,直接使用计算好的数据
    camera_plane_para = tl.camera_plane_para
    # 获取中心点O的映射点
    center_point_mapping = tl.get_mapping_point_in_camera_plane(center_point, camera_plane_para)
    # 将mesh顶点数据中的所有顶点映射到相机平面
    data_points_mapping = pfd.get_data_points_mapping(data_points, camera_plane_para)
    # 数据预处理完毕，寻找每个点对应的相机；这里注意找到相机之后需要添加到源数据点上，而不是映射后的数据点
    camera_index_to_points = pfd.get_data_points_from_which_camera(center_point_mapping, data_points_mapping,
                                                                       tl.cameras_coordinate_mapping, data_points)
    # 纹理映射部分，这里和之前先后顺序不同，要从三角面片出发，得到每个面对应的相机，再将三角面片上的三个顶点投影到这个相机对应的bmp图片上，找到uv值
    # faces_point = pfd.read_mesh_faces(file_path + obj_suffix,face_start_index)  # 读取obj中face的顶点数据
    uv_map_png,uv_val_in_obj,vt_list = ftm.mapping_faces_gray(data_points,camera_index_to_points, faces_point+1, imgs)  # 拿到所有面的纹理区域
    # ftm.write_gray_to_obj(faces_texture, file_path)
    return uv_map_png/255,uv_val_in_obj,vt_list-1

# '通过面进行纹理映射'
if __name__ == '__main__':
    pass

    file_path = 'outer_files/LFMB_Visual_Hull_Meshes256/001_1_2_01'
    obj_suffix = '.obj'
    # 拿到mesh所有顶点数据
    data_points = pfd.read_mesh_points(file_path + obj_suffix)
    uv_map_png,uv_val_in_obj,vt_list = uv_map(data_points)
    print(uv_map_png.shape)
    print(len(uv_val_in_obj))
    print(len(vt_list))


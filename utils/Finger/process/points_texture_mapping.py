import numpy as np
from utils.Finger.tool import tools as tl
import cv2


# 基于点的纹理映射
# 根据处理好的数据和相机的内外参进行处理，后续可换成文件名形式进行读取数据
# 因为已经事先根据内外参得到了相机的投影矩阵，因此直接用投影矩阵计算
# 获取三维点在二维图像上的位置（u,v）
def get_uv_for_points(data_points, camera_index_and_uv):
    for i in range(len(data_points)):
        cur_uv = get_texture_for_single_point(data_points[i], camera_index_and_uv[i][0])
        camera_index_and_uv[i][1] = cur_uv[0]
        camera_index_and_uv[i][2] = cur_uv[1]
    return camera_index_and_uv


def get_texture_for_single_point(point_data, camera_index):
    '这里要注意用的是哪个投影矩阵'
    camera_projection_mat = tl.all_camera_projection_mat_640_400[camera_index]
    camera_projection_mat = np.mat(camera_projection_mat)
    # 根据公式，将点的x,y,z坐标变为4*1矩阵形式，最后补1
    point_mat = np.mat([[point_data[0]],
                        [point_data[1]],
                        [point_data[2]],
                        [1]])
    # 将3*4投影矩阵与4*1点的矩阵相乘，得到一个3*1的结果
    res = camera_projection_mat * point_mat
    u = res[0, 0] / res[2, 0]
    v = res[1, 0] / res[2, 0]
    # uv 取整
    u = round(u)  # todo  后续做插值而不是取整
    v = round(v)
    # point_data.append(u)
    # point_data.append(v)
    return [u, v]


# 获取所有数据点的灰度值
def mapping_points_gray(data_points, camera_index_and_uv, file_path):
    path_str = file_path.split("/")
    picture_path_prefix = 'outer_files/images/' + path_str[2]  # todo 注意这里的索引会随着文件路径改变而改变
    points_gray = []
    for i in range(len(data_points)):
        cur_gray = mapping_single_point_gray(data_points[i], camera_index_and_uv[i], picture_path_prefix)
        points_gray.append(cur_gray)
    return points_gray


# 获取单个点的灰度值
def mapping_single_point_gray(point, camera_index_and_uv, pic_path_prefix):
    camera_index = camera_index_and_uv[0]
    # camera_index = round(camera_index)
    camera_name = tl.camera_index_to_name[camera_index]
    pic_file_path = pic_path_prefix + '_' + camera_name + '.bmp'  # 拼接文件名
    u = camera_index_and_uv[1]
    v = camera_index_and_uv[2]
    # 归到正确的范围内
    if u > tl.cur_pic_size[0]:
        u = tl.cur_pic_size[0]
    if v > tl.cur_pic_size[1]:
        v = tl.cur_pic_size
    if u <= 0:
        u = 1
    if v <= 0:
        v = 1
    # 打开图片，根据uv获取灰度值
    gray = get_pic_gray(pic_file_path, camera_index, u, v)
    # point.append(gray)
    return gray


# 根据图片路径和像素u,v获取像素点的灰度值
def get_pic_gray(pic_file_path, camera_index, u, v):
    # cur_img = cv2.imread(pic_file_path, cv2.IMREAD_GRAYSCALE)  # 用opencv去读取bmp 直接拿到灰度值
    # 如果全局变量有当前的bmp像素，则直接去取，否则imread，然后放入全局变量中
    if len(tl.bmp_pixel[camera_index]) != 0:
        cur_img = tl.bmp_pixel[camera_index]
    else:
        cur_img = cv2.imread(pic_file_path)
        tl.bmp_pixel[camera_index] = cur_img
    # cur_img = cv2.imread(pic_file_path)
    gray = cur_img[v - 1][u - 1]  # 注意这里u，v和像素矩阵的索引是要反过来的，在图像坐标系中，u为横坐标，v为纵坐标，这里是v-1,u-1还是v u？
    # print("占用内存为：", sys.getsizeof(cur_img))
    return gray


# 根据灰度list和obj文件路径，将灰度值写入到新的obj文件中，完成纹理映射
def write_gray_to_obj(points_gray, obj_file_path):
    lines = []
    with open(obj_file_path + '.obj', 'r') as f:
        content = f.readlines()
        index = 0  # 记录位置
        # 跳过开头可能出现的几行信息
        while index < 5 and content[index] and content[index][0] != 'v':  # 避免开头可能出现的信息
            index += 1
        for index, j in zip(range(index, len(content)), range(0, len(points_gray))):
            line = content[index]
            gray = points_gray[j]
            line = line[0:-1] + " " + str(gray[0]) + " " + str(gray[1]) + " " + str(gray[2]) + '\n'
            lines.append(line)
        index += 1
        i = index
        # 跳过顶点数据和面数据之间可能出现的空行
        while index < i + 5 and content[index] and content[index][0] != 'f':
            index += 1
        for index in range(index, len(content)):
            line = content[index]
            lines.append(line)
    with open(obj_file_path + '_new.obj', 'w+') as f_new:
        f_new.writelines(lines)

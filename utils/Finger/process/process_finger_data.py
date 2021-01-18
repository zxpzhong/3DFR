import numpy as np
from utils.Finger.tool import tools as tl
import sys


# 根据obj文件获得mesh的顶点数据
# 数据点的数据结构选择list而不是数组,方便后续拓展
def read_mesh_points(obj_file_path):
    points = []
    # try:
    with open(obj_file_path) as file:
        lines = file.readlines()
        i = 0
        for i in range(0, len(lines)):
            if not lines[i]:
                break
            strs = lines[i].split(" ")
            if i <= 5 and strs[0] != "v":  # 避免开头可能出现的信息
                continue
            if strs[0] == 'v':
                cur = [float(strs[1]), float(strs[2]), float(strs[3])]
                points.append(cur)
            else:
                break
    # except Exception as e:
    #     print("错误:", e)
    #     sys.exit(1)
    points = np.array(points)
    return points, i  # 这里返回的i方便后续直接定位面所在的数据，避免重复遍历


# 根据obj文件获得mesh的面数据
def read_mesh_faces(obj_file_path, face_start_index):
    with open(obj_file_path) as file:
        faces = []
        lines = file.readlines()
        for i in range(face_start_index, len(lines)):
            if not lines[i]:
                break
            strs = lines[i].split(" ")
            if strs[0] == "v" or strs[0] == "\n":
                continue
            elif strs[0] == "f":
                cur = [int(strs[1]), int(strs[2]), int(strs[3])]
                faces.append(cur)
            else:
                break
    faces = np.array(faces)
    return faces


# 读取已处理好的txt数据（含uv）
def read_uv_points(txt_file_path):
    with open(txt_file_path) as file:
        uv_points = []
        while 1:
            line = file.readline()
            if not line:
                break
            str = line.split(" ")
            if str[0]:
                cur = [float(str[3]), float(str[4]), float(str[5])]
                uv_points.append(cur)
            else:
                break
    return uv_points


# 获得mesh的中心点数据
def get_center_point(points):
    x_total = 0
    y_total = 0
    z_total = 0
    for p in points:
        x_total += p[0]
        y_total += p[1]
        z_total += p[2]
    size = len(points)
    center_point = [x_total / size, y_total / size, z_total / size]
    center_point = np.array(center_point)
    return center_point


# 获取所有相机在世界坐标系下的坐标
def get_cameras_coordinate():
    camera_origins = [get_single_camera_origin(tl.camera_a_outer_para),
                      get_single_camera_origin(tl.camera_b_outer_para),
                      get_single_camera_origin(tl.camera_c_outer_para),
                      get_single_camera_origin(tl.camera_d_outer_para),
                      get_single_camera_origin(tl.camera_e_outer_para),
                      get_single_camera_origin(tl.camera_f_outer_para)]
    # 将list转为array
    camera_origins = np.array(camera_origins)
    return camera_origins


# 根据相机外参获得相机在世界坐标系下的坐标
def get_single_camera_origin(m2):
    # 先对外参矩阵求逆
    m2 = m2.I
    m2 = np.array(m2)
    # 按公式可得坐标就是逆矩阵中每行的最后一个元素
    origin = m2[:3, 3]  # 取前三行第四个元素即可
    return origin


# 获取相机平面ax+by+cz+d=0
def get_camera_plane(cameras_coordinate):
    # B相机坐标
    x1 = cameras_coordinate[1][0]
    y1 = cameras_coordinate[1][1]
    z1 = cameras_coordinate[1][2]
    # C相机坐标
    x2 = cameras_coordinate[2][0]
    y2 = cameras_coordinate[2][1]
    z2 = cameras_coordinate[2][2]
    # D相机坐标
    x3 = cameras_coordinate[3][0]
    y3 = cameras_coordinate[3][1]
    z3 = cameras_coordinate[3][2]

    a = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
    b = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
    c = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    d = -a * x1 - b * y1 - c * z1
    plane_para = [a, b, c, d]
    return plane_para


# 数据集所有点映射到平面
def get_data_points_mapping(data_points, camera_plane_para):
    data_points_mapping = []
    for i in range(len(data_points)):
        cur_point_mapping = tl.get_mapping_point_in_camera_plane(data_points[i], camera_plane_para)
        data_points_mapping.append(cur_point_mapping)
    return data_points_mapping


# 方法一 根据映射投影矢量之间夹角最小来判断mesh上所有顶点来自哪个摄像机拍摄（与哪个摄像机最近）
def get_data_points_from_which_camera(center_point, data_points_mapping, cameras_coordinate_mapping, data_points):
    # 设当前顶点为N，中心点为0，两个相邻的相机为X，Y。则判断方法为
    # 根据O_N_向量与OA,OB...等向量夹角，找到夹角最小的相机即为所选择

    # 计算一下每个相机拥有的点的个数，判断是否均衡(发现基本均衡)
    # camera_index_count = [0, 0, 0, 0, 0, 0]
    # 相机与点的一一对应数组(更改输入输出格式后，由于数组不可拓展，创建新的相机数组与点的数组进行对应)
    camera_index_to_points = np.zeros(len(data_points), dtype=np.int)
    for i in range(len(data_points_mapping)):
        cur_target_camera_index = get_single_point_from_which_camera(center_point, data_points_mapping[i],
                                                                     cameras_coordinate_mapping)
        camera_index_to_points[i] = cur_target_camera_index
        # data_points[i].append(cur_target_camera_index)  # 将找到的相机添加在当前数据后面
        # camera_index_count[cur_target_camera_index] += 1
    # print("每个相机出现的次数为：", camera_index_count)  # 分别为38, 49, 51, 36, 40, 42
    return camera_index_to_points  # 这里返回的应该是源数据 而不是映射数据


def get_data_points_from_which_camera2(center_point, data_points_mapping, cameras_coordinate_mapping, data_points):
    camera_index_and_uv = np.zeros([len(data_points), 3], dtype=np.int)
    for i in range(len(data_points_mapping)):
        cur_target_camera_index = get_single_point_from_which_camera(center_point, data_points_mapping[i],
                                                                     cameras_coordinate_mapping)
        camera_index_and_uv[i][0] = cur_target_camera_index
        # data_points[i].append(cur_target_camera_index)  # 将找到的相机添加在当前数据后面
        # camera_index_count[cur_target_camera_index] += 1
    # print("每个相机出现的次数为：", camera_index_count)  # 分别为38, 49, 51, 36, 40, 42
    return camera_index_and_uv  # 这里返回的应该是源数据 而不是映射数据


# 判断mesh上单一顶点来自哪个摄像机拍摄（与哪个摄像机最近）
# 根据ON向量与OA,OB,...向量夹角比较，夹角越小，余弦值越大，即为所需
def get_single_point_from_which_camera(center_point, cur_point, cameras_coordinate):
    cur_vector = calculate_vector(center_point, cur_point)
    max_vector_cosine = -2  # 初始化最大值为一个很小的值，余弦值为[-1,1]
    target_camera_index = 0  # 初始化A相机是所求的相机下标，0代表A，1代表B 以此类推
    for i in range(len(cameras_coordinate)):
        camera_vector = calculate_vector(center_point, cameras_coordinate[i])
        cur_cosine = tl.calculate_cosine(cur_vector, camera_vector)
        if cur_cosine > max_vector_cosine:
            max_vector_cosine = cur_cosine
            target_camera_index = i
    return target_camera_index


# 方法二 根据叉乘（向量积）来判断点来自于哪个相机
def get_point_from_which_camera2(cur_point, center_point, camera_points):
    cur_vector = calculate_vector(center_point, cur_point)
    count = 0
    for i in range(len(camera_points)):
        camera_vector1 = calculate_vector(center_point, camera_points[i])
        if i != len(camera_points) - 1:
            camera_vector2 = calculate_vector(center_point, camera_points[i + 1])
        else:
            camera_vector2 = calculate_vector(center_point, camera_points[0])  # F相机和A相机的情况
        vector_product1 = tl.calculate_vector_product(camera_vector1, cur_vector)
        vector_product2 = tl.calculate_vector_product(camera_vector2, cur_vector)
        # 判断计算出的两个向量积的夹角
        if tl.calculate_cosine(vector_product1, vector_product2) <= 0:
            count += 1  # 只是判断是否存在一个点的值小于0
    return count


# 根据两个点计算向量
def calculate_vector(from_point, to_point):
    vector = [to_point[0] - from_point[0], to_point[1] - from_point[1], to_point[2] - from_point[2]]
    return vector

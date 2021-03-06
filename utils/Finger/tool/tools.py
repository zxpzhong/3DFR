# 定义全局变量和方法
import numpy as np
import math
# import process.process_finger_data as pfd

# 目前选用的图片尺寸
cur_pic_size = [640, 400]
# cur_pic_size = [1280, 800]
# 相机索引对应相机名称
camera_index_to_name = ['A', 'B', 'C', 'D', 'E', 'F']
# 6个相机的外参
camera_a_outer_para = np.mat([[0.574322111, 0.771054881, 0.275006333, 0.93847817],
                              [0.565423192, -0.130698104, -0.814379899, -0.36935905],
                              [-0.591988790, 0.623211341, -0.511035123, 4.78810628],
                              [0, 0, 0, 1]])
camera_b_outer_para = np.mat([[0.456023570, 0.727006744, 0.513326112, 1.72205846],
                              [-0.146061166, 0.630108915, -0.762645980, -0.30452329],
                              [-0.877900131, 0.272807532, 0.393531969, 5.53092307],
                              [0, 0, 0, 1]])
camera_c_outer_para = np.mat([[0.609183831, 0.528225460, 0.591500569, 1.59956459],
                              [-0.738350101, 0.649953779, 0.179997814, 0.5030131],
                              [-0.289368602, -0.546386263, 0.785956655, 5.58635091],
                              [0, 0, 0, 1]])
camera_d_outer_para = np.mat([[0.771746127, 0.478767298, 0.418556793, 0.955855425],
                              [-0.476877262, 0.000270229651, 0.878969854, 0.477556906],
                              [0.420708915, -0.877941799, 0.228521787, 4.61760675],
                              [0, 0, 0, 1]])
camera_e_outer_para = np.mat([[0.788882832, 0.555210653, 0.263448302, 0.71648894],
                              [0.159053746, -0.598545227, 0.785140445, 0.00777088],
                              [0.593604063, -0.577481378, -0.560490387, 4.30437514],
                              [0, 0, 0, 1]])
camera_f_outer_para = np.mat([[0.712321206, 0.689000523, 0.133704068, 1.13938413],
                              [0.694227260, -0.719684989, 0.0101009224, -0.28640104],
                              [0.103184351, 0.0856259076, -0.990969825, 4.49819911],
                              [0, 0, 0, 1]])

# 六个相机的内参
camera_a_inner_para = np.mat([[967.5377197, 0, 703.1273732, 0],
                              [0, 967.9393921, 351.0187561, 0],
                              [0, 0, 1, 0]])
camera_b_inner_para = np.mat([[963.2991943, 0, 589.8122291, 0],
                              [0, 962.7422485, 412.5244055, 0],
                              [0, 0, 1, 0]])
camera_c_inner_para = np.mat([[967.4086914, 0, 612.7826353, 0],
                              [0, 968.0758667, 451.7366286, 0],
                              [0, 0, 1, 0]])
camera_d_inner_para = np.mat([[961.0868530, 0, 692.7282436, 0],
                              [0, 960.6126708, 417.4375162, 0],
                              [0, 0, 1, 0]])
camera_e_inner_para = np.mat([[955.4882812, 0, 730.3056525, 0],
                              [0, 953.7589722, 451.5117967, 0],
                              [0, 0, 1, 0]])
camera_f_inner_para = np.mat([[962.0779419, 0, 595.2503222, 0],
                              [0, 961.0998535, 396.8389609, 0],
                              [0, 0, 1, 0]])

# 六个相机的投影矩阵为 投影矩阵=内参x外参
# 所有相机的投影矩阵放到一个三维矩阵里(1280x800)
all_camera_projection_mat = [
    [[1.39434783e+02, 1.18422163e+03, -9.32437833e+01, 4.27466162e+03],
     [3.39496212e+02, 9.22510264e+01, -9.67653298e+02, 1.32319794e+03],
     [-5.91988790e-01, 6.23211341e-01, -5.11035123e-01, 4.78810628e+00]],
    [[-7.85090956e+01, 8.61230229e+02, 7.26596598e+02, 4.92106359e+03],
     [-5.02774485e+02, 7.19172239e+02, -5.71889964e+02, 1.98846331e+03],
     [-8.77900131e-01, 2.72807532e-01, 3.93531969e-01, 5.53092307e+00]],
    [[4.12009678e+02, 1.76193887e+02, 1.05384338e+03, 4.97065152e+03],
     [-8.45497311e+02, 3.82381880e+02, 5.29296949e+02, 3.01051417e+03],
     [-2.89368602e-01, -5.46386263e-01, 7.85956655e-01, 5.58635091e+00]],
    [[1.03315200e+03, -1.48038125e+02, 5.60572927e+02, 4.11740670e+03],
     [-2.82474656e+02, -3.66226258e+02, 9.39743146e+02, 2.38630951e+03],
     [4.20708915e-01, -8.77941799e-01, 2.28521787e-01, 4.61760675e+00]],
    [[1.18728070e+03, 1.08759358e+02, -1.57607533e+02, 3.82810628e+03],
     [4.19718174e+02, -8.31607535e+02, 4.95766722e+02, 1.95088770e+03],
     [5.93604063e-01, -5.77481378e-01, -5.60490387e-01, 4.30437514e+00]],
    [[7.46729038e+02, 7.13841054e+02, -4.61241373e+02, 3.77373081e+03],
     [7.08169289e+02, -6.57709441e+02, -3.83547441e+02, 1.50980066e+03],
     [1.03184351e-01, 8.56259076e-02, -9.90969825e-01, 4.49819911e+00]]
]
# camera_a_projection_mat = np.mat([[1.39434783e+02, 1.18422163e+03, -9.32437833e+01, 4.27466162e+03],
#                                   [3.39496212e+02, 9.22510264e+01, -9.67653298e+02, 1.32319794e+03],
#                                   [-5.91988790e-01, 6.23211341e-01, -5.11035123e-01, 4.78810628e+00]])
#
# camera_b_projection_mat = np.mat([[-7.85090956e+01, 8.61230229e+02, 7.26596598e+02, 4.92106359e+03],
#                                   [-5.02774485e+02, 7.19172239e+02, -5.71889964e+02, 1.98846331e+03],
#                                   [-8.77900131e-01, 2.72807532e-01, 3.93531969e-01, 5.53092307e+00]])
#
# camera_c_projection_mat = np.mat([[4.12009678e+02, 1.76193887e+02, 1.05384338e+03, 4.97065152e+03],
#                                   [-8.45497311e+02, 3.82381880e+02, 5.29296949e+02, 3.01051417e+03],
#                                   [-2.89368602e-01, -5.46386263e-01, 7.85956655e-01, 5.58635091e+00]])
#
# camera_d_projection_mat = np.mat([[1.03315200e+03, -1.48038125e+02, 5.60572927e+02, 4.11740670e+03],
#                                   [-2.82474656e+02, -3.66226258e+02, 9.39743146e+02, 2.38630951e+03],
#                                   [4.20708915e-01, -8.77941799e-01, 2.28521787e-01, 4.61760675e+00]])
#
# camera_e_projection_mat = np.mat([[1.18728070e+03, 1.08759358e+02, -1.57607533e+02, 3.82810628e+03],
#                                   [4.19718174e+02, -8.31607535e+02, 4.95766722e+02, 1.95088770e+03],
#                                   [5.93604063e-01, -5.77481378e-01, -5.60490387e-01, 4.30437514e+00]])
#
# camera_f_projection_mat = np.mat([[7.46729038e+02, 7.13841054e+02, -4.61241373e+02, 3.77373081e+03],
#                                   [7.08169289e+02, -6.57709441e+02, -3.83547441e+02, 1.50980066e+03],
#                                   [1.03184351e-01, 8.56259076e-02, -9.90969825e-01, 4.49819911e+00]])

# 将图片缩小为640*400后的相机内参为: 四个参数都除以二
camera_a_inner_para_640_400 = np.mat([[483.76885985, 0, 351.5636866, 0],
                                      [0, 483.96969605, 175.50937805, 0],
                                      [0, 0, 1, 0]])
camera_b_inner_para_640_400 = np.mat([[481.64959715, 0, 294.90611455, 0],
                                      [0, 481.37112425, 206.26220275, 0],
                                      [0, 0, 1, 0]])
camera_c_inner_para_640_400 = np.mat([[483.7043457, 0, 306.39131765, 0],
                                      [0, 484.03793335, 225.8683143, 0],
                                      [0, 0, 1, 0]])
camera_d_inner_para_640_400 = np.mat([[480.5434265, 0, 346.3641218, 0],
                                      [0, 480.3063354, 208.7187581, 0],
                                      [0, 0, 1, 0]])
camera_e_inner_para_640_400 = np.mat([[477.7441406, 0, 365.15282625, 0],
                                      [0, 476.8794861, 225.75589835, 0],
                                      [0, 0, 1, 0]])
camera_f_inner_para_640_400 = np.mat([[481.03897095, 0, 297.6251611, 0],
                                      [0, 480.54992675, 198.41948045, 0],
                                      [0, 0, 1, 0]])
# 将图片resize为640*400后的投影矩阵
all_camera_projection_mat_640_400 = [
    [[6.97173914e+01, 5.92110817e+02, - 4.66218917e+01, 2.13733081e+03],
     [1.69748106e+02, 4.61255132e+01, - 4.83826649e+02, 6.61598968e+02],
     [-5.91988790e-01, 6.23211341e-01, - 5.11035123e-01, 4.78810628e+00]],
    [[-3.92545478e+01, 4.30615115e+02, 3.63298299e+02, 2.46053180e+03],
     [-2.51387243e+02, 3.59586119e+02, - 2.85944982e+02, 9.94231657e+02],
     [-8.77900131e-01, 2.72807532e-01, 3.93531969e-01, 5.53092307e+00]],
    [[2.06004839e+02, 8.80969434e+01, 5.26921691e+02, 2.48532576e+03],
     [-4.22748655e+02, 1.91190940e+02, 2.64648475e+02, 1.50525708e+03],
     [-2.89368602e-01, - 5.46386263e-01, 7.85956655e-01, 5.58635091e+00]],
    [[5.16576002e+02, - 7.40190623e+01, 2.80286464e+02, 2.05870335e+03],
     [-1.41237328e+02, - 1.83113129e+02, 4.69871573e+02, 1.19315475e+03],
     [4.20708915e-01, - 8.77941799e-01, 2.28521787e-01, 4.61760675e+00]],
    [[5.93640352e+02, 5.43796790e+01, - 7.88037663e+01, 1.91405314e+03],
     [2.09859087e+02, - 4.15803768e+02, 2.47883361e+02, 9.75443850e+02],
     [5.93604063e-01, - 5.77481378e-01, - 5.60490387e-01, 4.30437514e+00]],
    [[3.73364519e+02, 3.56920527e+02, - 2.30620687e+02, 1.88686540e+03],
     [3.54084644e+02, - 3.28854721e+02, - 1.91773720e+02, 7.54900332e+02],
     [1.03184351e-01, 8.56259076e-02, - 9.90969825e-01, 4.49819911e+00]]
]

# 六个相机在世界坐标系下的坐标
cameras_coordinate = [[2.50436065, -3.75589484, 1.88800446],
                      [4.02581981, -2.56894275, -3.29281609],
                      [1.01348544, 1.88043939, -5.4273143],
                      [-2.45261002, 3.5962286, -1.87506165],
                      [-3.12155638, 2.09254542, 2.21770186],
                      [-1.07692383, -1.37631717, 4.3081322]]
# 六个相机组成的空间平面方程参数 AX+BY+CZ+D=0
camera_plane_para = [19.467678495159983, 18.098947303577706, 10.253452426300939, 1.884526845005233]

# 六个相机映射到同一平面后的相机坐标,这里选用的是BCD三个相机作为相机平面，因此只需要将AEF映射到平面
cameras_coordinate_mapping = [[2.45592658, -3.80092362, 1.86249467],
                              [4.02581981, -2.56894275, -3.29281609],
                              [1.01348544, 1.88043939, -5.4273143],
                              [-2.45261002, 3.5962286, -1.87506165],
                              [-3.16297766, 2.05403639, 2.19588564],
                              [-1.08130466, -1.38038999, 4.30582486]]

# 六张bmp图片的像素信息，读取后放在全局变量中，避免每次都去重新读取
bmp_pixel = [[], [], [], [], [], []]
# 哈希表，存储顶点对应的像素uv信息
map_vertex_to_texture = dict()

# 哈希表,存储三角面片顶点对应的vt的index(行数)
map_vertex_to_vt_index = dict()
# 每个相机对应的三角面片 如faces_belong_camera_A=[[1,3,5],[2,3,5]...]
# faces_belong_camera_A = []
# faces_belong_camera_B = []
# faces_belong_camera_C = []
# faces_belong_camera_D = []
# faces_belong_camera_E = []
# faces_belong_camera_F = []

# 所有相机对应的三角面片，A相机放在0索引，以此类推
faces_belong_camera = [[], [], [], [], [], []]

# 所有相机对应的bmp应该crop出的范围，[Umin,Vmin,Umax,Vmax],初始化时给相反的最大最小值,这里取的10000和-100，因为不可能有超过这个范围的了
bmp_crop_ranges = [[10000, 10000, -100, -100], [10000, 10000, -100, -100],
                   [10000, 10000, -100, -100], [10000, 10000, -100, -100],
                   [10000, 10000, -100, -100], [10000, 10000, -100, -100]]
# 提前计算出crop的宽度u_width和高度v_height,先初始化为0
crops_width_and_height = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
# 在得到crops_width_and_height后，提前计算出各个相机crop出的图在png中v所占的范围比重（0-1），例如A：0-0.25，B：0.25-0.4...F：0.8-1
crops_v_scale_in_png = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
# uvmap_png的长度和宽度
uv_map_size = [0, 0]

# face的索引 寻找bug时使用
face_index = 1


# 打印数据点
def print_data_points(data_points):
    for li in data_points:
        print(li)


# 计算两个向量的夹角的余弦
# 公式为cos<a,b>=a.b/|a||b|. a.b=(x1x2+y1y2+z1z2) |a|=√(x1^2+y1^2+z1^2), |b|=√(x2^2+y2^2+z2^2).
def calculate_cosine(vector1, vector2):
    a = vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]
    b = math.sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1] + vector1[2] * vector1[2])
    c = math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1] + vector2[2] * vector2[2])
    res = a / (b * c)
    return res


# 计算两个向量的向量积
# AB=(x1,y1,z1)  CD=(x2,y2,z2) cross(AB,CD)=(y1*z2-y2z1,z1x2-z2x1,x1y2-x2y1)
def calculate_vector_product(vector1, vector2):
    vector_product = [vector1[1] * vector2[2] - vector1[2] * vector2[1],
                      vector1[2] * vector2[0] - vector1[0] * vector2[2],
                      vector1[0] * vector2[1] - vector1[1] * vector2[0]]
    return vector_product


# 点到空间平面的映射点（投影）
def get_mapping_point_in_camera_plane(point, camera_plane_para):
    a = camera_plane_para[0]
    b = camera_plane_para[1]
    c = camera_plane_para[2]
    d = camera_plane_para[3]
    x = point[0]
    y = point[1]
    z = point[2]
    # 避免重复计算，不知python是否已有优化
    a_ = a * a
    b_ = b * b
    c_ = c * c
    temp = a_ + b_ + c_
    x_ = ((b_ + c_) * x - a * (b * y + c * z + d)) / temp
    y_ = ((a_ + c_) * y - b * (a * x + c * z + d)) / temp
    z_ = ((a_ + b_) * z - c * (a * x + b * y + d)) / temp
    point_ = [x_, y_, z_]
    return point_


# # 全局变量中部分数据的由来（在main函数中直接使用了）（因为外参已经固定，所以部分数据基本不会改变，减少计算量）
# def pre_process():
#     # 求出六个相机在世界坐标系下的坐标
#     cameras_coordinate = pfd.get_cameras_coordinate()
#     # 求出相机参数平面
#     camera_plane_para = pfd.get_camera_plane(cameras_coordinate)
#     # 获取A，E，F的映射点
#     camera_a_point = get_mapping_point_in_camera_plane(cameras_coordinate[0], camera_plane_para)
#     camera_e_point = get_mapping_point_in_camera_plane(cameras_coordinate[4], camera_plane_para)
#     camera_f_point = get_mapping_point_in_camera_plane(cameras_coordinate[5], camera_plane_para)
#     # 六个相机归到一个平面之后的坐标：BCD不变，AEF映射到BCD平面
#     camera_point_mapping = [camera_a_point, cameras_coordinate[1], cameras_coordinate[2],
#                             cameras_coordinate[3], camera_e_point, camera_f_point]
#     camera_point_mapping = np.array(camera_point_mapping)

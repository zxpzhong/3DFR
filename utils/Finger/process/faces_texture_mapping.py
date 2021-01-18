import numpy as np
from utils.Finger.tool import tools as tl
import cv2
import time
import matplotlib.pyplot as plt


# 面的纹理映射
def mapping_faces_gray(data_points, camera_index_to_points, faces_point, imgs):
    # 得出每个三角面片都属于哪一个相机，两个点及以上属于一个相机，则该面属于该相机
    camera_index_to_faces = get_faces_belong_which_camera(camera_index_to_points, faces_point)

    # todo 考虑是否需要将这个写入本地文件
    # 拿到三角面片对应的相机后，对该三角面片做相应图片的映射，三维——>二维
    for face, camera_index in zip(faces_point, camera_index_to_faces):
        get_texture_from_bmp(face, camera_index, data_points, imgs)  # 会将所有三角面片对应点的纹理全放进哈希表中,同时对面片按相机分类
    start = time.time()
    # 根据全局变量bmp_crop_ranges，去对bmp图片做crop，然后放入uv map png图片中
    uv_map_png = crop_bmp_to_png(imgs)

    # 对每个面进行遍历，获取面上的点再uvmap_png中的对应uv值，然后按预期格式会写到obj文件中
    uv_val_in_obj, vt_list = get_png_uv_from_crops(faces_point, camera_index_to_faces)
    write_uv_to_obj(uv_val_in_obj, vt_list)
    return uv_map_png,uv_val_in_obj, vt_list

# 获得三角面片属于什么相机
def get_faces_belong_which_camera(camera_index_to_points, faces_point):
    camera_index_to_faces = np.zeros(len(faces_point), dtype=int)
    for i in range(0, len(faces_point)):
        face = faces_point[i]
        face_with_camera = []
        for v in face:
            camera_index = camera_index_to_points[v - 1]  # 因为obj中的v都是从1开始的，因此减一
            face_with_camera.append(camera_index)
        max_count_camera = max(face_with_camera, key=face_with_camera.count)  # 得出列表中出现最多次数的相机索引
        camera_index_to_faces[i] = max_count_camera
        # face.append(max_count_camera)
        # camera_index_to_faces.append(face)
    return camera_index_to_faces


def get_texture_from_bmp(face, camera_index, data_points, file_path):
    # 用face里面存储的点的索引，去data_points_contain_camera里拿到对应的数据点
    # camera_index = face[3]
    # 将face根据不同的相机放进全局变量
    tl.faces_belong_camera[camera_index].append(face)  # 该变量目前还没有用到
    # todo 将单独每个三角面crop出来，看看纹理是否符合预期
    face_vertex = []
    for vertex_index in face:  # 注意这里只有前三个才是顶点索引
        vertex_data = data_points[vertex_index - 1]
        cur_vertex = get_texture_for_vertex(vertex_data, camera_index, vertex_index)
        face_vertex.append(cur_vertex)
    # 查看单一面片的纹理crop
    '该函数及该函数所用到的变量只用于debug'
    # show_single_face_crop(face_vertex, camera_index, file_path)


# 这个方法和之前求纹理的方法get_texture_for_single_point类似，只是输入输出参数不同
def get_texture_for_vertex(vertex_data, camera_index, vertex_index):
    key = str(camera_index) + "_" + str(vertex_index)
    # 判断哈希表中是否已经存在该数据，避免重复计算
    if key not in tl.map_vertex_to_texture.keys():
        camera_projection_mat = tl.all_camera_projection_mat_640_400[camera_index]
        camera_projection_mat = np.mat(camera_projection_mat)
        # 根据公式，将点的x,y,z坐标变为4*1矩阵形式，最后补1
        point_mat = np.mat([[vertex_data[0]],
                            [vertex_data[1]],
                            [vertex_data[2]],
                            [1]])
        # 将3*4投影矩阵与4*1点的矩阵相乘，得到一个3*1的结果
        res = camera_projection_mat * point_mat
        u = res[0, 0] / res[2, 0]
        v = res[1, 0] / res[2, 0]
        # uv 取整
        # todo 为什么u会出现负数
        u = round(u)
        v = round(v)
        # todo  uv 取整时不应该超过uv的应有范围，后续还是应该采用精度更高的做法，另外uv和像素矩阵的对应关系也应该确定是否是v-1.u-1
        # 由于(u,v)只代表像素的列数与行数,而四舍五入存在误差，为了不超过uv的范围，将它强行归到1-1280/640范围，1-800/400范围
        if u > tl.cur_pic_size[0]:
            u = tl.cur_pic_size[0]
        if v > tl.cur_pic_size[1]:
            v = tl.cur_pic_size[1]
        if u <= 0:
            u = 1
        if v <= 0:
            v = 1
        # 根据相机索引和像素点下标拼接key值，然后将uv放到哈希表中
        tl.map_vertex_to_texture[key] = [u, v]
        # 同时更新全局变量中的uv crop范围 umin vmin umax vmax
        tl.bmp_crop_ranges[camera_index][0] = min(u, tl.bmp_crop_ranges[camera_index][0])
        tl.bmp_crop_ranges[camera_index][1] = min(v, tl.bmp_crop_ranges[camera_index][1])
        tl.bmp_crop_ranges[camera_index][2] = max(u, tl.bmp_crop_ranges[camera_index][2])
        tl.bmp_crop_ranges[camera_index][3] = max(v, tl.bmp_crop_ranges[camera_index][3])
        return [u, v]
    else:
        return tl.map_vertex_to_texture[key]


# 显示单个三角面片，单纯用于debug，出现错误时注释掉即可
def show_single_face_crop(face_vertex, camera_index, file_path):
    u_min = min(face_vertex[0][0], face_vertex[1][0], face_vertex[2][0])
    v_min = min(face_vertex[0][1], face_vertex[1][1], face_vertex[2][1])
    u_max = max(face_vertex[0][0], face_vertex[1][0], face_vertex[2][0])
    v_max = max(face_vertex[0][1], face_vertex[1][1], face_vertex[2][1])
    crop_range = [u_min, v_min, u_max, v_max]
    crop_img = crop_bmp(crop_range, camera_index, file_path)
    # cv2.imwrite( str(camera_index) + '_' + str(tl.face_index) + 'crop.png', crop_img)
    file_path += '/' + str(camera_index) + '_' + str(tl.face_index) + '_crop.png'
    cv2.imwrite(file_path, crop_img)
    tl.face_index += 1
    '这个index可以让我们指定在代码运行到第index个三角面片的时候，方便我们调试'
    # if tl.face_index >= 715:
    #     plt.imshow(crop_img, cmap="gray")  # todo 为什么这样会变黑
    #     plt.show()
    # plt.imshow(crop_img, cmap="gray")  # todo 为什么这样会变黑
    # plt.show()


def crop_bmp_to_png(imgs):
    # 初始化png大小，全0,png组成为A B C D E F 竖排摆放
    # 计算各相机crop出的宽度和高度
    calculate_crop_width_and_height()
    png_width = max(tl.crops_width_and_height[0][0], tl.crops_width_and_height[1][0], tl.crops_width_and_height[2][0],
                    tl.crops_width_and_height[3][0], tl.crops_width_and_height[4][0], tl.crops_width_and_height[5][0])
    png_height = tl.crops_width_and_height[0][1] + tl.crops_width_and_height[1][1] + tl.crops_width_and_height[2][1] + \
                 tl.crops_width_and_height[3][1] + tl.crops_width_and_height[4][1] + tl.crops_width_and_height[5][1]
    uv_map_png = np.zeros((int(png_height), int(png_width)), dtype=np.uint8)
    # 放入全局变量
    tl.uv_map_size[:] = png_width, png_height
    # 计算出crop的v在png中所占的比重范围
    calculate_crop_v_scale_in_png()
    target_gray = 0
    start = time.time()
    for i in range(0, 6):
        cur_crop_range = tl.bmp_crop_ranges[i]
        cur_crop_bmp = crop_bmp(cur_crop_range, i, imgs)
        # todo  平均灰度值 将A相机的crop作为灰度值基准，其他相机的crop平均到该基准
        if i == 0:
            target_gray = get_average_gray(cur_crop_bmp)
        else:
            cur_crop_bmp = average_png_gray(cur_crop_bmp, target_gray)
        # 将crop出的图放入png中
        put_crop_into_png(cur_crop_bmp, uv_map_png, i)
    # resize成1280*1600大小
    # print(time.time() - start)
    

    size = [640, 2240]
    png = resize_png(uv_map_png, size)  # 造成的形变是否会影响结果
    # 将png写入本地
    cv2.imwrite('1' + '.png', png)
    return png


# 计算crop出的图片宽度和高度
def calculate_crop_width_and_height():
    tl.crops_width_and_height[0] = [tl.bmp_crop_ranges[0][2] - tl.bmp_crop_ranges[0][0],
                                    tl.bmp_crop_ranges[0][3] - tl.bmp_crop_ranges[0][1]]
    tl.crops_width_and_height[1] = [tl.bmp_crop_ranges[1][2] - tl.bmp_crop_ranges[1][0],
                                    tl.bmp_crop_ranges[1][3] - tl.bmp_crop_ranges[1][1]]
    tl.crops_width_and_height[2] = [tl.bmp_crop_ranges[2][2] - tl.bmp_crop_ranges[2][0],
                                    tl.bmp_crop_ranges[2][3] - tl.bmp_crop_ranges[2][1]]
    tl.crops_width_and_height[3] = [tl.bmp_crop_ranges[3][2] - tl.bmp_crop_ranges[3][0],
                                    tl.bmp_crop_ranges[3][3] - tl.bmp_crop_ranges[3][1]]
    tl.crops_width_and_height[4] = [tl.bmp_crop_ranges[4][2] - tl.bmp_crop_ranges[4][0],
                                    tl.bmp_crop_ranges[4][3] - tl.bmp_crop_ranges[4][1]]
    tl.crops_width_and_height[5] = [tl.bmp_crop_ranges[5][2] - tl.bmp_crop_ranges[5][0],
                                    tl.bmp_crop_ranges[5][3] - tl.bmp_crop_ranges[5][1]]


# 提前计算出各个相机crop出的图在png中v所占的范围比重（0-1），例如A：0-0.25，B：0.25-0.4...F：0.8-1
def calculate_crop_v_scale_in_png():
    png_height = tl.uv_map_size[1]
    for i in range(0, 6):
        if i == 0:
            tl.crops_v_scale_in_png[i][1] = tl.crops_width_and_height[i][1] / png_height
        else:
            tl.crops_v_scale_in_png[i][0] = tl.crops_v_scale_in_png[i - 1][1]
            tl.crops_v_scale_in_png[i][1] = tl.crops_v_scale_in_png[i][0] + (
                    tl.crops_width_and_height[i][1] / png_height)


def crop_bmp(crop_range, camera_index, imgs):
    # 拼接bmp文件路径，拿到bmp，对其crop
    # path_str = file_path.split("/")
    # camera_name = tl.camera_index_to_name[camera_index]
    # pic_path_prefix = '/home/data/finger_vein/LFMB-3DFB_Pictures_Seged_Rectified_640_400/' + path_str[2]  # todo 注意这里的索引会随着文件路径改变而改变
    # pic_file_path = pic_path_prefix + '_' + camera_name + '.bmp'  # 拼接文件名
    # print(pic_file_path)
    # cur_img = cv2.imread(pic_file_path, cv2.IMREAD_GRAYSCALE)
    cur_img = imgs[camera_index]
    # 根据crop range进行crop
    crop_img = cur_img[int(crop_range[1]):int(crop_range[3]), int(crop_range[0]):int(crop_range[2])]
    # plt.imshow(crop_img, cmap="gray")
    # plt.show()
    # 将crop放入png
    # cv2.rectangle(cur_img, (crop_range[0], crop_range[1]), (crop_range[2], crop_range[3]), (255, 0, 0), 1)
    # cv2.imshow("rectangle", cur_img)
    return crop_img


# 计算图片平均灰度值，如果不计算灰度值为0的情况，则效率会比较低(注释的代码)，因此考虑效率直接做平均
def get_average_gray(png):
    # size = png.shape[0] * png.shape[1]
    # sum_gray = 0
    # for i in range(0, png.shape[0]):
    #     for j in range(0, png.shape[1]):
    #         if png[i][j] == 0:
    #             size -= 1
    #         else:
    #             sum_gray += png[i][j]
    # average_gray = sum_gray / size
    average_gray = np.mean(png)
    return average_gray


# 将当前图片灰度值平均到指定的灰度值
def average_png_gray(cur_crop_bmp, target_gray):
    cur_average_gray = get_average_gray(cur_crop_bmp)
    coefficient = float(target_gray / cur_average_gray)
    coefficient = round(coefficient, 2)
    cur_crop_bmp = cur_crop_bmp * coefficient
    return cur_crop_bmp


def put_crop_into_png(crop_pic, uv_map_png, camera_index):
    v_start = 0
    crop_height = crop_pic.shape[0]
    crop_wight = crop_pic.shape[1]
    i = 0
    while i < camera_index:
        v_start += tl.crops_width_and_height[i][1]  # 累积前面的高度
        i += 1
    uv_map_png[int(v_start):int(v_start + crop_height), 0:int(crop_wight)] = crop_pic
    # plt.imshow(uv_map_png, cmap="gray")
    # plt.show()


# 获得obj中所需要的信息
def get_png_uv_from_crops(faces_point, camera_index_to_faces):
    vt_list = []  # 每一行放在obj文件中f i/_ j/_ k/_
    vt_uv_val = []  # 存放uv具体信息  u,v:0->1
    i = 1  # vt_index 按照obj规定 ，从1开始
    # 用index方便定位三角面片调试
    for index in range(0, len(faces_point)):
        # if index >= 719:
        #     print("debug")
        face = faces_point[index]
        # camera_index = face[3]
        camera_index = camera_index_to_faces[index]
        vt_in_face = []
        for vertex in face[0:3]:
            key = str(camera_index) + "_" + str(vertex)
            # 先判断key是否存在于全局哈希表map_vertex_to_vt_index中，若存在，不用后续操作，直接取出
            if key not in tl.map_vertex_to_vt_index.keys():
                cur_texture = tl.map_vertex_to_texture[key]  # 从全局变量中取出
                cur_uv_in_png = get_uv_from_png(cur_texture, camera_index)
                vt_uv_val.append(cur_uv_in_png)
                vt_in_face.append(i)
                # 将key和值i放入全局哈希表map_vertex_to_vt_index
                tl.map_vertex_to_vt_index[key] = i  # 这个顶点对应的是第i个vt,然后在得到是第几个vt之后，就可以去vt_list取出
                i += 1
            else:
                vt_in_face.append(tl.map_vertex_to_vt_index[key])
        vt_list.append(vt_in_face)
    # tl.print_data_points(vt_uv_val)
    # tl.print_data_points(vt_list)
    vt_uv_val = np.array(vt_uv_val)
    vt_list = np.array(vt_list)
    return vt_uv_val, vt_list


# 根据像素信息获取png中对应的uv uv范围为0-1
def get_uv_from_png(cur_texture, camera_index):
    png_u = (cur_texture[0] - tl.bmp_crop_ranges[camera_index][0]) / tl.uv_map_size[0]  # 这里uv还需要考虑crop前后的坐标变化
    # cur_height, i = 0, 0
    # while i < camera_index:
    #     cur_height += tl.crops_width_and_height[i][1]  # 累积上面的高度
    #     i += 1
    # cur_height += (cur_texture[1] - tl.bmp_crop_ranges[camera_index][1])  # 再加上自身的高度
    # 下面的代码比上面注释掉的更快，避免每次都需要重复计算累积的高度
    cur_height = (cur_texture[1] - tl.bmp_crop_ranges[camera_index][1]) / tl.uv_map_size[1]
    png_v_error = tl.crops_v_scale_in_png[camera_index][0] + cur_height  # 注意这是错误的v，因为uv坐标原点在左下角而不是左上角！
    png_v = 1 - png_v_error  # 正确的v由于v坐标轴相反，所以用1-原来的值
    # if png_v < tl.crops_v_scale_in_png[camera_index][0] or png_v > tl.crops_v_scale_in_png[camera_index][1]:
    #     # 运行到这里说明出现了错误的范围
    #     print(png_v, camera_index)
    return [png_u, png_v]


def resize_png(png, size):
    resize_res = cv2.resize(png, (size[0], size[1]), interpolation=cv2.INTER_AREA)  # interpolation为插值算法
    return resize_res


def write_uv_to_obj(uv_val_in_obj, vt_list, file_path='/home/zf/vscode/3d/DR_3DFM/data/cylinder_template_mesh/cylinder1022'):
    lines = []
    mtl_info = 'mtllib saved_spot.mtl' + '\n'
    lines.append(mtl_info)
    with open(file_path + '.obj', 'r') as f:
        content = f.readlines()
        index = 0  # 记录位置
        # 先添加首部的顶点数据
        for index in range(0, len(content)):
            if index <= 5 and content[index] and content[index][0] != 'v':  # 避免开头可能出现的信息
                continue
            if content[index][0] == 'v':
                lines.append(content[index])
                continue
            else:
                break

        # 在中部放入计算出来的uv信息
        for uv_val in uv_val_in_obj:
            cur_str = 'vt' + " " + str(uv_val[0]) + " " + str(uv_val[1]) + '\n'
            lines.append(cur_str)
        mtl_info2 = 'usemtl material_1' + '\n'
        lines.append(mtl_info2)
        # 在底部更新三角面片数据
        i = index
        # 避免顶点数据和面片数据之间可能出现的一行或多行空格（5行之下）
        while i < index + 5 and content[i] and content[i][0] != 'f':
            i += 1
        for i, j in zip(range(i, len(content)), range(0, len(vt_list))):
            line = content[i]
            # if i <= i + 5 and line[0] != 'f':
            #     j -= 1
            #     continue
            vt_index = vt_list[j]
            face = line.split(" ")  # 先将字符串按空格切分成数组,取出末尾换行符，再进行拼接
            cur_str = 'f' + " " + face[1] + "/" + str(vt_index[0]) + \
                      " " + face[2] + "/" + str(vt_index[1]) + " " + \
                      face[3].replace('\n', '') + "/" + str(vt_index[2]) + '\n'
            lines.append(cur_str)

    with open(file_path + '_new.obj', 'w+') as f_new:
        f_new.writelines(lines)

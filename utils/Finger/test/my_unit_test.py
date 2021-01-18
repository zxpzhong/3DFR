import unittest
from process import process_finger_data as pfd
import matplotlib.pyplot as plt
from tool import tools as tl
from tool import read_24bit_bmp as rbm
import cv2
import time
import numpy as np
from process import faces_texture_mapping as ftm


class Test(unittest.TestCase):

    # 显示相机平面位置
    def test_show_camera_plane(self):
        camera_origins = [[2.50436065, -3.75589484, 1.88800446],
                          [4.02581981, -2.56894275, -3.29281609],
                          [1.01348544, 1.88043939, -5.4273143],
                          [-2.45261002, 3.5962286, -1.87506165],
                          [-3.12155638, 2.09254542, 2.21770186],
                          [-1.07692383, -1.37631717, 4.3081322],
                          # [-1.14422972e-01, - 9.73947995e-02, 2.05371429e-01],
                          # [-3.02246150e-01, 6.58760412e-02, 2.73782622e-01],
                          # [-4.98036103e-01, 2.93154709e-01, 2.44336074e-01],
                          # [3.18449475e-01, - 1.73387355e-01, - 4.82361449e-01],
                          # [1.11402481e+00, - 7.18796123e-01, - 1.03014576e+00],
                          # [1.49183233e-01, - 4.59760317e-01, 3.44508327e-01]
                          # [0.3165423775809235, -0.10786089526290894, -0.5944049978975746],
                          # [-1.261181, -2.678640, -2.523260]
                          ]
        camera_origins = np.array(camera_origins)
        x = camera_origins[:, 0]
        y = camera_origins[:, 1]
        z = camera_origins[:, 2]

        ax = plt.subplot(111, projection='3d')
        ax.scatter(x[:], y[:], z[:], c='r')
        ax.set_zlabel('Z')  # 坐标轴
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        plt.show()

    # 测试计算向量
    def test_calculate_vector(self):
        point1 = [1, 2, 3]
        point2 = [3, 2, 1]
        vector = pfd.calculate_vector(point1, point2)
        self.assertEqual(vector, [2, 0, -2])

    # 测试计算向量积函数
    def test_calculate_vector_product(self):
        vector1 = [1, 2, 3]
        vector2 = [3, 2, 1]
        vector_product = tl.calculate_vector_product(vector1, vector2)
        self.assertEqual(vector_product, [-4, 8, -4])

    # 测试计算向量夹角余弦值函数
    def test_calculate_cosine(self):
        vector1 = [1, 0, 0]
        vector2 = [0, 0, 1]
        res = tl.calculate_cosine(vector1, vector2)
        print(tl.calculate_cosine([1, 2, 3], [3, 2, 1]))
        self.assertEqual(res, 0)

    def test_get_single_point_from_which_camera(self):
        center_point_ = [0.3165423775809235, -0.10786089526290894, -0.5944049978975746]
        cur_point_ = [-4.98036103e-01, 2.93154709e-01, 2.44336074e-01]
        camera_origins = [[2.50436065, -3.75589484, 1.88800446],
                          [4.02581981, -2.56894275, -3.29281609],
                          [1.01348544, 1.88043939, -5.4273143],
                          [-2.45261002, 3.5962286, -1.87506165],
                          [-3.12155638, 2.09254542, 2.21770186],
                          [-1.07692383, -1.37631717, 4.3081322]]
        index = pfd.get_single_point_from_which_camera(center_point_, cur_point_, camera_origins)
        print("目标相机索引是：", index)

    # 测试判断点来自哪两个相机之间的函数
    def test_get_point_from_which_camera2(self):
        center_point = [-0.58192716, - 0.94316095, - 1.06762088]
        center_point_ = [0.3165423775809235, -0.10786089526290894, -0.5944049978975746]
        cur_point = [-1.261181, -2.678640, -2.523260]
        camera_origins = [[2.50436065, -3.75589484, 1.88800446],
                          [4.02581981, -2.56894275, -3.29281609],
                          [1.01348544, 1.88043939, -5.4273143],
                          [-2.45261002, 3.5962286, -1.87506165],
                          [-3.12155638, 2.09254542, 2.21770186],
                          [-1.07692383, -1.37631717, 4.3081322]]
        count = pfd.get_point_from_which_camera2(cur_point, center_point_, camera_origins)
        self.assertEqual(count, 1)

    # 计算六个相机的投影矩阵
    def test_calculate_camera_projection_mat(self):
        camera_a_projection_mat = tl.camera_a_inner_para * tl.camera_a_outer_para
        print(camera_a_projection_mat, "\n")
        camera_b_projection_mat = tl.camera_b_inner_para * tl.camera_b_outer_para
        print(camera_b_projection_mat, "\n")
        camera_c_projection_mat = tl.camera_c_inner_para * tl.camera_c_outer_para
        print(camera_c_projection_mat, "\n")
        camera_d_projection_mat = tl.camera_d_inner_para * tl.camera_d_outer_para
        print(camera_d_projection_mat, "\n")
        camera_e_projection_mat = tl.camera_e_inner_para * tl.camera_e_outer_para
        print(camera_e_projection_mat, "\n")
        camera_f_projection_mat = tl.camera_f_inner_para * tl.camera_f_outer_para
        print(camera_f_projection_mat)

    # 计算resize后的变化的投影矩阵
    def test_calculate_camera_projection_mat_after_resize(self):
        camera_a_projection_mat_resize = tl.camera_a_inner_para_640_400 * tl.camera_a_outer_para
        print(camera_a_projection_mat_resize, "\n")
        camera_b_projection_mat_resize = tl.camera_b_inner_para_640_400 * tl.camera_b_outer_para
        print(camera_b_projection_mat_resize, "\n")
        camera_c_projection_mat_resize = tl.camera_c_inner_para_640_400 * tl.camera_c_outer_para
        print(camera_c_projection_mat_resize, "\n")
        camera_d_projection_mat_resize = tl.camera_d_inner_para_640_400 * tl.camera_d_outer_para
        print(camera_d_projection_mat_resize, "\n")
        camera_e_projection_mat_resize = tl.camera_e_inner_para_640_400 * tl.camera_e_outer_para
        print(camera_e_projection_mat_resize, "\n")
        camera_f_projection_mat_resize = tl.camera_f_inner_para_640_400 * tl.camera_f_outer_para
        print(camera_f_projection_mat_resize)

    # 测试读取uv数据
    def test_read_uv_points(self):
        file_path = 'outer_files/LFMB_Visual_Hull_Meshes256/001_1_2_01'
        suffix = ".txt"
        uv_points = pfd.read_uv_points(file_path + suffix)
        tl.print_data_points(uv_points)

    # 测试读取faces数据
    def test_read_mesh_faces(self):
        file_path = '../outer_files/LFMB_Visual_Hull_Meshes256/002_1_2_01'
        suffix = ".obj"
        faces_point = pfd.read_mesh_faces(file_path + suffix)
        tl.print_data_points(faces_point)

    # 测试用手写的函数读取位图
    def test_read_bmp(self):
        start = time.time()
        file_path = '../outer_files/images/00_1_2_01_A.bmp'
        img = rbm.read_rows(file_path)
        print(time.time() - start)
        plt.imshow(img, cmap="gray")
        plt.show()

    # 测试用cv内置库去读取位图
    def test_read_bmp2(self):
        start = time.time()
        file_path = '../outer_files/images/003_1_2_01_A.bmp'
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        print(time.time() - start)
        plt.imshow(img, cmap="gray")
        plt.show()

    def test_write_gray_to_bmp(self):
        # 此时不需要关闭文件，a+ 可读可写（末尾追加再写），文件不存在就创建，r+可读可写不存在报错
        fp = open("../outer_files/LFMB_Visual_Hull_Meshes256/001_1_2_01.txt", "a+", encoding="utf-8")
        line = fp.readline()
        fp.write("hello python1")  # \n用来换行
        fp.seek(0, 0)
        data = fp.read()
        fp.close()
        print(data)

    # 测试三维mesh与uv map（png）的纹理映射关系
    def test_uv_map_relation_mesh(self):
        uv_map_file = '../outer_files/Mesh_UVmap/saved_spot.png'
        img = cv2.imread(uv_map_file)
        pass

    # 测试uvmap空白png
    def test_gene_uvmap_png(self):
        uv_map_png = np.zeros((1280, 800), dtype=np.uint8)
        crop_png = np.ones((300, 100), dtype=np.uint8)
        uv_map_png[100:500, 200:400] = crop_png[0:, 0:]
        plt.imshow(uv_map_png, cmap="gray")
        plt.show()
        # cv2.imshow('0', uv_map_png)

    # 测试resize图像
    def test_resize_bmp(self):
        file_path = '../outer_files/images/001_1_2_01_F.bmp'
        img = cv2.imread(file_path)
        height, width = img.shape[:2]
        resize_img = cv2.resize(img, (int(0.5 * width), int(0.5 * height)), interpolation=cv2.INTER_AREA)
        cv2.imwrite('../outer_files/images/001_resize_F.bmp', resize_img)
        # plt.imshow(img, cmap="gray")
        plt.imshow(resize_img, cmap="gray")
        plt.show()

    def test_crop_bmp(self):
        path = 'C:/Users/10327/Desktop/testmesh/test2/test3/saved_spot.png'
        img = cv2.imread(path)
        crop_img = img[961:1362, :]
        plt.imshow(crop_img, cmap="gray")
        plt.show()

    # 测试点到空间平面的投影
    def test_get_mapping_point_in_camera_plane(self):
        camera_plane_para = [1, 2, -1, 1]
        point = [-1, 2, 0]
        mapping_point = tl.get_mapping_point_in_camera_plane(point, camera_plane_para)
        print(mapping_point)

    # 测试cv2.imread()是否会造成内存过大的问题
    def test_cv2_imread_memory(self):
        for i in range(0, 4848):
            file_path = '../outer_files/images/001_1_2_01_F.bmp'
            img = cv2.imread(file_path)

    # 异常处理测试
    def test_read_file_exception(self):
        pfd.read_mesh_points('../outer_files/images/sss')

    def test_show_single_face_crop(self):
        pass

    # 在png上显示三角面片区域，看是否符合预期
    def test_show_single_face_area_in_png(self):
        file_path = '../outer_files/LFMB_Visual_Hull_Meshes256/001_1_2_01.png'
        png = cv2.imread(file_path)
        face_vertex_in_png = [[56, 953], [115, 1023], [134, 976]]
        u_min = min(face_vertex_in_png[0][0], face_vertex_in_png[1][0], face_vertex_in_png[2][0])
        v_min = min(face_vertex_in_png[0][1], face_vertex_in_png[1][1], face_vertex_in_png[2][1])
        u_max = max(face_vertex_in_png[0][0], face_vertex_in_png[1][0], face_vertex_in_png[2][0])
        v_max = max(face_vertex_in_png[0][1], face_vertex_in_png[1][1], face_vertex_in_png[2][1])
        crop_range = [u_min, v_min, u_max, v_max]
        crop_img = png[crop_range[1]:crop_range[3], crop_range[0]:crop_range[2]]
        plt.imshow(crop_img, cmap="gray")
        plt.show()

    # 根据uv信息，在png中以白色点的形式显示，查看uv信息是否正确
    def test_show_uv_in_png(self):
        # 读取uv信息
        uv_file = '../outer_files/LFMB_Visual_Hull_Meshes256/001_1_2_01_new.obj'
        uv_list = []
        with open(uv_file) as f:
            for line in f:
                str = line.split(" ")
                if str[0] == 'vt':
                    cur_uv = [str[1], str[2].replace('\n', '')]
                    uv_list.append(cur_uv)
                else:
                    continue

        # 读取png
        png_file = '../outer_files/LFMB_Visual_Hull_Meshes256/001_1_2_01.png'
        png = cv2.imread(png_file)
        v_height = png.shape[0]
        u_width = png.shape[1]
        for uv in uv_list:
            uv = [float(uv[0]), float(uv[1])]
            u = uv[0] * u_width
            v = uv[1] * v_height
            u = int(u)
            v = int(v)
            if u <= 2:
                u = 2
            if v <= 2:
                v = 2
            if u >= u_width - 3:
                u = u_width - 3
            if v >= v_height - 3:
                v = v_height - 3
            # 将这个坐标置为白色
            # png[v - 1][u - 1] = [255, 0, 0]
            cv2.rectangle(png, (u - 2, v - 2), (u + 2, v + 2), (255, 0, 0), 1)
        cv2.imwrite('../outer_files/LFMB_Visual_Hull_Meshes256/uv_in_png.png', png)

    def test_get_average_gray(self):
        png = np.array([[1, 2], [3, 4]])
        gray = ftm.get_average_gray(png)
        self.assertEqual(gray, 2.5)
        png2 = np.array([[0, 1, 2], [3, 4, 5]])
        gray2 = ftm.get_average_gray(png2)
        self.assertEqual(gray2, 3)

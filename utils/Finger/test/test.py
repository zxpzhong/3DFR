import numpy as np
from process import process_finger_data as pfd
import matplotlib.pyplot as plt

a = np.mat([[0.574322111, 0.771054881, 0.275006333, 0.93847817],
            [0.565423192, -0.130698104, -0.814379899, -0.36935905],
            [-0.591988790, 0.623211341, -0.511035123, 4.78810628],
            [0, 0, 0, 1]])
a2 = a.I
print(a2)

origin = pfd.get_single_camera_origin(a)
# print(origin)

camera_origins = pfd.get_cameras_coordinate()
print(camera_origins)
camera_origins = np.array(camera_origins)


# 验证摄像机在世界坐标系下的三维位置
def show_camera_origins_in_3d(camera_origins):
    x = camera_origins[:, 0]
    y = camera_origins[:, 1]
    z = camera_origins[:, 2]

    ax = plt.subplot(111, projection='3d')
    ax.scatter(x[:], y[:], z[:], c='r')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


show_camera_origins_in_3d(camera_origins)

# 验证平面函数正确性
a = np.array([[0.52, 0.7, 0.2], [0, 2, 0], [2, 0, 0], [0, 2, 2], [0.5, -0.7, 0.1], [0, 0, 1]])
plane = pfd.get_camera_plane(a)
print(plane)
# 求出相机平面方程
plane2 = pfd.get_camera_plane(camera_origins)
# print(plane2)

# 验证映射函数的正确性
point = [3, 7, 5]
plane_para = [1, 2, 1, -16]
point_ = pfd.get_mapping_point_in_camera_plane(point, plane_para)
print(point_)


# 验证读取图片

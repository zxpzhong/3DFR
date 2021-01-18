# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/13 14:27
@Auth ： 零分
@File ：main_base_vertex.py
@IDE ：PyCharm
@github:https://github.com/huoxubugai/3DFinger
"""

from process import process_finger_data as pfd, points_texture_mapping as tm
import numpy as np
from tool import tools as tl
import os
import time

'通过mesh顶点进行纹理映射'
if __name__ == '__main__':
    start = time.time()
    # todo 遍历文件夹下的所有mesh 操作每一个
    file_path = 'outer_files/LFMB_Visual_Hull_Meshes256/0001_2_01'
    obj_suffix = '.obj'
    uv_file_path = file_path + '.txt'
    # todo 文件读取异常处理
    # 拿到mesh所有顶点数据
    data_points, _ = pfd.read_mesh_points(file_path + obj_suffix)  # 数据点的数据结构选择list而不是数组,方便后续改动
    # 求出所有顶点对应的中心点O
    center_point = pfd.get_center_point(data_points)
    # 获取相机平面的参数ax+by+cz+d=0
    camera_plane_para = tl.camera_plane_para
    # 获取中心点O的映射点
    center_point_mapping = tl.get_mapping_point_in_camera_plane(center_point, camera_plane_para)
    # 将mesh顶点数据中的所有顶点映射到相机平面
    data_points_mapping = pfd.get_data_points_mapping(data_points, camera_plane_para)
    # 数据预处理完毕，寻找每个点对应的相机
    # 这里注意找到相机之后需要添加到源数据点上，而不是映射后的数据点
    camera_index_and_uv = pfd.get_data_points_from_which_camera2(center_point_mapping, data_points_mapping,
                                                                 tl.cameras_coordinate_mapping, data_points)
    # 得到每个点是由什么相机拍摄之后，进行纹理映射部分
    # 得到每个点对应二维图像上的u，v值
    camera_index_and_uv = tm.get_uv_for_points(data_points, camera_index_and_uv)
    points_gray = tm.mapping_points_gray(data_points, camera_index_and_uv, file_path)
    # 拿到灰度值list 写入到obj文件中
    tm.write_gray_to_obj(points_gray, file_path)
    print(time.time() - start)

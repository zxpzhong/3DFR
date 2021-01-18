import matplotlib.pyplot as plt


# 由于bmp图片的特殊性，需要写一个函数来读取该种类型文件
# 用16进制查看器发现手指的位图为24位的位图，即每一个像素由24位也就是3个字节表示，分别是r,g,b
# bmp头文件包含的信息有：文件总字节数，图片宽度和高度，每个像素占有位数等
# 第二天补充：CV2里有对应的函数，其实不用这么麻烦写
def read_rows(path):
    image_file = open(path, "rb")
    # 跳过bmp的文件头（含54个字节）
    image_file.seek(54)
    rows = []
    row = []
    pixel_index = 0
    count = 0  # 计算有多少个不为0的像素值

    while True:
        if pixel_index == 1280:
            pixel_index = 0
            rows.insert(0, row)
            if len(row) != 1280 * 1:
                raise Exception("row与预期尺寸不一致")
            row = []
        pixel_index += 1

        r_string = image_file.read(1)
        g_string = image_file.read(1)
        b_string = image_file.read(1)

        if len(r_string) == 0:
            if len(rows) != 800:
                print("读取到的像素与预期不一致")
            break

        if len(g_string) == 0:
            print("读取到0长度的green，跳出")
            break

        if len(b_string) == 0:
            print("读取到0长度的blue，跳出")
            break
        r = ord(r_string)
        g = ord(g_string)
        b = ord(b_string)
        # if r != 0 or g != 0 or b != 0:
        #     count += 1
        #     print(count)
        gray = (r * 299 + g * 587 + b * 114) / 1000  # 将r,g，b转换为灰度值
        gray = round(gray)
        # row.append(b)
        # row.append(g)
        # row.append(r)
        row.append(gray)

    image_file.close()

    return rows


#
# def repack_sub_pixels(rows):
#     # print "Repacking pixels..."
#     sub_pixels = []
#     for row in rows:
#         for sub_pixel in row:
#             sub_pixels.append(sub_pixel)
#
#     diff = len(sub_pixels) - 1280 * 800 * 1
#     # print "Packed", len(sub_pixels), "sub-pixels."
#     if diff != 0:
#         print("Error! Number of sub-pixels packed does not match 1280*800: (" + str(
#             len(sub_pixels)) + " - 1280 * 800 * 3 = " + str(diff) + ").")
#
#     return sub_pixels


rows = read_rows("../outer_files/images/001_1_2_01_A.bmp")
# plt.imshow(rows, cmap="gray")
# plt.show()

# sub_pixels = repack_sub_pixels(rows)

import cv2, os
import numpy as np


def draw_matches(main_image, sub_image, kp1, kp2, good_matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算变换矩阵并应用到小拼图上
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = sub_image.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # 整体添加灰色蒙版
    main_image = cv2.addWeighted(main_image, 0.7, np.zeros(main_image.shape, main_image.dtype), 0.5, 0)

    # 在大拼图上绘制小拼图的边框
    main_image = cv2.polylines(main_image, [np.int32(dst)], True, (0, 255, 0), 5, cv2.LINE_AA)

    # 绘制所有匹配点
    main_image = cv2.drawMatches(sub_image, kp1, main_image, kp2, good_matches, None, flags=2)

    # show sub image with fix width
    max_height = 800
    scale = max_height / main_image.shape[0]
    new_width = int(main_image.shape[1] * scale)
    main_image = cv2.resize(main_image, (new_width, max_height))

    return main_image


def find_subimage_core(main_image, sub_image_path, accuracy=0.78):
    # 读取大拼图和小拼图
    sub_image = read_image(sub_image_path)
    if sub_image is None:
        print("Failed to load sub_image2")

    # 使用AKAZE检测特征点和计算描述符
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(sub_image, None)
    kp2, des2 = akaze.detectAndCompute(main_image, None)

    # 使用FLANN匹配器进行特征点匹配
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 根据Lowe's ratio测试获取良好的匹配点
    good_matches = []
    for mn in matches:
        if len(mn) != 2:
            continue
        m, n = mn
        if m.distance < accuracy * n.distance:  # 这里是用来调整的参数
            good_matches.append(m)

    if len(good_matches) <= 10:
        print("Not enough matches found.")
        return False
    return [sub_image, kp1, kp2, good_matches]


def read_image(path):
    if not os.path.isfile(path):
        raise Exception(f"File {path} not found.")
    return cv2.imread(path)

def find_subimage_akaze(main_image_path, sub_image_path, accuracy=0.78):
    main_image = read_image(main_image_path)
    # 查找单个小拼图
    # sub_image_paths = [sub_image_path, "cleaned_img/corner.JPG"]
    # datas = []
    # for sub_image in sub_image_paths:
    #     result = find_subimage_core(main_image, sub_image, accuracy)
    #     if result is False:
    #         continue  # TODO: 站空位
    #     datas.append(result)

    result = find_subimage_core(main_image, sub_image_path, accuracy)
    if result is False:
        return False

    # 绘制匹配结果
    # image = draw_matches_multi(main_image, datas)
    image = draw_matches(main_image, result[0], result[1], result[2], result[3])

    # 保存结果
    get_file_name = lambda path: path.split('/')[-1].split('.')[0] + '_processed.JPG'
    # cv2.imwrite('temp/' + get_file_name(sub_image_path), image)
    cv2.imwrite('temp/' + "output.jpg", image)

    # resize
    max_height = 800
    scale = max_height / image.shape[0]
    new_width = int(image.shape[1] * scale)
    image = cv2.resize(image, (new_width, max_height))

    # 显示结果
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True


if __name__ == '__main__':
    # get sys args
    import sys
    if len(sys.argv) < 2:
        print("Please input image path")
        exit(1)
    main_image_path = sys.argv[1]
    sub_image_path = sys.argv[2]
    find_subimage_akaze(main_image_path, sub_image_path)

    # TODO make it  return Green frame, start-end lines.
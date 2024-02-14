import cv2, os
import numpy as np


def draw_debug_matches(main_image, sub_image, kp1, kp2, good_matches):
    def draw_point_text(image, x, y):
        color = tuple([int(x) for x in np.random.randint(0, 255, 3)])
        # 随机在x,y坐标上加减10，防止文字重叠
        new_x = x + np.random.randint(-200, 200)
        new_y = y + np.random.randint(-200, 200)
        # 文字增加边框
        cv2.putText(image, f"{x:.1f},{y:.1f}", (int(new_x), int(new_y)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (128, 128, 128), 3)
        cv2.putText(image, f"{x:.1f},{y:.1f}", (int(new_x), int(new_y)), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        cv2.circle(image, (int(x), int(y)), 10, color, 2)

    # 在每一个匹配点上 注明坐标文字, 并且用绿色的圆圈标记，方便观察
    for m in good_matches:
        x, y = kp2[m.trainIdx].pt
        draw_point_text(main_image, x, y)

        x, y = kp1[m.queryIdx].pt
        draw_point_text(sub_image, x, y)

    return main_image, sub_image

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

    # debugger mode
    # main_image, sub_image = draw_debug_matches(main_image, sub_image, kp1, kp2, good_matches)

    # 绘制所有匹配点
    main_image = cv2.drawMatches(sub_image, kp1, main_image, kp2, good_matches, None, flags=2)

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

def find_subimage_akaze(main_image_path, sub_image_path, save_folder, accuracy=0.78):
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


    # 显示结果 resize
    # max_height = 800
    # scale = max_height / image.shape[0]
    # new_width = int(image.shape[1] * scale)
    # image = cv2.resize(image, (new_width, max_height)) 
    # cv2.imshow('Result', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    get_file_name = lambda path: path.split('/')[-1].split('.')[0] + '_processed.JPG'
    save_path =  os.path.join(save_folder, get_file_name(sub_image_path))
    # 保存结果
    # cv2.imwrite('temp/' + get_file_name(sub_image_path), image)
    cv2.imwrite(save_path, image)
    return save_path


if __name__ == '__main__':
    # get sys args
    import sys
    if len(sys.argv) < 2:
        print("Please input image path")
        exit(1)
    main_image_path = sys.argv[1]
    sub_image_path = sys.argv[2]
    save_folder = sys.argv[3]
    accuracy = float(sys.argv[4])

    result_path = find_subimage_akaze(main_image_path, sub_image_path, save_folder, accuracy)
    print(result_path)

    # print("Done")
    # TODO make it  return Green frame, start-end lines.
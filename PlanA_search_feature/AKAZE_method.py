import cv2, os
import numpy as np

def find_subimage_akaze(main_image_path, sub_image_path, accuracy=0.78):
    if not os.path.isfile(main_image_path):
        raise FileNotFoundError(f"No such file: '{main_image_path}'")
    if not os.path.isfile(sub_image_path):
        raise FileNotFoundError(f"No such file: '{sub_image_path}'")

    # 读取大拼图和小拼图
    main_image = cv2.imread(main_image_path)
    sub_image = cv2.imread(sub_image_path)

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
        m,n = mn
        if m.distance < accuracy * n.distance:  # 这里是用来调整的参数
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算变换矩阵并应用到小拼图上
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = sub_image.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # 整体添加灰色蒙版
        main_image = cv2.addWeighted(main_image, 0.5, np.zeros(main_image.shape, main_image.dtype), 0.3, 0)

        # 在大拼图上绘制小拼图的边框
        main_image = cv2.polylines(main_image, [np.int32(dst)], True, (0, 255, 0), 5, cv2.LINE_AA)

        # 绘制所有匹配点
        main_image = cv2.drawMatches(sub_image, kp1, main_image, kp2, good_matches, None, flags=2)

        # 显示结果
        max_height = 1000
# show image with fix width
        scale = max_height / main_image.shape[0]
        new_width = int(main_image.shape[1] * scale)
        main_image = cv2.resize(main_image, (new_width, max_height))

        cv2.imshow('Result', main_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Not enough matches found.")

# 示例用法
# failed
# find_subimage_akaze('testCases/case2_workprofile/main_image.jpg',
#                    'testCases/case2_workprofile/sub_image2.jpg')
# find_subimage_akaze("puzzleCases/raw/corner.JPG", "puzzleCases/raw/corner_part_noise.JPG")

# find_subimage_akaze("testCases/floor/medium_angle.JPG", "testCases/floor/up.JPG")
# find_subimage_akaze("testCases/floor/medium_angle.JPG", "testCases/floor/slight_angle.JPG")
# find_subimage_akaze("puzzleCases/fully/full_puzzle3_corner_part.jpg", "puzzleCases/raw/corner_part.JPG")  # _noise
        


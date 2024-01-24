import cv2
import numpy as np

def find_subimage_surf(main_image_path, sub_image_path):
    # 读取大拼图和小拼图
    main_image = cv2.imread(main_image_path)
    sub_image = cv2.imread(sub_image_path)

    # Check if the images are loaded correctly
    if main_image is None:
        print(f"Failed to load main image from {main_image_path}")
        return
    if sub_image is None:
        print(f"Failed to load sub image from {sub_image_path}")
        return

    # 转换为灰度图
    gray_main = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    gray_sub = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=10000)

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray_sub, None)
    kp2, des2 = orb.detectAndCompute(gray_main, None)

    # Use BFMatcher instead of FLANN for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (the lower the better)
    matches = sorted(matches, key=lambda x: x.distance)

    # 根据Lowe's ratio测试获取良好的匹配点
    good_matches = []
    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        distance = match.distance
        # if m.distance < 0.7 * n.distance:
        #     good_matches.append(m)
        if distance < 0.99:
            good_matches.append(match)

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算变换矩阵并应用到小拼图上
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = gray_sub.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # 在大拼图上绘制小拼图的边框
        main_image = cv2.polylines(main_image, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # 显示结果
        cv2.imshow('Result', main_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Not enough matches found.")

# 示例用法
# find_subimage_surf('testCases/case2_workprofile/main_image.jpg',
#                    'testCases/case2_workprofile/sub_image2.jpg')

find_subimage_surf("puzzleCases/full_puzzle.jpeg", "puzzleCases/full_puzzle_mojipart2.jpeg")

# find_subimage_surf("puzzleCases/full_puzzle.jpeg", "puzzleCases/raw1_rotate.jpg")
# find_subimage_surf("puzzleCases/full_puzzle.jpeg", "puzzleCases/full_puzzle_towerpart.jpg")
# find_subimage_surf("puzzleCases/full_puzzle.jpeg", "puzzleCases/full_puzzle_mojipart.jpeg")

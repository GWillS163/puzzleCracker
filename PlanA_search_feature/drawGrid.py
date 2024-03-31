import os
import numpy as np
# from PIL import Image
# import cv2 from opencv-python
import cv2


# q: 如何安装cv2
# A: pip install opencv-python

def is_point_in_parallelogram(point, parallelogram):
    """判断点是否在平行四边形内"""

    def cross_product(p1, p2, p3):
        """计算向量p1p2和向量p1p3的叉积"""
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    p1, p2, p3, p4 = parallelogram
    return (cross_product(p1, p2, point) >= 0 and
            cross_product(p2, p3, point) >= 0 and
            cross_product(p3, p4, point) >= 0 and
            cross_product(p4, p1, point) >= 0)


def calculate_intersection(line1, line2):
    """计算两条线的交点"""
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def calculate_grid_position(grid_row_lines, grid_col_lines, targetPart):
    """根据网格线，推算在哪个网格内"""
    for r, row_lines in enumerate(grid_row_lines):
        for c, col_lines in enumerate(grid_col_lines):
            if r == 0 or c == 0:
                continue
            # 计算网格的四个交点, 从左上角开始顺时针 !important
            p1 = calculate_intersection(grid_row_lines[r - 1],
                                        grid_col_lines[c - 1])
            p2 = calculate_intersection(grid_row_lines[r - 1],
                                        grid_col_lines[c])
            p3 = calculate_intersection(grid_row_lines[r],
                                        grid_col_lines[c])
            p4 = calculate_intersection(grid_row_lines[r],
                                        grid_col_lines[c - 1])
            if is_point_in_parallelogram(targetPart, [p1, p2, p3, p4]):
                return r, c
    return -1, -1


def getCenter(start, end):
    return (int((start[0] + end[0]) / 2), int((start[1] + end[1]) / 2))


def getGridLinesPos(image, gridData):
    up_left, up_right, down_left, down_right, rows, cols = gridData["up_left"], gridData["up_right"], gridData[
        "down_left"], gridData["down_right"], gridData["rows"], gridData["cols"]
    row_px = 1 / rows
    col_px = 1 / cols
    row_lines = [i * row_px for i in range(rows + 1)]
    col_lines = [i * col_px for i in range(cols + 1)]

    # 计算每行的起点终点，在图像上绘制行
    row_lines_pos = []
    for index, ratio in enumerate(row_lines):
        start_point = [int(up_left[0] + ratio * (down_left[0] - up_left[0])),  # 起始点 + 比例 * 起始点到终点的距离
                       int(up_left[1] + ratio * (down_left[1] - up_left[1]))]
        end_point = [int(up_right[0] + ratio * (down_right[0] - up_right[0])),
                     int(up_right[1] + ratio * (down_right[1] - up_right[1]))]

        row_lines_pos.append([start_point, end_point])

    # 计算每列的起点终点，在图像上绘制列
    col_line_pos = []
    for c in col_lines:
        start_point = [int(up_left[0] + c * (up_right[0] - up_left[0])),
                       int(up_left[1] + c * (up_right[1] - up_left[1]))]
        end_point = [int(down_left[0] + c * (down_right[0] - down_left[0])),
                     int(down_left[1] + c * (down_right[1] - down_left[1]))]
        col_line_pos.append([start_point, end_point])

    return row_lines_pos, col_line_pos


def draw_grid(image, row_lines, col_lines):
    # 放置行标签
    for index, _ in enumerate(row_lines):
        start_point, end_point = row_lines[index]
        cv2.line(image, tuple(start_point), tuple(end_point), (0, 255, 0), 4)
        if index == 0:
            continue
        label_pos = getCenter(row_lines[index - 1][0], row_lines[index][0])
        cv2.putText(image, str(index), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10)
        cv2.putText(image, str(index), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 122, 255), 8)

    # 放置列标签
    for index, _ in enumerate(col_lines):
        start_point, end_point = col_lines[index]
        cv2.line(image, tuple(start_point), tuple(end_point), (0, 255, 0), 4)
        if index == 0:
            continue
        label_pos = getCenter(col_lines[index - 1][0], col_lines[index][0])
        cv2.putText(image, str(index), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10)
        cv2.putText(image, str(index), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 122, 255), 8)

    return image


def calculateTrueGrids(grid_row_lines, grid_col_lines, parallelogram, targetPart_point):
    if is_point_in_parallelogram(targetPart_point, parallelogram):
        return calculate_grid_position(grid_row_lines, grid_col_lines, targetPart_point)
    return None


def showGird(girds, target_point):
    # show
    image_obj = cv2.imread(main_image)

    grids_lines = []
    for girdData in girds:
        # 获取网格线的 两端位置
        row_lines, col_lines = getGridLinesPos(image_obj, girdData)
        # 计算目标点在哪个网格内
        targetRes = calculateTrueGrids(row_lines, col_lines,
                                       [girdData["up_left"], girdData["up_right"], girdData["down_right"],
                                        girdData["down_left"]],
                                       target_point)  # , girdData["rows"], girdData["cols"])
        if targetRes is None:
            print("targetPoint is not in the parallelogram")
            continue
        # 画网格
        image_obj = draw_grid(image_obj, row_lines, col_lines)
        print("targetGrid:", targetRes)
        break

    # exit(0) # debug

    # resize to fit screen
    # cv2.imwrite("temp.jpg", image_obj)
    max_height = 800
    scale = max_height / image_obj.shape[0]
    new_width = int(image_obj.shape[1] * scale)
    image_obj = cv2.resize(image_obj, (new_width, max_height))
    cv2.imshow("image", image_obj)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Test Area
    main_image = r"D:\Project\puzzleCracker\puzzleCases\fully\full_combine.jpg"
    target_part_point = [560, 560]
    # target_part_point = [3775, 923]

    grids = [
        # {
        #     "up_left": [93, 116],
        #     "up_right": [3011, 116],
        #     "down_left": [138, 3910],
        #     "down_right": [3011, 3910],
        #     "rows": 25,
        #     "cols": 20
        # },
        {
            "up_left": [3036, 71],
            "up_right": [5932, 71],
            "down_left": [3036, 3980],
            "down_right": [5932, 3960],
            "rows": 25,
            "cols": 20
        },
        {
            "up_left": [400, 400],
            "up_right": [1500, 600],
            "down_left": [800, 1800],
            "down_right": [2000, 2000],
            "rows": 3,
            "cols": 4
        },
    ]

    showGird(grids, target_part_point)
    print("Done")

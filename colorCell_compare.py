import cv2, os
import numpy as np

def masaico_full(image_path=r'puzzleCases\fully\full_puzzle3.jpg'):
    # read original imager
    original_image = cv2.imread(image_path)

    # slice it into 3x3 pieces by grid
    rows, cols = 25, 40
    row_offset, col_offset = 0, 0
    height, width, _ = original_image.shape
    piece_height = height // rows
    piece_width = width // cols
    for i in range(rows + 1):
        height_inline = i * piece_height + row_offset
        cv2.line(original_image, (0, height_inline), (width, height_inline), (0, 255, 0), 1)
    for j in range(cols):
        width_inline = j * piece_width + col_offset
        cv2.line(original_image, (width_inline, 0), (width_inline, height), (0, 255, 0), 1)
    # show it in a window with 1000px width
    window_width = 1000
    scale = window_width / original_image.shape[1]
    window_height = int(original_image.shape[0] * scale)
    original_image = cv2.resize(original_image, (window_width, window_height))
    # cv2.imshow('Original Image', original_image)
    # cv2.waitKey(0)

    # 将每个格子混合成一个色块
    # read all pieces
    pieces = []
    for i in range(rows):
        row = []
        for j in range(cols):
            # 指定每个小块的位置
            x_start = j * piece_width + col_offset
            x_end = (j + 1) * piece_width + col_offset
            y_start = i * piece_height + row_offset
            y_end = (i + 1) * piece_height + row_offset

            piece = original_image[y_start:y_end, x_start:x_end]
            row.append(piece)
        pieces.append(row)

    # 混合每个格子所有像素的颜色
    mixed_pieces = []
    color_grid = [ [] for _ in range(rows) ]
    for i, row in enumerate(pieces):
        row_pieces = []
        for piece in row:
            # 计算每个像素的平均颜色
            average_color_per_row = np.average(piece, axis=0)
            average_color = np.average(average_color_per_row, axis=0)
            average_color = np.uint8(average_color)

            # 生成彩色正方形
            color_grid[i].append(list(average_color))
            mixed_piece = np.zeros((piece_height, piece_width, 3), dtype=np.uint8)

            mixed_piece[:, :] = average_color
            row_pieces.append(mixed_piece)

        mixed_pieces.append(row_pieces)

    # 拼接所有格子
    mixed_image = np.zeros(original_image.shape, dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            # 指定每个小块的位置
            x_start = j * piece_width + col_offset
            x_end = (j + 1) * piece_width + col_offset
            y_start = i * piece_height + row_offset
            y_end = (i + 1) * piece_height + row_offset

            mixed_image[y_start:y_end, x_start:x_end] = mixed_pieces[i][j]

    return mixed_pieces, mixed_image

def user_piece_color(image_path="testCases/corner_test.JPG"):
    # 和用户的原图进行比较
    # user_image = cv2.imread("processed.JPG")
    user_image = cv2.imread(image_path)
    # 去掉rgb = 0,0,0 的部分，计算平均颜色
    # Create a mask for black pixels
    mask = np.all(user_image != [0,0,0], axis=-1)

    # Get the non-black pixels
    non_black_pixels = user_image[mask]

    # Compute the average color of the non-black pixels
    user_image_avg_color = np.average(non_black_pixels, axis=0)

    # Convert the average color to uint8
    user_image_avg_color = np.uint8(user_image_avg_color)
 
    cv2.imshow('user image', user_image)


def compare_color(mixed_pieces, user_image_avg_color):
    # 计算在原图中的位置
    for cell in mixed_pieces:
        cell_avg_color = np.average(cell, axis=0)
        if np.array_equal(user_image_avg_color, cell_avg_color):
            print("Found it!")
            cv2.imshow('Found it!', cell)
            break

    # 显示结果
    # cv2.imshow('Mixed Image', mixed_image)
    cv2.waitKey(0)
         

if __name__ == "__main__":
    mixed_pieces, mixed_image = masaico_full()
    cv2.imshow('mixed_image image', mixed_image)
    cv2.waitKey(0)

    # TODO： 无法对齐所有的颜色，debug的颜色与show的颜色不一致
    # user_piece_color()
    # 
    # compare_color()
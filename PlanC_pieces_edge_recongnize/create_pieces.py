import cv2
import numpy as np

def generate_colored_square(size, color):
    # 生成彩色正方形
    square = np.zeros((size, size, 3), dtype=np.uint8)
    square[:, :] = color
    return square

def custom_cut(square, rows, cols):
    # 手动指定小块的位置，确保拼合连续性
    height, width, _ = square.shape
    piece_height = height // rows
    piece_width = width // cols

    pieces = []
    offset = 10
    for i in range(rows):
        for j in range(cols):
            # 指定每个小块的位置
            x_start = j * piece_width
            x_end = (j + 1 + offset) * piece_width
            y_start = i * piece_height
            y_end = (i + 1+ offset) * piece_height

            piece = square[y_start:y_end, x_start:x_end]
            pieces.append(piece)

            offset += 5

    return pieces

# 生成蓝绿色正方形
square_size = 300
blue_green_square = generate_colored_square(square_size, (0, 255, 0))  # 使用蓝绿色 (0, 255, 0)

# 手动指定小块的位置
rows, cols = 3, 3
pieces = custom_cut(blue_green_square, rows, cols)

# 显示结果
for i, piece in enumerate(pieces):
    cv2.imshow(f'Piece {i+1}', piece)
    # save
    cv2.imwrite(f'piece_{i+1}.jpg', piece)

cv2.waitKey(0)
cv2.destroyAllWindows()

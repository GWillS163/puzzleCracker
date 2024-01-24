import cv2
import numpy as np


def generate_puzzle_pieces(num_pieces, puzzle_size):
    puzzle_pieces = []

    for _ in range(num_pieces):
        # 随机生成不同形状的拼图碎片
        shape_type = np.random.choice(['rectangle', 'triangle', 'circle'])

        if shape_type == 'rectangle':
            piece = np.zeros((puzzle_size, puzzle_size), dtype=np.uint8)
            cv2.rectangle(piece, (10, 10), (puzzle_size - 10, puzzle_size - 10), 255, -1)

        elif shape_type == 'triangle':
            piece = np.zeros((puzzle_size, puzzle_size), dtype=np.uint8)
            points = np.array([[puzzle_size // 2, 10], [10, puzzle_size - 10], [puzzle_size - 10, puzzle_size - 10]])
            cv2.fillPoly(piece, [points], 255)

        elif shape_type == 'circle':
            piece = np.zeros((puzzle_size, puzzle_size), dtype=np.uint8)
            cv2.circle(piece, (puzzle_size // 2, puzzle_size // 2), puzzle_size // 2 - 10, 255, -1)

        puzzle_pieces.append(piece)

    return puzzle_pieces


def assemble_puzzle(puzzle_pieces):
    assembled_puzzle = np.zeros((len(puzzle_pieces) * 100, len(puzzle_pieces) * 100), dtype=np.uint8)

    for i, piece in enumerate(puzzle_pieces):
        x_offset = i * 100
        assembled_puzzle[x_offset:x_offset + 100, x_offset:x_offset + 100] = piece

    return assembled_puzzle


# 生成十个拼图碎片
puzzle_pieces = generate_puzzle_pieces(10, 100)

# 组装拼图
assembled_puzzle = assemble_puzzle(puzzle_pieces)

# 显示结果
cv2.imshow('Assembled Puzzle', assembled_puzzle)
cv2.waitKey(0)
cv2.destroyAllWindows()

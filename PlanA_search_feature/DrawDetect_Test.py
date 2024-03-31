import unittest
from drawGrid import *


class MyTestCase(unittest.TestCase):
    def test_calculateTrueGrids(self):
        grid_col_lines = [[[400, 400], [800, 1800]], [[675, 450], [1100, 1850]], [[950, 500], [1400, 1900]],
                          [[1225, 550], [1700, 1950]], [[1500, 600], [2000, 2000]]]
        grid_row_lines = [[[400, 400], [1500, 600]], [[533, 866], [1666, 1066]], [[666, 1333], [1833, 1533]],
                          [[800, 1800], [2000, 2000]]]
        parallelogram = [[400, 400], [1500, 600], [2000, 2000], [800, 1800]]  # 顺时针
        target_part_point = [560, 560]  # answer is 1, 1

        # answer = calculate_grid_position(grid_col_lines, grid_row_lines, target_part_point)
        answer = calculateTrueGrids(grid_row_lines, grid_col_lines, parallelogram, target_part_point)
        self.assertEqual(answer,
                         (1, 1))

    def test_intersection(self):
        self.assertEqual(calculate_intersection([[400, 400], [1500, 600]], [[400, 400], [800, 1800]]),
                         (400.0, 400.0))

    def test_calculate_grid_position(self):
        grid_col_lines = [[[400, 400], [800, 1800]], [[675, 450], [1100, 1850]], [[950, 500], [1400, 1900]],
                          [[1225, 550], [1700, 1950]], [[1500, 600], [2000, 2000]]]
        grid_row_lines = [[[400, 400], [1500, 600]], [[533, 866], [1666, 1066]], [[666, 1333], [1833, 1533]],
                          [[800, 1800], [2000, 2000]]]

        self.assertEqual(calculate_grid_position(
            grid_row_lines, grid_col_lines,
            [560, 560]),
            (1, 1))


if __name__ == '__main__':
    unittest.main()

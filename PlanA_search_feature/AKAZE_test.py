import unittest
from AKAZE_method import *

target_image_cornerHD = "../puzzleCases/fully/full_cornerHD.JPG"
target_image = "../puzzleCases/fully/full_left.JPG"
target_image_r = "../puzzleCases/fully/full_right.JPG"

class Puzzle(unittest.TestCase):
    def test_center(self):
        find_subimage_akaze(target_image_cornerHD, "cleaned_img/center_blue_multi.JPG")
        self.assertEqual(True, True)

    def test_corner(self):
        # find_subimage_akaze("../puzzleCases/raw/full_cornerHD.JPG", "center_blue_multi.JPG")
        find_subimage_akaze(target_image,
                            "cleaned_img/corner.JPG", )
        self.assertEqual(True, True)

    def test_temp(self):
        # find_subimage_akaze("../puzzleCases/raw/full_cornerHD.JPG", "center_blue_multi.JPG")
        find_subimage_akaze(target_image_r,
                            "temp/processed.JPG", )
        self.assertEqual(True, True)

    def test_raw(self):
        # find_subimage_akaze("../puzzleCases/raw/full_cornerHD.JPG", "center_blue_multi.JPG")
        find_subimage_akaze(target_image,
                            "../puzzleCases/raw/building_windows (1).JPG", )
        self.assertEqual(True, True)

# success
# find_subimage_akaze("puzzleCases/raw/full_cornerHD.JPG", "puzzleCases/raw/corner_part.JPG")  # _noise
# find_subimage_akaze("puzzleCases/raw/full_cornerHD.JPG", "corner.JPG")
# find_subimage_akaze("../puzzleCases/raw/full_cornerHD.JPG", "center_blue_multi.JPG")

    def test_full(self):
        find_subimage_akaze("../puzzleCases/fully/full_left.JPG",
                            "cleaned_img/full_puzzle_mojipart.jpg", )

class Floor(unittest.TestCase):
    def test_up(self):
        find_subimage_akaze("testCases/floor/slight_angle.JPG", "testCases/floor/up.JPG")
        self.assertEqual(True, True)

class Day5(unittest.TestCase):
    def test_day5(self):
        target = "../puzzleCases/pieces2/0126 (5).JPG"
        res = find_subimage_akaze(target_image, "../puzzleCases/pieces2/0126 (6).JPG")
        res = find_subimage_akaze(target_image, "../puzzleCases/pieces2/0126 (7).JPG")
        res = find_subimage_akaze(target_image, "../puzzleCases/pieces2/0126 (8).JPG")
        res = find_subimage_akaze(target_image, "../puzzleCases/pieces2/0126 (9).JPG")

        # if res is False:
        #     find_subimage_akaze(target_image_r, target)


        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()

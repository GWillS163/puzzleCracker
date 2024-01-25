import unittest
from AKAZE_method import *

target_image_cornerHD = "../puzzleCases/raw/full_cornerHD.JPG"
target_image = "../puzzleCases/raw/full_left.JPG"

class Puzzle(unittest.TestCase):
    def test_center(self):
        find_subimage_akaze(target_image, "center_blue_multi.JPG") #"puzzleCases/raw/corner_part.JPG")  # _noise
        self.assertEqual(True, True)  # add assertion here

    def test_corner(self):
        # find_subimage_akaze("../puzzleCases/raw/full_cornerHD.JPG", "center_blue_multi.JPG") #"puzzleCases/raw/corner_part.JPG")  # _noise
        find_subimage_akaze(target_image,
                            "cleaned_img/corner.JPG", ) #"puzzleCases/raw/corner_part.JPG")  # _noise
        self.assertEqual(True, True)  # add assertion here

# success
# find_subimage_akaze("puzzleCases/raw/full_cornerHD.JPG", "puzzleCases/raw/corner_part.JPG")  # _noise
# find_subimage_akaze("puzzleCases/raw/full_cornerHD.JPG", "corner.JPG") #"puzzleCases/raw/corner_part.JPG")  # _noise
# find_subimage_akaze("../puzzleCases/raw/full_cornerHD.JPG", "center_blue_multi.JPG") #"puzzleCases/raw/corner_part.JPG")  # _noise

    def test_full(self):
        find_subimage_akaze("../puzzleCases/raw/full_left.JPG",
                            "cleaned_img/full_puzzle_mojipart.jpg", )

class Floor(unittest.TestCase):
    def test_up(self):
        find_subimage_akaze("testCases/floor/slight_angle.JPG", "testCases/floor/up.JPG")
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()

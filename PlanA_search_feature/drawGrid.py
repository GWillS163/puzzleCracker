import os
import numpy as np
# from PIL import Image
# import cv2 from opencv-python
import cv2


#q: 如何安装cv2
#A: pip install opencv-python


main_image = r"D:\Project\puzzleCracker\puzzleCases\fully\full_combine.jpg"
row, col = 25, 40


# 在main_image上画网格
def draw_grid(image, pos, start_p, end_p, rows, cols):
    h, w = image.shape[:2]
    color = (0, 255, 0)

    # row
    h_px = (end_p[1] - start_p[1]) / rows
    row_lines = [int(start_p[1] + h_px * i) for i in range(rows)]
    row_lines.append(end_p[1])
    for r in range(rows + 1):
        cv2.line(image,
                 (start_p[0], int(row_lines[r])),
                 (end_p[0], int(row_lines[r])), color, 3)
        # write text
        position = (start_p[0]-50, int(row_lines[r]))
        if start_p == "right":
            position = (end_p[0]+50, int(row_lines[r]))
        cv2.putText(image, str(r), position, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

    # col
    w_px = (end_p[0] - start_p[0]) / cols
    col_lines = [int(start_p[0] + w_px * i) for i in range(col)]
    col_lines.append(end_p[0])
    for c in range(cols + 1):
        cv2.line(image,
                 (int(col_lines[c]), start_p[1]),
                 (int(col_lines[c]), end_p[1]), color, 3)
        # write text
        position = (int(col_lines[c]), start_p[1]-50)
        if start_p == "right":
            position = (int(col_lines[c]), end_p[1]+50)
        cv2.putText(image, str(c), position, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

    return image


# show
image_obj = cv2.imread(main_image)
# image_obj = draw_grid(image_obj, "left", start_p=[85, 110], end_p=[3014, 3905], rows=row, cols=21, )
image_obj = draw_grid(image_obj, "right", start_p=[3036, 71], end_p=[5932, 3960], rows=row, cols=20,  )


# resize to fit screen
max_height = 800
scale = max_height / image_obj.shape[0]
new_width = int(image_obj.shape[1] * scale)
image_obj = cv2.resize(image_obj, (new_width, max_height))
cv2.imshow("image", image_obj)
cv2.waitKey(0)
cv2.destroyAllWindows()
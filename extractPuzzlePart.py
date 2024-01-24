import cv2
import numpy as np

def segment_image(image_path, rect):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return

    # 初始化GrabCut所需的参数
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 使用GrabCut算法
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # 创建一个新的mask图像，将确定或很可能的前景区域标记为255，其余区域标记为0
    new_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 使用新的mask图像来提取前景区域
    image = image * new_mask[:, :, np.newaxis]

    # 显示结果
    # cv2.imshow('Result', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存结果
    cv2.imwrite('processed.PNG', image)

# 示例用法
# segment_image('puzzleCases/raw/corner_part.JPG', (0, 0, 770, 800))
segment_image('puzzleCases/raw/corner_part_noise.JPG', (0, 0, 770, 800))

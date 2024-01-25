import cv2
import numpy as np

def segment_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return
    
    # resize image to 800px width
    w, h = image.shape[:2]
    rect = (0, 0, w, h)

    new_width = 800
    scale = new_width / image.shape[1]
    new_height = int(image.shape[0] * scale)
    image = cv2.resize(image, (new_width, new_height))
    rect = tuple([int(x * scale) for x in rect])

    # 初始化GrabCut所需的参数
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)


    # method 1: 使用GrabCut算法
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 6, cv2.GC_INIT_WITH_RECT)
 
    # method 2: 使用 颜色阈值分割算法， 保留 中心区域的数据
    # convert to hsv
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # # define range of blue color in HSV
    # lower_blue = np.array([0, 0, 0])
    # upper_blue = np.array([180, 255, 120])
    # # Threshold the HSV image to get only blue colors
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # # keep mask
    # mask = cv2.bitwise_not(mask)
    # # Bitwise-AND mask and original image
    # image = cv2.bitwise_and(image, image, mask=mask)



    # 显示mask图像
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    # 创建一个新的mask图像，将确定或很可能的前景区域标记为255，其余区域标记为0
    new_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 使用新的mask图像来提取前景区域
    image = image * new_mask[:, :, np.newaxis]

    # 显示结果
    # cv2.imshow('Result', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存结果
    get_file_name = lambda path: path.split('/')[-1].split('.')[0] + '_processed.JPG'
    cv2.imwrite('temp/' + get_file_name, image)

# 示例用法
# segment_image('puzzleCases/raw/corner_part.JPG', (0, 0, 770, 800))
# segment_image('puzzleCases/raw/corner_part_noise.JPG', (0, 0, 770, 800))
# segment_image('puzzleCases/raw/pieces2.JPG', (670, 800, 2700, 3500))
segment_image('../puzzleCases/raw/single_pieces (1).JPG')

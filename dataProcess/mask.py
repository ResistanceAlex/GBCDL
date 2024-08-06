import argparse
import cv2
import numpy as np

def main():
    images_bg = r'E:\GBCDL\dataTest\meCT\88\912939224.png'
    # 加载原始输入图像，并展示
    image = cv2.imread(images_bg)
    cv2.imshow("Original", image)
    # 掩码和原始图像具有相同的大小，但是只有俩种像素值：0（背景忽略）、255（前景保留）
    # 构造一个圆形掩码（半径为140px，并应用位运算）
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (160, 200), 60, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    # 展示输出图像
    cv2.imshow("Circular Mask Applied to Image", masked)
    cv2.waitKey()
    TACH_PATTERN_PATH2 = r'./mask.png'
    cv2.imwrite(TACH_PATTERN_PATH2, mask)

if __name__ == "__main__":
    main()

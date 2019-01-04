import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image
source_img = cv2.imread('images/source/source6.jpg')
plt.figure("initial images")
plt.imshow(source_img)
print(source_img.shape)
gray_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

# denoising
denoised_img = cv2.fastNlMeansDenoising(gray_img, None, 10)
denoised_img = cv2.blur(denoised_img, (5, 5))
plt.figure("denoised images")
plt.imshow(denoised_img)

# 边缘检测
canny = cv2.Canny(denoised_img, 40, 80)
plt.figure("canny images")
plt.imshow(canny)

circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=5, maxRadius=300)
# 输出返回值，方便查看类型
print(circles)
# 输出检测到圆的个数
print(len(circles[0]))

print('-------------我是条分割线-----------------')
# 根据检测到圆的信息，画出每一个圆
for circle in circles[0]:
    # 圆的基本信息
    print(circle[2])
    # 坐标行列
    x = int(circle[0])
    y = int(circle[1])
    # 半径
    r = int(circle[2])
    # 在原图用指定颜色标记出圆的位置
    img = cv2.circle(source_img, (x, y), r, (0, 0, 255), -1)
# 显示新图像
plt.figure("final images")
plt.imshow(source_img)

plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image
source_img = cv2.imread('source.jpg', cv2.CV_8UC1)
source_img1 = cv2.imread('source.jpg')
print(source_img1.shape)

# denoising
denoised_img = cv2.fastNlMeansDenoising(source_img, None, 10)

# hist 30 54 117
hist = cv2.calcHist([denoised_img], [0], None, [256], [0, 256])
plt.figure("hist")
plt.plot(hist)

# extract the white block(circle and ellipse) through binary processing
ret, thresh_img = cv2.threshold(denoised_img, 46, 255, cv2.THRESH_BINARY)
plt.figure("white blocks")
plt.imshow(thresh_img)

# delete the ellipse block
# erode
erode_img = cv2.erode(thresh_img, np.ones((3, 3), np.uint8), iterations=3)
# extract circles
contours_img, contours, hierarchy = cv2.findContours(erode_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
area_threshold = 20
height_width_threshold = 48
for i in range(len(contours)):
    # delete the small area block
    area = cv2.contourArea(contours[i])
    if area < area_threshold:
        cv2.drawContours(contours_img, [contours[i]], 0, 0, -1)

    # delete the ellipse block according to the width and height
    else:
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        if (height - width) > height_width_threshold:
            cv2.drawContours(contours_img, [contours[i]], 0, 0, -1)

# dilate
dilate_img = cv2.dilate(contours_img, np.ones((3, 3), np.uint8), iterations=2)
plt.figure("processed image")
plt.imshow(dilate_img)

# get circle center point
processed_img, contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# calc the radius
area, num = 0, 0
for i in range(len(contours)):
    temp = cv2.contourArea(contours[i])
    print("the area of block " + str(i) + " is: " + str(temp))
    if 100 < temp:
        num += 1
        area += temp

average_area = area / num
radius = np.sqrt(average_area / np.pi)
print("radius is: " + str(radius))

# set the image boundary
boundary_max_x = source_img1.shape[1] - radius
boundary_min_x = radius
boundary_max_y = source_img1.shape[0] - radius
boundary_min_y = radius

# get the centre point coordinates
points = []
for i in range(len(contours)):
    rect = cv2.minAreaRect(contours[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    height = abs(box[0][1] - box[2][1])
    width = abs(box[0][0] - box[2][0])
    c_x = int((box[0][0] + box[2][0]) / 2)
    c_y = int((box[0][1] + box[2][1]) / 2)

    # ignore the uncompleted circle in the boundary
    if c_x < boundary_min_x or c_x > boundary_max_x or c_y < boundary_min_y or c_y > boundary_max_y:
        continue

    if float(width) / float(height) > 2:  # deal with two circles connected horizontally
        c_x_1 = int((box[0][0] + box[2][0]) / 2) - int(1.5 * radius)
        c_x_2 = int((box[0][0] + box[2][0]) / 2) + int(1.5 * radius)
        c_y = int((box[0][1] + box[2][1]) / 2)
        points.append((c_x_1, c_y))
        points.append((c_x_2, c_y))
    else:  # normal circle
        points.append((c_x, c_y))

# draw centre points
for item in points:
    cv2.circle(source_img1, item, 3, (255, 255, 0), -1)
plt.figure("circle with centre point")
plt.imshow(source_img1)

plt.show()


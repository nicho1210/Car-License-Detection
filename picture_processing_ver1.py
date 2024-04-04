#masking the color at the edge of the image in the image 'word_1.jpg' to 'word_8.jpg'

import cv2
import numpy as np
from matplotlib import pyplot as plt
## 根据每行和每列的黑色和白色像素数进行图片分割。

# 1、读取图像，并把图像转换为灰度图像并显示
img_ = cv2.imread('plate.jpg')  # 读取图片
img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
# cv2.imshow('gray', img_gray)  # 显示图片
# cv2.waitKey(0)

# 2、将灰度图像二值化，设定阈值是100
ret, img_thre = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

# 4、分割字符
white = []  # 记录每一列的白色像素总和
black = []  # ..........黑色.......
height = img_thre.shape[0]
width = img_thre.shape[1]
white_max = 0
black_max = 0

# 计算每一列的黑白色像素总和
for i in range(width):
    s = 0  # 这一列白色总数
    t = 0  # 这一列黑色总数
    for j in range(height):
        if img_thre[j][i] == 255:
            s += 1
        if img_thre[j][i] == 0:
            t += 1
    white_max = max(white_max, s)
    black_max = max(black_max, t)
    white.append(s)
    black.append(t)
    # print(s)
    # print(t)

arg = False  # False表示白底黑字；True表示黑底白字
if black_max > white_max:
    arg = True

# 分割图像
def find_end(start_):
    end_ = start_ + 1
    for m in range(start_ + 1, width - 1):
        if (black[m] if arg else white[m]) > (0.95 * black_max if arg else 0.95 * white_max):  # 0.95这个参数请多调整，对应下面的0.05（针对像素分布调节）
            end_ = m
            break
    return end_

n = 1
start = 1
end = 2
word = []

while n < width - 2:
    n += 1
    if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
        # 上面这些判断用来辨别是白底黑字还是黑底白字
        # 0.05这个参数请多调整，对应上面的0.95
        start = n
        end = find_end(start)
        n = end
        if end - start > 5:
            cj = img_[1:height, start:end]
            word.append(cj)
            # cv2.imshow('caijian', cj)
            # cv2.waitKey(0)
print(len(word))
for i,j in enumerate(word):
    cv2.imwrite('word_' + str(i) + '.jpg', j)

#blurring the image 'word_1.jpg' to 'word_8.jpg'
for i in range(1, 9):
    img = cv2.imread('word_' + str(i) + '.jpg')
    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite('word_' + str(i) + '.jpg', img)
    plt.subplot(1, 9, i)
    plt.imshow(img, cmap='gray')

#preprocessing the image 'word_1.jpg' to 'word_8.jpg'
for i in range(1, 9):
    img = cv2.imread('word_' + str(i) + '.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img, (28, 28))
    cv2.imwrite('word_' + str(i) + '.jpg', img)
    plt.subplot(1, 9, i)
    plt.imshow(img, cmap='gray')

#cut the image 'word_1.jpg' to 'word_8.jpg'
for i in range(1, 9):
    img = cv2.imread('word_' + str(i) + '.jpg')
    img = img[75:255, 0:150]
    cv2.imwrite('word_' + str(i) + '.jpg', img)
    plt.subplot(1, 9, i)
    plt.imshow(img, cmap='gray')

#resize the image 'word_1.jpg' to 'word_8.jpg' to height 23 pixels, and maintain the aspect ratio
for i in range(1, 9):
    img = cv2.imread('word_' + str(i) + '.jpg')
    h, w = img.shape[:2]
    aspect_ratio = w / h
    new_height = 21
    new_width = int(aspect_ratio * new_height) + 1
    img = cv2.resize(img, (new_width, new_height))
    cv2.imwrite('word_' + str(i) + '.jpg', img)
    plt.subplot(1, 9, i)
    plt.imshow(img, cmap='gray')

#generate a white picture that is 28 * 28
white = np.ones((28, 28, 3), np.uint8) * 255
cv2.imwrite('white' + '.jpg', white)
plt.subplot(1, 9, 9)
plt.imshow(img, cmap='gray')

#paste the image 'word_1.jpg' to 'word_8.jpg' to the image 'white.jpg', and don't change any width or height
for i in range(1, 9):
    img = cv2.imread('word_' + str(i) + '.jpg')
    white = cv2.imread('white.jpg')
    # get the shape of the image and the white background
    h_img, w_img = img.shape[:2]
    h_white, w_white, _ = white.shape
    # calculate the starting point
    start_h = h_white//2 - h_img//2
    start_w = w_white//2 - w_img//2
    # paste the img to the white
    white[start_h:start_h+h_img, start_w:start_w+w_img] = img
    cv2.imwrite('word_' + str(i) + '_whiten' + '.jpg', white)
    plt.subplot(1, 9, i)
    plt.imshow(img, cmap='gray')
"""
for i in range(1, 9):
    img = cv2.imread('word_' + str(i) + '.jpg')
    white = cv2.imread('white.jpg')
    #paste the img to the white
    #h, w = img.shape[:2]
    white[2:23, 6:17] = img
    cv2.imwrite('word_' + str(i) + '_whiten' + '.jpg', white)
    #cv2.imwrite('word_' + str(i) + '_whiten' + '.jpg', img)
    plt.subplot(1, 9, i)
    plt.imshow(img, cmap='gray')
"""
"""
#paste the image 'word_1.jpg' to 'word_8.jpg' to a white background that is 28x28 pixels
for i in range(1, 9):
    img = cv2.imread('word_' + str(i) + '.jpg')
    img = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    #img = cv2.resize(img, (28, 28))
    cv2.imwrite('word_' + str(i) + '_whiten' + '.jpg', img)
    plt.subplot(1, 9, i)
    plt.imshow(img, cmap='gray')
"""
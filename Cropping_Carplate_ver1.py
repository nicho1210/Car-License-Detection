import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

name_car = 'car.jpg'

#display the image
def plt_show(img, title="", gray=False):
    plt.figure(figsize=(10, 10))
    if not gray:
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
#find the white track of the image
def find_end(start, black, black_max, width):
    end = start + 1
    for m in range(start + 1, width - 1):
        if black[m] > 0.95 * black_max:
            end = m
            break
    return end

#read the image
img = cv.imread(name_car)
plt_show(img, "Original Image")

#convert the image to gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt_show(gray, "Gray Image", gray=True)

#Gaussian blur
img_blur = cv.GaussianBlur(gray, (19, 19), 0)
plt_show(img_blur, "Blurred Image", gray=True)

#edge detection
img_canny = cv.Canny(img_blur, 30, 150)
plt_show(img_canny, "Canny Edge Detection", gray=True)  

#find the contours
contours, hierarchy = cv.findContours(img_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

plate_img = None

#assume the largest contour is the license plate
for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = w / float(h)
    if 2 < aspect_ratio < 5 and w > 100 and h > 20:
        plate_img = img[y:y+h, x:x+w]
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        break

plt_show(img, "Detected License Plate on Original Image")
plt_show(plate_img, "License Plate", gray=False)

#save the image to a file
cv.imwrite('plate.jpg', plate_img)

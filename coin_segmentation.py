# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 23:45:20 2025
@author: sushm
"""
import cv2 as cv
import numpy as np
import random
import os

img = cv.imread("D:/courses/vr/photots/images/cooins2.jpg")
result = img.copy()
count = img.copy()
img = cv.resize(img, (500, 500))
result = cv.resize(result, (500, 500))
count = cv.resize(count, (500, 500))
cv.imshow('coins', img)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (7, 7), 2)
edges = cv.Canny(blurred, 50, 200)  
cv.imshow('edges', edges)


contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)



cv.drawContours(result, contours, -1, (0, 0, 255), 2)
cv.imshow("result", result)


mask = np.zeros_like(img, dtype=np.uint8)


save_dir = "D:/courses/vr/photots/images/"
os.makedirs(save_dir, exist_ok=True)


for i, coin in enumerate(contours):
    color = [random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)]
    cv.drawContours(mask, [coin], -1, color, thickness=cv.FILLED)

    M = cv.moments(coin)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])  
        cy = int(M["m01"] / M["m00"])  
        cv.putText(count, str(i+1), (cx, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

    x, y, w, h = cv.boundingRect(coin)
    

    if w > 10 and h > 10:
        segmented_coin = img[y:y+h, x:x+w]
        cv.imwrite(f"{save_dir}/coin_{i+1}.jpg", segmented_coin)

alpha = 0.5  
segmented_output = cv.addWeighted(img, 1-alpha, mask, alpha, 0)
cv.imshow("Segmented Coins", segmented_output)
def count_coins(contours):
    coin_count = len(contours)
    cv.putText(count, f"Total Coins: {coin_count}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
    cv.imshow("count", count)

count_coins(contours)

cv.imwrite("D:/courses/vr/photots/images/1_a_canny.jpg", edges)
cv.imwrite("D:/courses/vr/photots/images/1_a_result.jpg", result)
cv.imwrite("D:/courses/vr/photots/images/1_b_segmented_output.jpg", segmented_output)
cv.imwrite("D:/courses/vr/photots/images/1_c_count.jpg", count)

cv.waitKey(0)
cv.destroyAllWindows()

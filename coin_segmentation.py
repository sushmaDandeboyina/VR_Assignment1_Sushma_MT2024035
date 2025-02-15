# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 23:45:20 2025

@author: sushm
"""
import cv2 as cv
import numpy as np
import random

img = cv.imread("D:/courses/vr/photots/cooins2.jpg")
result = cv.imread("D:/courses/vr/photots/cooins2.jpg")
count = cv.imread("D:/courses/vr/photots/cooins2.jpg")
img = cv.resize(img,(500,500))
result = cv.resize(result,(500,500))
count = cv.resize(count,(500,500))
cv.imshow('coins',img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

blurred =cv.GaussianBlur(gray, (7,7), 2)
cv.imshow('blurred',blurred)

edges = cv.Canny(blurred, 50, 200)
cv.imshow('edges',edges)

border, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
coin_count = len(border)

cv.drawContours(result, border, -1, (0, 0, 255), 2)

cv.imshow("result",result)

mask = np.zeros_like(img, dtype=np.uint8)

for i,coin in enumerate(border):
    color = [random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)] 
    cv.drawContours(mask, [coin], -1, color, thickness=cv.FILLED)
    M = cv.moments(coin)
    if M["m00"] != 0:
       cx = int(M["m10"] / M["m00"])  
       cy = int(M["m01"] / M["m00"])  
       cv.putText(count, str(i+1), (cx, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

alpha = 0.5  
segmented_output = cv.addWeighted(img, 1-alpha, mask, alpha, 0)
cv.imshow("Segmented Coins", segmented_output)
cv.putText(count, f"Total Coins: {coin_count}", (50, 50), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
cv.imshow("count",count)

cv.imwrite("D:/courses/vr/photots/1_a_canny.jpg",edges)
cv.imwrite("D:/courses/vr/photots/1_a_result.jpg", result)
cv.imwrite("D:/courses/vr/photots/1_b_segmented_output.jpg", segmented_output)
cv.imwrite("D:/courses/vr/photots/1_c_count.jpg", count)

cv.waitKey(0)
cv.destroyAllWindows()
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 05:39:09 2025

@author: sushm
"""

import cv2 as cv
import numpy as np
# import random

pano1 = cv.imread("D:/courses/vr/photots/images/pano1.jpg")
pano2 = cv.imread("D:/courses/vr/photots/images/pano2.jpg")
pano3 = cv.imread("D:/courses/vr/photots/images/pano3.jpg")
pano1 = cv.resize(pano1,(500,500))
pano2 = cv.resize(pano2,(500,500))
pano3 = cv.resize(pano3,(500,500))
cv.imshow('pano1',pano1)
cv.imshow('pano2',pano2)
cv.imshow('pano3',pano3)

gray1 = cv.cvtColor(pano1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(pano2, cv.COLOR_BGR2GRAY)
gray3 = cv.cvtColor(pano3, cv.COLOR_BGR2GRAY)
cv.imshow('gray1',gray1)
cv.imshow('gray2',gray2)
cv.imshow('gray3',gray3)

sift = cv.SIFT_create()
kp1_sift, des1_sift = sift.detectAndCompute(gray1, None)
kp2_sift, des2_sift = sift.detectAndCompute(gray2, None)
kp3_sift, des3_sift = sift.detectAndCompute(gray3, None)

img1_sift = cv.drawKeypoints(gray1, kp1_sift, None, color=(255, 0, 0))
img2_sift = cv.drawKeypoints(gray2, kp2_sift, None, color=(255, 0, 0))
img3_sift = cv.drawKeypoints(gray3, kp3_sift, None, color=(255, 0, 0))


cv.imshow("SIFT - Image 1", img1_sift)
cv.imshow("SIFT - Image 2", img2_sift)
cv.imshow("SIFT - Image 3", img3_sift)

bf = cv.BFMatcher()
matches12 = bf.knnMatch(des1_sift, des2_sift, k=2)
matches23 = bf.knnMatch(des2_sift, des3_sift, k=2)


good_matches12 = [m for m, n in matches12 if m.distance < 0.75 * n.distance]
good_matches23 = [m for m, n in matches23 if m.distance < 0.75 * n.distance]


def get_matched_points(good_matches, kpA, kpB):
    ptsA = np.float32([kpA[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return ptsA, ptsB

pts1, pts2 = get_matched_points(good_matches12, kp1_sift, kp2_sift)
pts2_, pts3 = get_matched_points(good_matches23, kp2_sift, kp3_sift)

if len(pts1) >= 4 and len(pts2) >= 4:
    H21, _ = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)
else:
    print("Not enough matches found between pano1 and pano2!")
    exit()

if len(pts2_) >= 4 and len(pts3) >= 4:
    H32, _ = cv.findHomography(pts3, pts2_, cv.RANSAC, 5.0)
else:
    print("Not enough matches found between pano2 and pano3!")
    exit()

H31 = H21 @ H32  


height, width, _ = pano1.shape
canvas_width = width * 3  

img2_warped = cv.warpPerspective(pano2, H21, (canvas_width, height))
img3_warped = cv.warpPerspective(pano3, H31, (canvas_width, height))


panorama = np.zeros((height, canvas_width, 3), dtype=np.uint8)


panorama[:height, :width] = pano1


panorama = np.where(img2_warped > 0, img2_warped, panorama)
panorama = np.where(img3_warped > 0, img3_warped, panorama)

cv.imwrite("D:/courses/vr/photots/images/pano1_sift.jpg",img1_sift)
cv.imwrite("D:/courses/vr/photots/images/pano2_sift.jpg", img2_sift)
cv.imwrite("D:/courses/vr/photots/images/pano3_sift.jpg", img3_sift)
cv.imwrite("D:/courses/vr/photots/images/panorama.jpg", panorama)
cv.imshow("Panorama", panorama)
cv.waitKey(0)
cv.destroyAllWindows()
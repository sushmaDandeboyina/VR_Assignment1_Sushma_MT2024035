import cv2 as cv
import numpy as np


palace1 = cv.imread("D:/courses/vr/photots/images/college_left.jpg")
palace2 = cv.imread("D:/courses/vr/photots/images/college_center.jpg")
palace3 = cv.imread("D:/courses/vr/photots/images/college_right.jpg")

palace1 = cv.resize(palace1, (500, 500))
palace2 = cv.resize(palace2, (500, 500))
palace3 = cv.resize(palace3, (500, 500))

gray1 = cv.cvtColor(palace1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(palace2, cv.COLOR_BGR2GRAY)
gray3 = cv.cvtColor(palace3, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
kp3, des3 = sift.detectAndCompute(gray3, None)

img1_sift = cv.drawKeypoints(gray1, kp1, None, color=(255, 0, 0))
img2_sift = cv.drawKeypoints(gray2, kp2, None, color=(255, 0, 0))
img3_sift = cv.drawKeypoints(gray3, kp3, None, color=(255, 0, 0))


bf = cv.BFMatcher()
matches12 = bf.knnMatch(des1, des2, k=2)
matches32 = bf.knnMatch(des3, des2, k=2)

good_matches12 = [m for m, n in matches12 if m.distance < 0.75 * n.distance]
good_matches32 = [m for m, n in matches32 if m.distance < 0.75 * n.distance]

def get_matched_points(good_matches, kpA, kpB):
    ptsA = np.float32([kpA[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return ptsA, ptsB

pts1, pts2 = get_matched_points(good_matches12, kp1, kp2) 
pts3, pts2_ = get_matched_points(good_matches32, kp3, kp2)  

H12, _ = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0) if len(pts1) >= 4 else None
H32, _ = cv.findHomography(pts3, pts2_, cv.RANSAC, 5.0) if len(pts3) >= 4 else None

if H12 is None or H32 is None:
    print("Error: Not enough keypoints matched!")
    exit()

h, w = palace2.shape[:2]
corners = np.array([
    [[0, 0]], [[0, h]], [[w, 0]], [[w, h]]  
], dtype=np.float32)

warped_corners1 = cv.perspectiveTransform(corners, H12)
warped_corners3 = cv.perspectiveTransform(corners, H32)

x_min = min(warped_corners1[:, 0, 0].min(), 0, warped_corners3[:, 0, 0].min())
x_max = max(warped_corners1[:, 0, 0].max(), w, warped_corners3[:, 0, 0].max())
y_min = min(warped_corners1[:, 0, 1].min(), 0, warped_corners3[:, 0, 1].min())
y_max = max(warped_corners1[:, 0, 1].max(), h, warped_corners3[:, 0, 1].max())

offset_x = int(abs(x_min))
offset_y = int(abs(y_min))


panorama2_width = int(x_max - x_min)
panorama2_height = int(y_max - y_min)

translation_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

H12 = translation_matrix @ H12 
H32 = translation_matrix @ H32  

palace1_warped = cv.warpPerspective(palace1, H12, (panorama2_width, panorama2_height))
palace3_warped = cv.warpPerspective(palace3, H32, (panorama2_width, panorama2_height))
palace2_warped = cv.warpPerspective(palace2, translation_matrix, (panorama2_width, panorama2_height))

panorama2 = np.maximum(np.maximum(palace1_warped, palace2_warped), palace3_warped)
cv.imwrite("D:/courses/vr/photots/images/college_left_sift.jpg",img1_sift)
cv.imwrite("D:/courses/vr/photots/images/college_center_sift.jpg", img2_sift)
cv.imwrite("D:/courses/vr/photots/images/college_right_sift.jpg", img3_sift)
cv.imwrite("D:/courses/vr/photots/images/panorama1.jpg", panorama2)
cv.imshow("panorama2", panorama2)

cv.waitKey(0)
cv.destroyAllWindows()

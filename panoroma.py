import cv2 as cv
import numpy as np

# Load images
left = cv.imread("D:/courses/vr/photots/images/lib1.jpg")
center = cv.imread("D:/courses/vr/photots/images/lib2.jpg")
right = cv.imread("D:/courses/vr/photots/images/lib3.jpg")

left = cv.resize(left, (500, 500))
center = cv.resize(center, (500, 500))
right = cv.resize(right, (500, 500))
cv.imshow("lib1",left)
cv.imshow("lib2",center)
cv.imshow("lib3",right)

gray1 = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(center, cv.COLOR_BGR2GRAY)
gray3 = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
kp3, des3 = sift.detectAndCompute(gray3, None)

img1_sift = cv.drawKeypoints(gray1, kp1, None, color=(255, 0, 0))
img2_sift = cv.drawKeypoints(gray2, kp2, None, color=(255, 0, 0))
img3_sift = cv.drawKeypoints(gray3, kp3, None, color=(255, 0, 0))
cv.imshow("lib1_sift",img1_sift)
cv.imshow("lib2_sift",img2_sift)
cv.imshow("lib3_sift",img3_sift)

bf = cv.BFMatcher()
matches12 = bf.knnMatch(des1, des2, k=2)
matches32 = bf.knnMatch(des3, des2, k=2)

good_matches12 = [m for m, n in matches12 if m.distance < 0.75 * n.distance]
good_matches32 = [m for m, n in matches32 if m.distance < 0.75 * n.distance]

matched_img12 = cv.drawMatches(img1_sift, kp1, img2_sift, kp2, good_matches12, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
matched_img32 = cv.drawMatches(img3_sift, kp3, img2_sift, kp2, good_matches32, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow("common points in left and center",matched_img12)
cv.imshow("common points in center and right",matched_img32)


def get_matched_points(good_matches, kpA, kpB):
    ptsA = np.float32([kpA[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return ptsA, ptsB

pts1, pts2 = get_matched_points(good_matches12, kp1, kp2)  # Left to Center
pts3, pts2_ = get_matched_points(good_matches32, kp3, kp2)  # Right to Center

H12, _ = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0) if len(pts1) >= 4 else None
H32, _ = cv.findHomography(pts3, pts2_, cv.RANSAC, 5.0) if len(pts3) >= 4 else None

if H12 is None or H32 is None:
    print("Error: Not enough keypoints matched!")
    exit()

h, w = center.shape[:2]
corners = np.array([
    [[0, 0]], [[0, h]], [[w, 0]], [[w, h]]  # Corners of center
], dtype=np.float32)

warped_corners1 = cv.perspectiveTransform(corners, H12)
warped_corners3 = cv.perspectiveTransform(corners, H32)

x_min = min(warped_corners1[:, 0, 0].min(), 0, warped_corners3[:, 0, 0].min())
x_max = max(warped_corners1[:, 0, 0].max(), w, warped_corners3[:, 0, 0].max())
y_min = min(warped_corners1[:, 0, 1].min(), 0, warped_corners3[:, 0, 1].min())
y_max = max(warped_corners1[:, 0, 1].max(), h, warped_corners3[:, 0, 1].max())

offset_x = int(abs(x_min))
offset_y = int(abs(y_min))

# New panorama2 size
panorama2_width = int(x_max - x_min)
panorama2_height = int(y_max - y_min)

translation_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

H12 = translation_matrix @ H12 
H32 = translation_matrix @ H32  

left_warped = cv.warpPerspective(left, H12, (panorama2_width, panorama2_height))
right_warped = cv.warpPerspective(right, H32, (panorama2_width, panorama2_height))
center_warped = cv.warpPerspective(center, translation_matrix, (panorama2_width, panorama2_height))

panorama2 = np.maximum(np.maximum(left_warped, center_warped), right_warped)
cv.imwrite("D:/courses/vr/photots/images/lib1_sift.jpg",img1_sift)
cv.imwrite("D:/courses/vr/photots/images/lib2_sift.jpg", img2_sift)
cv.imwrite("D:/courses/vr/photots/images/lib3_sift.jpg", img3_sift)
cv.imwrite("D:/courses/vr/photots/images/panorama.jpg", panorama2)
cv.imshow("panorama", panorama2)

cv.waitKey(0)
cv.destroyAllWindows()

# VR_Assignment1_Sushma_MT2024035
============================================================================================
## Part 1: Use computer vision techniques to Detect, segment, and count coins from an image containing scattered Indian coins.

### Steps Implemented:
- **Detection:**
  - Used Canny edge detection to identify coin edges by converting the image to gray scale as canny's work best with single channeled image and also smoothened the image with guassian blur.
  - Extarcted the outmost contour as it the outline of the coin and marked the ourline on the coins
- **Segmentation:**
  - Created a empty image with all black pixels which acts a mask, drew the identified contours on the mask.
  - Assigned random colors to segmented regions, and overlaid the mask on the original image by keeping the transparecy as 0.5.
- **Counting:**
  - Counted the number of segmented regions and labeled each coin with a number at its centroid location.

 ### How to Run:
1. Ensure you have OpenCV installed:  
   ```sh
   pip install opencv-python numpy
2. Open the terminal, navigate to the script location, and run:
   ```sh
   python coin_segmentation.py
### Input and Output Files(images folder):
    * Input Image: coins2.jpg
    * Canny Edge Detection Output: 1_a_edges.jpg
    * Outlines of Detected Coins: 1_a_result.jpg
    * Segmented Coins: 1_b_segmented_output.jpg
    * Coin Count with Labels: 1_c_count.jpg
-------------------------------------------------------------------------------------------------

## Part 2: Create a stitched panorama from multiple overlapping images.

### Steps Implemented:
- **Key points:**
   - Naming the images in the order from left to right and converting into gray scale
   - Using SIFT algorithm to detect key points and descriptors of the image.
- **Image Stitching:**
   - Applied Brute force algorithm to find the common keypoints in images.
   - Using Lowe's ratio test found good matches.
   - Computed homographies of the images using RANSAC to deal with outliers and also aligned images.
   - Warped images using the estimated transformations and blended them into a single panoramic view.

 ### How to Run:
1. Ensure you have OpenCV installed:  
   ```sh
   pip install opencv-python numpy
2. Open the terminal, navigate to the script location, and run:
   ```sh
   python panoroma.py
### Input and Output Files(images folder):
  #### Eaxample 1:-
    * Input Image1: pano1.jpg
    * Input Image2: pano2.jpg
    * Input Image3: pano3.jpg
    * SIFT on image1:- panp1_sift.jpg
    * SIFT on image2:- pano2_sift.jpg
    * SIFT on image3:- pano3_sift.jpg
    * final result:- panorama.jpg
  #### Eaxample 2:-
    * Input Image1: img1.jpg
    * Input Image2: img2.jpg
    * Input Image3: img3.jpg
    * SIFT on image1:- img1_sift.jpg
    * SIFT on image2:- img2_sift.jpg
    * SIFT on image3:- img3_sift.jpg
    * final result:- panorama1.jpg
  

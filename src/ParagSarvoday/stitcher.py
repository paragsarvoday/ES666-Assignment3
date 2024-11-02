import pdb
import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import imutils

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        homography_matrix_list = []

        stitched_image = None

        img_idx = 1
        for i in range(len(all_images)-1,0,-1):
            
            print(f"Image being stitched: {img_idx}")

            current_img = cv2.imread(all_images[i])
            current_img_resized = imutils.resize(current_img, width=400)

            print(f"Finding homography matrix for the matrix list")

            if stitched_image is None:
                stitched_image = current_img_resized  

            else:
                stitched_image, H_ = self.stitch(stitched_image, current_img_resized)
                homography_matrix_list.append(H_)
            img_idx += 1
        
        return stitched_image, homography_matrix_list



    def stitch(self, img1, img2):
        
        keypoints1, keypoints2 = self.detect_and_match_features(img1, img2)

        H = self.computeHomography_RANSAC(keypoints1, keypoints2)

        if H is None:
            return None

        result = self.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        result[:, 0:400] = img2

        return result, H

    def warpPerspective(self,image, H, size):
        w, h = size
        result = np.zeros((h, w, 3), dtype=image.dtype)

        H_inv = np.linalg.inv(H)

        for y in range(h):
            for x in range(w):
                pt = np.array([x, y, 1.0])
                src_pt = H_inv @ pt
                src_pt /= src_pt[2]

                src_x, src_y = int(src_pt[0]), int(src_pt[1])

                # Check if the coordinates are within bounds of the original image
                if 0 <= src_x < image.shape[1] and 0 <= src_y < image.shape[0]:
                    result[y, x] = image[src_y, src_x]
            
        return result

    def computeHomography_RANSAC(self, points1, points2, num_iters = 2000, samples=4, threshold=4.0):
        max_inliers = 0
        best_H = None
        epsilon = 1e-8

        points1 = np.array(points1)
        points2 = np.array(points2)
        assert points1.shape[0] == points2.shape[0] and points1.shape[0] >= 4 #Need at least 4 point correspondences
        

        #__________Function to simply find the homography matrix________________________
        def computeHomography(points1,points2):
            A = []
            for (x1, y1), (x2, y2) in zip(points1,points2):
                A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
                A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
            A = np.array(A)

            U, S, V = np.linalg.svd(A)
            H = np.reshape(V[-1], (3, 3))
            H = (1 / H.item(8)) * H
            
            return H
        #_________________________________________________________________________________


        for i in range(num_iters):
            random_indices = np.random.choice(range(len(points1)), samples, replace=False)
            random_points1 = points1[random_indices]
            random_points2 = points2[random_indices]
            
            H = computeHomography(random_points1, random_points2)
            
            transformed_points = []
            for pt in points1:
                x, y = pt[0], pt[1]
                transformed_pt = H @ np.array([x, y, 1])
                transformed_pt /= (transformed_pt[2] + epsilon)  # Normalize to make it homogeneous
                transformed_points.append(transformed_pt[:2])
            transformed_points = np.array(transformed_points)
            
            # Count inliers based on threshold
            inliers = []
            for j in range(len(points1)):
                if np.linalg.norm(points2[j] - transformed_points[j]) < threshold:
                    inliers.append(j)
                    
            # Update best homography if more inliers are found
            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_H = computeHomography(points1[inliers], points2[inliers])
            
        return best_H
    
    def detect_and_match_features(self, img1, img2):

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)

        # Filter matches using Lowe's ratio test
        good_matches = []
        good = [] #redundant
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                good.append([m])

        #_____A visual verification hack_________
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()

        # Convert keypoints to coordinates
        if good_matches:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        else:
            print("Unable to find good matches in the two given images")
            pts1, pts2 = np.array([]), np.array([])

        return pts1, pts2
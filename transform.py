# Script for camera calibration

import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Pentru stocare de object points si image points
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/cal*.jpg')

# Cautam chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # Adaugare object point si image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

cv2.destroyAllWindows()

import pickle
# %matplotlib inline

# Testare Distorsiune
img = cv2.imread('camera_cal\\calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Camera calibration pentru object points si image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


dst = cv2.undistort(img, mtx, dist, None, mtx)
# cv2.imwrite('calibration_wide/test_undist.jpg',dst)
# Salvam camera calibration pentru a fi folosit mai tarziu
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "./wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion

# Nu merge plt show
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7))
ax1.imshow(img, cmap='gray')
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst, cmap='gray')
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()
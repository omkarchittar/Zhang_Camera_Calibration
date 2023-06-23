import cv2
import os
import numpy as np

CHECKERBOARD = (6,9)
reproj_error_list = []
error_tot = 0.0
image_pts = [] 
world_pts = []
# Defining the world coordinates for 3D points
world_coords = np.zeros((1, 6 * 9, 3), np.float32)
world_coords[0,:,:2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)*21.5

for img in os.listdir('Calibration_Imgs/'):
    image = cv2.imread('Calibration_Imgs/' + img)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flag, corners = cv2.findChessboardCorners(gray_image, (6,9), None)
    if flag:
        world_pts.append(world_coords)
        image_pts.append(corners)
        image = cv2.drawChessboardCorners(image, (6,9), corners, flag)
        image = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))
        cv2.imshow("Corners Detection", image)
        cv2.waitKey(250)
cv2.destroyAllWindows()

n = len(world_pts)
flag, K, dist, R, T = cv2.calibrateCamera(world_pts, image_pts, gray_image.shape[::-1], None, None)

print("Intrinsic matrix:\n", K, "\n")

print("Images and their reprojection errors:")
for i, img in enumerate(os.listdir('Calibration_Imgs/')):
    proj_pts, _ = cv2.projectPoints(world_pts[i], R[i], T[i], K, dist)
    error = cv2.norm(image_pts[i], proj_pts, cv2.NORM_L2)/54
    reproj_error_list.append(error)
    print(img,": ",error)
    error_tot += error

print("\nMean error = ", error_tot/n)
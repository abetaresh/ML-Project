from skimage.feature import peak_local_max
from skimage.feature.corner import corner_harris, corner_subpix
from skimage.color import rgb2gray
from skimage import io
from skimage.io import imread
import pylab as plt
import numpy as np
import skimage
import cv2 as cv

def dectector(filepath):
    img = skimage.io.imread(filepath, as_gray = True)
   
    # Creates the edgse
    corners =  peak_local_max(corner_harris(img), min_distance=10, threshold_rel=0, num_peaks=1)
    print(corners)
    subpix_edges = skimage.feature.corner_subpix(img, corners)
    # Discards nan values
    nb_of_edges = np.count_nonzero(~np.isnan(subpix_edges))
    return nb_of_edges


feature_corners = list(map(dectector, "images/sample.jpg"))

#-------------With CV2-------------#

img = cv.imread("sample.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray, maxCorners=49, qualityLevel=0.1, minDistance=20)
print(corners)

cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 20, 0.01))

harris = corner_harris(gray)
coords = peak_local_max(harris, min_distance=20, num_peaks=49)

corners_subpix = corner_subpix(gray, coords)

plt.gray()
plt.imshow(img, interpolation='nearest')
plt.plot(corners_subpix[:, 1], corners_subpix[:, 0], '+r')
plt.plot(corners[:, 0, 1], corners[:, 0, 0], '+b')
plt.show()

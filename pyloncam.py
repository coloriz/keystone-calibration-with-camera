from time import sleep
from pypylon import pylon
import cv2
import numpy as np

WINDOW_NAME = 'window'

# original = cv2.imread('images/ZtcjR.jpg')
original = 255 * np.ones((1080, 1920, 3), np.uint8)
cv2.putText(original, 'CV2', (55, 900), cv2.FONT_HERSHEY_SIMPLEX, 30, (0, 255, 0), 10, cv2.LINE_AA)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW_NAME, 1920, 0)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow(WINDOW_NAME, original)
cv2.waitKey()

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

numberOfImagesToGrab = 1
camera.StartGrabbingMax(numberOfImagesToGrab)

distorted = None

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data.
        distorted = grabResult.Array

    grabResult.Release()

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(original, None)
kp2, des2 = orb.detectAndCompute(distorted, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:20]
draw_matches = cv2.drawMatches(original, kp1, distorted, kp2, matches, None, flags=2)

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
M_inv, mask_inv = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

draw_matches = cv2.pyrDown(draw_matches)
cv2.imshow('draw_matches', draw_matches)
original = cv2.pyrDown(original)
original = cv2.pyrDown(original)
tmp = cv2.warpPerspective(original, M_inv, (1920, 1080))
cv2.imshow('tmp', tmp)
cv2.waitKey()
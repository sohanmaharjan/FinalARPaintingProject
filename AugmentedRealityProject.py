# import openCV and numpy libraries
import cv2
import numpy as np

# access mobile camera using IP address
cap = cv2.VideoCapture('http://192.168.101.10:8080/video')
# import images
imgTarget = cv2.imread('TargetImage.jpg')
myImage = cv2.imread('RenderImage2.jpg')

# read video frames from the webcam and store it
success, imgVideo = cap.read()
# get image shape
hT, wT, cT = imgTarget.shape
# change the shape of the image to be rendered to the shape of target image
renderImage = cv2.resize(myImage, (wT, hT))

# detect the features from the target images so that it can be compared later
orb = cv2.ORB_create(nfeatures=15000)
kp1, des1 = orb.detectAndCompute(imgTarget,None)
# draws the key features detected by ORB function
imgTarget = cv2.drawKeypoints(imgTarget,kp1,None)

while True:
    # read frames from the webcam
    success, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)

    # compare the features in targetimage and frames from webcam
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # creating empty matrix to store matching features
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)
    resizedImgFeatures = cv2.resize(imgFeatures, (900, 600))

    # if the features matches then proceed the loop
    if len(good) > 20:
        # source points
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        # destination points
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)

        imgWarp = cv2.warpPerspective(renderImage, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

        resizedImgAug = cv2.resize(imgAug, (900, 600))

    cv2.imshow('imgWarp', imgWarp)
    cv2.imshow('img2', img2)
    cv2.imshow('imgFeatures', resizedImgFeatures)
    cv2.imshow('ImgTarget', imgTarget)
    cv2.imshow('Webcam', imgVideo)
    # cv2.imshow('maskNew', resizedImgAug)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



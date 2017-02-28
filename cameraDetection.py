import os
from os.path import join
import numpy as np
import cv2

##extract 4 distinct points from given QR Code 
def getAnchorPoints(img):
    #convert image to grayscale
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    #threshold image
    ret,thresh = cv2.threshold(imgray,200,255,cv2.THRESH_BINARY) 
    #extract contours in the image
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    targetContours = []
    targetContours2 = [] 
    anchorPoints = []
    areas = []
    center = []
    #save controus with sides 4 
    for cnt in contours:
        epsilon = 0.02*cv2.arcLength(cnt,True) 
        approxCnt = cv2.approxPolyDP(cnt, epsilon,True)
        #ignore contours not having sides 4
        if(len(approxCnt) != 4):
            continue
        targetContours.append(approxCnt)
        #store areas for future referance
        areas.append(cv2.contourArea(approxCnt))

    areas = np.sort(areas)
    areaThreshold = areas[1]+500
    maxArea = areas[len(areas) - 1 ]
    #ignore contours with area outside the range
    for cnt2 in targetContours:
        cntArea = cv2.contourArea(cnt2) 
        if((cntArea > areaThreshold) & (cntArea < maxArea)):
            targetContours2.append(cnt2)
            #store anchor points
            anchor = sum(cnt2) /4
            anchorPoints.append(anchor)
        #store center of the image for the QR code for future refrence
        if(cntArea == areas[len(areas) - 1]):
            center = sum(cnt2) /4
            
    x = np.array(anchorPoints)
    dt = np.dtype([('a', x.dtype), ('b', x.dtype)])
    strct = x.view(dtype=dt).squeeze()
    #remove duplicate entires in ancho points
    anchorInStrcts, idx, inv = np.unique(strct, return_index=True, return_inverse=True)

    radius = 10
    anchorPoints = []
    distance = []
    for anchor in anchorInStrcts :
        anchorPoints.append((anchor[0],anchor[1]))
    #determine the distance form center of QR code to all anchor points
    distFromCenter = np.sum((center - anchorPoints)**2 , axis = 1)
    #the anchor point with least distance is the alignment pattern
    alignmentPt = np.array(anchorPoints[np.argmin(distFromCenter)])
    #determine the distane from alignment pattern to all anchor points
    distFromAlignmentPt = np.sum((alignmentPt - anchorPoints)**2 , axis = 1)
    #the anchor point with maximum distance form aligment pattern is the top left point
    topLeftPt = np.array(anchorPoints[ np.argmax(distFromAlignmentPt)])
    #find the diagonal line by substract alignmentPt and topLeftPt
    diagonal = alignmentPt - topLeftPt
    #find the line between topLeftPt and all anchor points
    sides =   anchorPoints - topLeftPt
    #find the cross product between daigonal and sides
    crossProduct = np.cross(diagonal , sides)
    #the top right point will have positive perpendicular
    topRightPt = np.array(anchorPoints[ np.argmax(crossProduct)])
    #the bottom left point will have negative perpendicular
    bottomLeftPt = np.array(anchorPoints[ np.argmin(crossProduct)])
    #display for debugging 
    cv2.circle(img,(alignmentPt[0], alignmentPt[1]), radius, (0, 255, 255) , thickness=-1)
    cv2.circle(img,(topLeftPt[0] , topLeftPt[1]), radius, (0, 0, 255) , thickness=-1)
    cv2.circle(img,(topRightPt[0] , topRightPt[1]), radius, (0, 255, 0) , thickness=-1)
    cv2.circle(img,(bottomLeftPt[0] , bottomLeftPt[1]), radius, (255, 0, 0) , thickness=-1)
    cv2.drawContours(img, targetContours2, -1, (0,255,0), 3)
    cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    cv2.imshow('Cam',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    anchorPoints = np.array([alignmentPt , topLeftPt , topRightPt ,bottomLeftPt])
    #return 4 anchor points
    return anchorPoints



referenceImage = cv2.imread('E:\Akash\coding chellage\Code\images\pattern.jpg')
#anchor points of referance sample
destinationPts = getAnchorPoints(referenceImage)
imagePath = 'E:\Akash\coding chellage\Code\images'
rows , cols , dim = referenceImage.shape
#estimated 3d coords of anchor points in QR code
objectPts = ((3.5 , -3.5 ,0),(-4 , 4 , 0 ) ,(4 , 4 , 0 ) ,  (-4 , -4 , 0) )
objectPts = np.reshape(objectPts , (1,4 ,3))
objectPts = objectPts /38 
destinationPts = np.reshape(destinationPts , (1,4 ,2))
#calibrate camera for internsic parmeters
rms, camera_matrix, dist_coefs, rvecs, tvecs =  cv2.calibrateCamera(np.asarray(objectPts , dtype=np.float32) , np.asarray(destinationPts , dtype=np.float32),(rows , cols) , None , None)

for imgName in os.listdir(imagePath):
    imgName = join(imagePath , imgName)
    img = cv2.imread(imgName)
    #extract anchor points
    sourcePts = getAnchorPoints(img)
    #slove pnp for rotation vector and translation vector
    retValue, rotVec, transVec =cv2.solvePnP(objectPts,np.asarray(sourcePts , dtype=np.float32) , camera_matrix,0)
    #obtain roation matrix
    rotMat = cv2.Rodrigues(rotVec)[0]
    #obtain camera position
    cameraPosition = -np.matrix(rotMat).T * np.matrix(transVec)
    #appped translation vector to rotation matrix
    transformationMat = np.hstack((rotMat , transVec))
    #extract euler angels
    yaw,pitch,roll = cv2.decomposeProjectionMatrix(transformationMat)[-1]
    '''
    rotMat = cv2.Rodrigues(rotVec)
    transformationMat = np.hstack((rotMat[0] , transVec))
    transformationMat = np.vstack((transformationMat  , [0,0,0,1]))
    transformationMatInv = np.linalg.pinv(transformationMat)
    perspectiveMat = cv2.getPerspectiveTransform(np.asarray(sourcePts, dtype=np.float32) , np.asarray(destinationPts , dtype=np.float32))
    perspectiveMatInv = cv2.getPerspectiveTransform( np.asarray(destinationPts,dtype=np.float32) , np.asarray(sourcePts, dtype=np.float32) )
    
    #point = np.array([0,0,0,1])
    point = np.array([[0],[0],[0],[1]])
    point = transformationMatInv.dot(point) 
    #point = perspectiveMatInv.dot((point[0] , point[1], point[2]))  
    #point = transVec
    point = point / point[2]
    #point = np.asarray(point , np.uint16)
    #cv2.circle(img,(point[0] , point[1]), 10, (255, 0, 0) , thickness=-1)
    #affineImg = np.zeros((np.shape(referenceImage)[0] , np.shape(referenceImage)[1]))
    rows , cols , dim = img.shape
    perspectiveImg = cv2.warpPerspective(img , perspectiveMat ,(300 , 300))
    print (point)
    '''
    '''
    cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    cv2.imshow('Cam',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
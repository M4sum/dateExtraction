import cv2
import numpy as np
import os
from PIL import Image
import pytesseract
import re

def maxArea(contours):
    max_area=0
    pos=0
    for i in contours:
        area=cv2.contourArea(i)
        if area>max_area:
            max_area=area
            pos=i
    return pos

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	return rect

def joinContours(contours):
    LENGTH = len(contours)
    status = np.zeros((LENGTH,1))

    for i,cnt1 in enumerate(contours):
        x = i    
        if i != LENGTH-1:
            for j,cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist = find_if_close(cnt1,cnt2)
                if dist == True:
                    val = min(status[i],status[x])
                    status[x] = status[i] = val
                else:
                    if status[x]==status[i]:
                        status[x] = i+1

    unified = []
    maximum = int(status.max())+1
    for i in range(maximum):
        pos = np.where(status==i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)
    return unified

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def preprocess(filepath):
    img=cv2.imread(filepath)
    r=500.0 / img.shape[1]
    dim=(500, int(img.shape[0] * r))
    img=cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    # print(img_area)
    # cv2.imshow('INPUT',img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(11,11),0)
    return img,gray

def cropReceipt(approx, contours, img_area):
    if len(approx)!=4 or cv2.contourArea(approx)<(0.4*img_area):
        peri=cv2.arcLength(maxArea(joinContours(contours)),True)
        approx=cv2.approxPolyDP(maxArea(joinContours(contours)),0.02*peri,True)
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
    else:
        box = []
        for i in approx:
            box.append(i[0])
        box = np.asarray(box)
    x, y, w, h = cv2.boundingRect(box)
    return box, (w,h)

def extractDate(file_path):
    # filename = 'receipt.jpeg'
    img,gray = preprocess(file_path)
    img_area = (img.shape)[0] * (img.shape)[1]
    edge=cv2.Canny(gray,80,170)
    (contours,_)=cv2.findContours(edge.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    # cv2.drawContours(img,contours,-1,[0,255,0],1)
    # cv2.imshow('Contours',img)
    # cv2.waitKey(0)
    peri=cv2.arcLength(maxArea(contours),True)
    approx=cv2.approxPolyDP(maxArea(contours),0.02*peri,True)
    box, dim = cropReceipt(approx, contours, img_area)
    # cv2.drawContours(img,[box],0,(0,0,255),2)
    # cv2.fillConvexPoly(img, approx, (255,255,153), lineType=8, shift=0)
    # cv2.drawContours(img,hull_list,-1,[0,255,0],1)
    # cv2.imshow('filpoly',img)
    # cv2.waitKey(0)
    # w,h,arr=transform(approx)
    # br = box[0]
    # bl = box[3]
    # tl = box[1]
    # tr = box[2]
    # arr = [tl,tr,br,bl]
    # pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])
    # pts1=np.float32(arr)
    # M=cv2.getPerspectiveTransform(pts1,pts2)
    # dst=cv2.warpPerspective(img,M,(w,h))

    dst = four_point_transform(img,box)

    image=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(dim[0],dim[1]),interpolation = cv2.INTER_AREA)
    img_th = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,8)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    text = pytesseract.image_to_string(img_th)
    # print(text)
    text = text.split(' ')
    regex = '((\d{2})|(\d))[\/-]((\d{2})|(\d))[\/-]((\d{4})|(\d{2}))'
    for t in text:
        if re.match(regex,t):
            return t
    return None
# 07-15-19 - done
# 07/28/18- done
# # 12/28/2017 - done
# # 5/24/2019 - done
# 29-MAY-2019
# 22.05.2019 - done
# 22.05.19 - done
# 28-JUNE-2019
# 03/Jun/2019
# 30-04-2019 - done
# Sep 29, 2018
# 08-Jul-10
# 21/5
# Nov-15
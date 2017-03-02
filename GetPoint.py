#/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
from match import *
import time

## 模块说明：将两幅图像通过SURF算法及金字塔LK光流法获取特征点

# 关键点转换
def keypointToPoint(KEYPOINT):
    point = np.zeros(len(KEYPOINT) * 2, np.float32)    
    for i in range(len(KEYPOINT)):
        point[i * 2] = KEYPOINT[i].pt[0]
        point[i * 2 + 1] = KEYPOINT[i].pt[1]
    return point.reshape(-1,2)
# 连线匹配的点
def MatchPoint(IMG_LETF,IMG_RIGHT,POINTS):
    # 设置偏移，总的长和宽
    OFFSET = IMG_LETF.shape[1]
    WIDTH = IMG_LETF.shape[1]+IMG_RIGHT.shape[1]
    HEIGHT = IMG_LETF.shape[0] if IMG_LETF.shape[0] > IMG_RIGHT.shape[0] else IMG_RIGHT.shape[0]
    # 图像拼接
    IMG_STITCH = np.zeros((HEIGHT,WIDTH,3),np.uint8)
    for x in xrange(HEIGHT):
        for y in xrange(WIDTH):
            if y < OFFSET:
                IMG_STITCH[x][y] = IMG_LETF[x][y]
            else:
                IMG_STITCH[x][y] = IMG_RIGHT[x][y-OFFSET]
    # 连线匹配的点
    for p in POINTS:
        cv2.line(IMG_STITCH,(p[0][0],p[0][1]),(int(p[1][0]+IMG_LETF.shape[1]),p[1][1]),255 * np.random.random(size=3),1,8)
    # 打开图片
    cv2.imshow('MatchPoint',IMG_STITCH)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('SITICH.jpg',IMG_STITCH)

# 获取特征点
def GetPoint(IMG_LEFT_GRAY,IMG_RIGHT_GRAY):
    # 通过SURF算法提取特征点
    # 初始化ORB对象，设置SURF海森特征点阈值为1000
    detector = cv2.SURF(1000)
    # 找出特征点
    kp_left, desc_left = detector.detectAndCompute(IMG_LEFT_GRAY, None)
    kp_right, desc_right = detector.detectAndCompute(IMG_RIGHT_GRAY, None)
    # 转换关键点
    pt_left = keypointToPoint(kp_left)        
    pt_right = keypointToPoint(kp_right)
    # 金字塔LK光流法预测下一帧小运动偏移点
    pyramid = dict(winSize=(10,10), maxLevel=2, criteria=(3L,10,0.03))                          
    pt_right, status, error = cv2.calcOpticalFlowPyrLK(IMG_LEFT_GRAY, IMG_RIGHT_GRAY, pt_left, None, **pyramid)
    # 过滤错误率高的点
    HIGH_ERRS = {}
    for i in range(len(pt_right)):
         if status[i][0] == 1 and error[i][0] < 12:
             HIGH_ERRS[i] = pt_right[i]
         else:
             status[i] = 0
    # BF匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc_left, desc_right)
    # 获取距离
    dist = [m.distance for m in matches]
    # 根据距离域值筛选
    maxdist = np.median(dist)
    rest = [m for m in matches if m.distance < maxdist]
    # 选取两幅图像中经过筛选之后的点
    points = np.float32([(kp_left[m.queryIdx].pt,kp_right[m.trainIdx].pt) for m in rest])
    return points

if __name__ == "__main__":
    start = time.time()
    # 加载分别从两个摄像头传入的图像
    IMG_LETF = cv2.imread('PRI_LEFT.jpg')
    IMG_RIGHT = cv2.imread('PRI_RIGHT.jpg')
    # 获取图像矩阵
    # ROWS, COLS, CHANNELS = IMG_LETF.shape
    # 颜色空间转换
    IMG_LETF = cv2.cvtColor(IMG_LETF, cv2.COLOR_BGR2RGB)
    IMG_RIGHT = cv2.cvtColor(IMG_RIGHT, cv2.COLOR_BGR2RGB)
    # 转换为灰度图
    IMG_LEFT_GRAY = cv2.cvtColor(IMG_LETF, cv2.COLOR_RGB2GRAY)
    IMG_RIGHT_GRAY = cv2.cvtColor(IMG_RIGHT, cv2.COLOR_RGB2GRAY)
    # 连线匹配的点
    end = time.time()
    MatchPoint(IMG_LETF,IMG_RIGHT,GetPoint(IMG_LEFT_GRAY,IMG_RIGHT_GRAY))
    # print 'Time costs : %f ' % (end - start)


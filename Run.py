#/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import time
from match import *
from GetPoint import *



if __name__ == "__main__":
	start = time.time()
	# 加载分别从两个摄像头传入的图像
	IMG_LETF = cv2.imread('IMG_LEFT.jpg')
	IMG_RIGHT = cv2.imread('IMG_RIGHT.jpg')
	# 获取图像矩阵
	# ROWS, COLS, CHANNELS = IMG_LETF.shape
	# 颜色空间转换
	IMG_LETF = cv2.cvtColor(IMG_LETF, cv2.COLOR_BGR2RGB)
	IMG_RIGHT = cv2.cvtColor(IMG_RIGHT, cv2.COLOR_BGR2RGB)
	# 转换为灰度图
	IMG_LEFT_GRAY = cv2.cvtColor(IMG_LETF, cv2.COLOR_RGB2GRAY)
	IMG_RIGHT_GRAY = cv2.cvtColor(IMG_RIGHT, cv2.COLOR_RGB2GRAY)
	end = time.time()
	print 'totally costs: %f' % (end - start)
	# 连线匹配的点
	MatchPoint(IMG_LETF,IMG_RIGHT,GetPoint(IMG_LEFT_GRAY,IMG_RIGHT_GRAY))
	
	
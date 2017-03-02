#coding:utf-8
import cv2
import numpy as np
import glob


# 模块说明：相机标定模块

def Calibration(PATH,WIDTH,HEIGHT,PX,PY):
	# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
	CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 30 ,0.001)
	# 设置棋盘规格，即角点个数为 witdth x height 
	# WIDTH = 7
	# HEIGHT = 7
	#  设置用于标定的图片像素
	SIZE = (PX,PY)
	# 世界坐标系中，角点坐标格式,去掉Z坐标，为二维矩阵
	obj_p = np.zeros((WIDTH * HEIGHT,3),np.float32)
	obj_p[:,:2] = np.mgrid[0:WIDTH,0:HEIGHT].T.reshape(-1,2)
	# 世界坐标系和图像坐标系点集
	obj_points = []
	img_points = []

	IMAGES = glob.glob(PATH)
	for pf in IMAGES:
		IMG = cv2.imread(pf)
		# 灰度图
		IMG_GRAY = cv2.cvtColor(IMG,cv2.COLOR_BGR2GRAY) 
		# 获取角点 
		RET,CORNERS = cv2.findChessboardCorners(IMG_GRAY,(WIDTH,HEIGHT),None)

		if RET == True:
			obj_points.append(obj_p)
			# 寻找亚像素点, 搜索窗口(winsize为11x11)，无死区
			SUBCORNERS = cv2.cornerSubPix(IMG_GRAY,CORNERS,(5,5),(-1,-1),CRITERIA)
			if SUBCORNERS is not None:
				img_points.append(SUBCORNERS)
			else:
				img_points.append(CORNERS)
	# 标定:
	# MTX : 内参数矩阵
	# DIST: 畸变系数
	# RVS : 旋转向量(外参数)
	# TVS : 平移向量(外参数)
	RET,MTX,DIST,RVS,TVS = cv2.calibrateCamera(obj_points,img_points,SIZE,None,None)
	return (IMG_GRAY,obj_points,img_points,RET,MTX,DIST,RVS,TVS)

# 去除畸变
def RemvDist(IMG_GRAY,MTX,DIST):
	H,W = IMG_GRAY.shape[:2]
	# 自由设置比例参数
	ncmtx,roi = cv2.getOptimalNewCameraMatrix(MTX,DIST,(W,H),1,(W,H))
	# 使用undistort函数去畸变
	dst = cv2.undistort(IMG_GRAY,MTX,DIST,None,ncmtx)
	return dst

# 反投影误差
def BackProjErr(obj_points,img_points,RET,MTX,DIST,RVS,TVS):
	Errs = 0
	AvgErr = 0
	for obj_p in xrange(len(obj_points)):
		img_points_pr ,_ = cv2.projectPoints(obj_points[obj_p],RVS[obj_p],TVS[obj_p],MTX,DIST)
		Errs += cv2.norm(img_points[obj_p],img_points_pr,cv2.NORM_L2)/len(img_points_pr)
	AvgErr = Errs / len(obj_points)
	return Errs,AvgErr

# 画线
def DrawAxis(img, corners, imgpoints):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpoints[0].ravel()), 255 * np.random.random(size=3), 5)
    cv2.line(img, corner, tuple(imgpoints[1].ravel()), 255 * np.random.random(size=3), 5)
    cv2.line(img, corner, tuple(imgpoints[2].ravel()), 255 * np.random.random(size=3), 5)
    return img
# 画立体对象
def DrawCube(img, corners, imgpoints):    
	imgpoints = np.int32(imgpoints).reshape(-1,2)
	cv2.drawContours(img, [imgpoints[:4]], -1, 255 * np.random.random(size=3), -3)
	for i,j in zip(range(4), range(4,8)):
		cv2.line(img, tuple(imgpoints[i]), tuple(imgpoints[j]), 255 * np.random.random(size=3), 3)
	cv2.drawContours(img, [imgpoints[4:]], -1, 255 * np.random.random(size=3), 3)
	return img
# 图片尺寸
def SizeofPic(PATH):
	pf = cv2.imread(PATH)
	return pf.shape[::-1]

if __name__ == '__main__':
	(IMG_GRAY,obj_points,img_points,RET,MTX,DIST,RVS,TVS) = Calibration("./assets/calibration/*.JPG",7,7,1024,768)
	# print BackProjErr(obj_points,img_points,RET,MTX,DIST,RVS,TVS)
	# IMG = RemvDist(IMG_GRAY,MTX,DIST)






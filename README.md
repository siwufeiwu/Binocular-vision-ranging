# Binocular-vision-ranging



## 基于OpenCV双目视觉测距
假设左右摄像头拍摄的图片为IMG_LEFT.jpg和IMG_RIGHT.jpg

## GetPoint模块:
基于SURF算子和LK光流法提取特征点


![stitch](https://github.com/sysorem/Binocular-vision-ranging/blob/master/assets/SITICH.jpg)


# Calibration模块
原图(用于标定的原图均在./assets/calibration/目录)
![stitch](https://github.com/sysorem/Binocular-vision-ranging/blob/master/assets/calibration_origin.jpg)
标定后
![stitch](https://github.com/sysorem/Binocular-vision-ranging/blob/master/assets/calibration.jpg)
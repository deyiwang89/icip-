# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:13:36 2020

@author: deyiwang@qq.com

实现功能: 选取两张图片的控制点, 完成对应视角的转换

参考代码: https://blog.csdn.net/fanzy1234/article/details/103072723?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param
以及: https://blog.csdn.net/qq_36281080/article/details/103928487
"""

from imutils import perspective
from skimage.filters import threshold_local
import cv2
import imutils
import numpy as np

def SetPoints(windowname, img):
    """
    输入图片，打开该图片进行标记点，返回的是标记的几个点的字符串
    """
    print('(提示：单击需要标记的坐标，Enter确定，Esc跳过，其它重试。)')
    points = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(temp_img, (x, y), 10, (102, 217, 239), -1)
            points.append([x, y])
            cv2.imshow(windowname, temp_img)

    temp_img = img.copy()
    cv2.namedWindow(windowname)
    cv2.imshow(windowname, temp_img)
    cv2.setMouseCallback(windowname, onMouse)
    key = cv2.waitKey(0)
    if key == 13:  # Enter
        print('坐标为：', points)
        del temp_img
        cv2.destroyAllWindows()
        return points
    elif key == 27:  # ESC
        print('跳过该张图片')
        del temp_img
        cv2.destroyAllWindows()
        return
    else:
        print('重试!')
        return SetPoints(windowname, img)
    
def extra_point(aol,img):
    points = SetPoints(str(aol), img)
    temp = np.array(points)
    pts = temp.reshape(4, 2).astype(np.float32)
    return pts

def read_img(path):
    image = cv2.imread(path)
    w=image.shape[0]
    h=image.shape[1]
    ratio = image.shape[0] / 500.0  # 比例
    orig = image.copy()
    image = imutils.resize(image, height=500)
    return image
    
def main():

    image1 = read_img("real.jpg")
    image2 = read_img("satellite.png")
    
    point1 = extra_point('real',image1)
    point2 = extra_point("sat", image2)
    
    
    #实际坐标点和提取的角点必须一一对应呀，
    M = cv2.getPerspectiveTransform(point1,point2)
    out_img = cv2.warpPerspective(image1,M,(image1.shape[0],700))
    dst=cv2.perspectiveTransform(point2.reshape(1,4,2), M)
     
     
    cv2.imshow("Original", image1)
    cv2.imshow("Scanned",cv2.resize(out_img,(image1.shape[0],700)))
    
if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 10:58:28 2020

@author: deyiwang@qq.com

本代码用来根据图片仿射变换矩阵转换yolohead输出的anchor的坐标, 进而可以映射到仿射变换后的图片中
其中, list1为一张图片的yolo输出: 类, confidence, (x,y,w,h)

坐标转换公式: https://blog.csdn.net/weixin_42905141/article/details/100745097?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v28-1-100745097.nonecase&utm_term=warpperspective&spm=1000.2123.3001.4430
"""


import numpy as np 
import cv2


def cord_trans(x,y):
    point1 = np.array([[90, 555], [661, 399], [1104, 641], [65, 674]])
    point2 = np.array([[188, 132], [510, 170], [472, 498], [154, 221]])
    point1 = point1.reshape(4, 2).astype(np.float32)
    point2 = point2.reshape(4, 2).astype(np.float32)
    #实际坐标点和提取的角点必须一一对应呀，
    M = cv2.getPerspectiveTransform(point1,point2)
    h = M.reshape(M.shape[0]*M.shape[1],1)
    #dist(x1-x2) = (h1x1+h2y1+h3)/(h7*x1+h8y1+h9)
    # dist(y1-y2) = (h4x1+h5y1+h6)/(h7*x1+h8y1+h9)
    bottom = h[6]*x+h[7]*y+h[8]
    x2 = (h[0]*x+h[1]*y+h[2])/bottom
    y2 = (h[3]*x+h[4]*y+h[5])/bottom
    return x2+x,y2+y
    
def apd_lists(list1):
    list_fin = []
    for i in list1:
        list_num = []
        list_upper = []
        list_upper.append(str(i[0]))
        list_upper.append(str(i[1]))
        # print(list_sub)
        new_tempx, new_tempy = cord_trans(i[2][0], i[2][1])
        list_num.append(float(new_tempx))
        list_num.append(float(new_tempy))
        list_num.append(i[2][2])
        list_num.append(i[2][3])
        list_num =tuple(list_num)
        # print(list_num)
        list_upper.append(list_num)
        # print(list_upper)
        list_fin.append(tuple(list_upper))
        # print(len(list_fin))
    return list_fin

def main():
    list1 = [('traffic light', '35.73', (200.47169494628906, 315.829345703125, 5.129374027252197, 19.77591896057129)), \
             ('person', '41.78', (349.8056640625, 346.9128112792969, 7.044824600219727, 23.11626625061035)), \
            ('car', '56.26', (8.933835983276367, 426.48382568359375, 18.36466407775879, 49.03221893310547)),\
            ('traffic light', '74.61', (209.309326171875, 324.0459899902344, 5.559812545776367, 23.085290908813477)),\
                ('bus', '99.66', (89.80174255371094, 386.2050476074219, 173.25027465820312, 152.223388671875))]

    print(apd_lists(list1))

    

    
if __name__=="__main__":
    main()
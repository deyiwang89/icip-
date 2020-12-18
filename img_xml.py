# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:14:32 2020

@author: deyiwang@qq.com

"""
#######################################
# 用途概述, 
# 对SSDD数据集按照<Attention Receptive Pyramid Network for Ship Detection in SAR Images>
# 中描述的方式 进行缩放-->>短边==350
# 同时修改 xml 文档内坐标
#######################################
import cv2
import xml.etree.ElementTree
from  xml.etree.ElementTree import parse
import xml.dom.minidom
import os


def transform_ratio(xmin, ymin, xmax, ymax, ratio):
    ################################
    # 根据对角坐标转换中心点坐标, 放缩后再转换成新的对角坐标
    ################################
    xmin = float(xmin)
    ymin = float(ymin)
    xmax = float(xmax)
    ymax = float(ymax)
    c_x = (xmin + xmax)/2.*ratio
    c_y = (ymin + ymax)/2.*ratio
    w_2 = abs(xmax - xmin)*ratio/2
    h_2 = abs(ymax - ymin)*ratio/2
    return int(c_x - w_2), int(c_y - h_2), int(c_x + w_2), int(c_y + h_2)
    

def change_xml_and_img(xml_path, img_path, out_img_path, out_xml_path):
    #####################################
    # 用来根据路径修改图片尺寸和对应xml文件内容
    #####################################
    dom=xml.dom.minidom.parse(xml_path)
    root=dom.documentElement
    #获取标签对
    pic_w = root.getElementsByTagName('width')
    pic_h = root.getElementsByTagName('height')
    xmin=root.getElementsByTagName('xmin')
    ymin=root.getElementsByTagName('ymin')
    xmax=root.getElementsByTagName('xmax')
    ymax=root.getElementsByTagName('ymax')
    # 取得比例值
    width = pic_w[0].firstChild.data
    height = pic_h[0].firstChild.data
    short_edge = min(width, height)
    if short_edge==height:
        ratio = 350/float(height)
        pic_h[0].firstChild.data = 350
        pic_w[0].firstChild.data = str(int(ratio * float(width)))
        
    else:
        ratio = 350/float(width)
        pic_w[0].firstChild.data = 350
        pic_h[0].firstChild.data = str(int(ratio * float(height)))
    
    # 修改bbox的坐标 
    # 根据anchor的中心坐标和wh转换来进行放缩
    for i in range(len(xmin)):
        
        xmin1 = xmin[i].firstChild.data
        ymin1 = ymin[i].firstChild.data
        xmax1 = xmax[i].firstChild.data
        ymax1 = ymax[i].firstChild.data
        
        new_xmin, new_ymin, new_xmax, new_ymax = transform_ratio(xmin1, ymin1, xmax1, ymax1, ratio)
        
        xmin[i].firstChild.data = str(new_xmin)
        ymin[i].firstChild.data = str(new_ymin)
        xmax[i].firstChild.data = str(new_xmax)
        ymax[i].firstChild.data = str(new_ymax)
    
    
    
    #修改属性值 使其不为零
    for j in range(len(xmin)):
        xmin[j].firstChild.data=int(float(xmin[j].firstChild.data))
        if (float(xmin[j].firstChild.data)< 0):
            xmin[j].firstChild.data=0
        # print(xmin[j].firstChild.data)
    #ymin
    for u in range(len(ymin)):
        ymin[u].firstChild.data=int(float(ymin[u].firstChild.data))
        if (float(ymin[u].firstChild.data)< 0):
            ymin[u].firstChild.data=0
        # print(ymin[u].firstChild.data)
    #xmax
    for v in range(len(xmax)):
        xmax[v].firstChild.data=int(float(xmax[v].firstChild.data))
        if (float(xmax[v].firstChild.data)< 0):
            xmax[v].firstChild.data=0
        # print(xmax[v].firstChild.data)
    #ymax
    for s in range(len(ymax)):
        ymax[s].firstChild.data=int(float(ymax[s].firstChild.data))
        if (float( ymax[s].firstChild.data)< 0):
            ymax[s].firstChild.data=0
        # print( ymax[s].firstChild.data)
    #保存修改到xml文件中
    with open((xml_path),'w') as wn:
        dom.writexml(wn)
        print(xml_path,"修改完成！")
    with open((out_xml_path),'w') as wn1:
        dom.writexml(wn1)
        print(out_xml_path,"修改完成！")
    #保存resized 图片
    # 根据新的尺寸resize 
    img = cv2.imread(img_path)
    img_test = cv2.resize(img, (int(pic_w[0].firstChild.data), int(pic_h[0].firstChild.data))) 
    cv2.imwrite(out_img_path, img_test)
    
if __name__=="__main__":
    xml_dir_path = "F:\\ssdd_test"
    xml_out_path = "F:\\ssdd_test1"
    img_dir_path =  "F:\\ssdd_test"
    img_out_path =  "F:\\ssdd_test1"
    
    if not os.path.exists(img_out_path):
        os.mkdir(img_out_path)

    for i in os.listdir(xml_dir_path):
        if i[-4:]==".xml":
            xml_file_path = os.path.join(xml_dir_path,i)
            xml_file_save_path = os.path.join(xml_out_path,i)
            img_name = i[:-4] + ".jpg"
            img_file_path = os.path.join(img_dir_path,img_name)
            img_file_save_path = os.path.join(img_out_path,img_name)
            # print(xml_file_path, img_file_path,img_file_save_path)
            change_xml_and_img(xml_file_path, img_file_path,img_file_save_path, xml_file_save_path)

                
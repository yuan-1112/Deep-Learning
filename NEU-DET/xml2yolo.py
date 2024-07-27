import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob

#classes代表类别数0         1             2           3                     4             5
classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

def convert(size, box):
    dw = 1./size[0]#总长为200，则dw为1/200
    dh = 1./size[1]
    #x y w h 分别是中心点的坐标和相对的宽和高
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    #归一化
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_name):
    in_file = open('./ANNOTATIONS/'+image_name[:-3]+'xml')#读进来xml文件
    out_file = open('./LABELS/'+image_name[:-3]+'txt','w')
    tree=ET.parse(in_file)#解析xml格式
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)#长
    h = int(size.find('height').text)#宽

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            print(cls)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')#得到bndbox
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')#写进txt文件

wd = getcwd()

#幂函数
if __name__ == '__main__':
    for image_path in glob.glob("./IMAGES/*.jpg"):#遍历
        image_name = image_path.split('\\')[-1]
        #print(image_path)
        convert_annotation(image_name)# 转换标签



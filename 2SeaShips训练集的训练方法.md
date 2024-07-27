## SeaShips7000训练集的训练方法

### 1.首先下载SeaShips训练集，以及yolov10模型

2.数据集制作

1.1转换格式

由于SeaShip数据集的格式不是标准的voc格式需要转换一下：（voc_seaship.py）

```python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('SeaShip', 'train'), ('SeaShip', 'val')]

classes = ["ore carrier", "general cargo ship", "bulk cargo carrier", "container ship", "fishing boat", "passenger ship"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('Annotations/%s.xml'%(image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        #difficult = obj.find('difficult').text
        cls = obj.find('name').text
        #if cls not in classes or int(difficult) == 1:
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(year, image_id)
    list_file.close()

```

> **目录结构：**
> **.**
> **├── Annotations**
> **├── ImageSets**
> **├── JPEGImages**
> **├── SeaShip_train.txt**
> **├── SeaShip_val.txt**
> **├── VOCdevkit**
> **└── voc_seaship.py**

生成好后把label里面的.txt文件放进./VOCdevkit/VOCSeaShip/JPEGImages
里面这步很重要

2.2环境搭建python3.9，torch2.0.1+cu117，cuda11.7，cudnn8500

2.3使用yolov10m.ymal作为配置文件，使用yolov10m.pt作为权重

2.4制作自己的.yaml文件，在coco.ymal的基础上进行修改，复制一份新的

2.5使用trainseaship进行训练


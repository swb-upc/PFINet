from PIL import Image
import numpy
import torch.nn as nn
import torch
import numpy as np
import os
from shutil import copyfile


def test1():
    image_path = "D:/学习/DataSet/ReID/SYSU-MM01/SYSU-MM01/cam2/0002/0001.jpg"
    rgb = [[[0.299]],[[0.487]],[[0.144]]]

    w_rgb = np.array(rgb).repeat(256,axis=1).repeat(128,axis=2)
    print(w_rgb.shape)

    image = Image.open(image_path).resize((128,256))
    image = np.array(image).transpose(2,0,1)
    print(image.shape)

    h = np.sum((w_rgb * image),axis=0,keepdims=True).repeat(3,axis=0)
    print(h.shape)
    image = Image.fromarray(np.uint8(h.transpose(1,2,0)))
    image.show()
def test2():
    a = torch.tensor([0,1,2,3])
    b = a.expand((4,4))
    print(b)

def test3():
    a = np.random.random((10,5,5))
    # b * n * n
    c = np.sum((a[:,:,] > 0.5),axis=-1,keepdims=True)
    print(c)

def test4():
    image = Image.open('D:\学习\DataSet\VOC2012\SegmentationClass\\2007_000480.png')
    image.show()
    image = np.array(image)
    image = image == 15
    print(image.shape)
    print(image)
    image = Image.fromarray(image)
    image.show()

def test5():
    mask_dir = 'C:/Users/swb792/Desktop/person_seg/mask'
    source_dir = 'D:/学习/DataSet/VOC2012/JPEGImages'
    target_dir = 'C:/Users/swb792/Desktop/person_seg/image'
    for _,_,files in os.walk(mask_dir):
        #for file in files:

            # =========convet RGB to gray
            # image = Image.open(os.path.join(target_dir,file)).convert('L')
            # image = np.array(image)
            # image = np.concatenate([np.array([image]),np.array([image]),np.array([image])])
            # image = Image.fromarray(np.uint8(image).transpose(1,2,0))
            # image.save(os.path.join(target_dir,file))

            # =========修改 多分类改为仅分类行人
            # image = Image.open(os.path.join(mask_dir,file))
            # image = np.array(image)
            # image = image == 15
            # image = Image.fromarray(image)
            # image.save(os.path.join(mask_dir,file))

        for i,file in enumerate(files):
            os.rename(os.path.join(mask_dir,file),os.path.join(mask_dir,str(i)+'.png'))
            os.rename(os.path.join(target_dir,file.replace('png','jpg')),os.path.join(target_dir,str(i)+'.jpg'))


#图像红色通道
def test6():
    path = 'D:/学习/DataSet/ReID/SYSU-MM01/SYSU-MM01/cam1/0004/0001.jpg'
    img = Image.open(path)
    img = np.array(img).transpose((2,0,1))
    img = img[0]
    img = img[np.newaxis,:]
    print(img.shape)
    img = Image.fromarray(np.concatenate([img,img,img]).transpose(1,2,0))
    img.show()
if __name__ == '__main__':
    test6()





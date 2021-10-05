from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math
import numpy as np

class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)

class Transform_RGB_to_Infrared():
    def __init__(self,cfg):
        self.weight = cfg.w_RGB
        self.cfg = cfg
    def __call__(self, img):
        w_rgb = np.array(self.weight).repeat(self.cfg.image_size[0], axis=1).repeat(self.cfg.image_size[1], axis=2)
        h = np.sum((w_rgb * img), axis=0, keepdims=True).repeat(3, axis=0)
        # image = Image.fromarray(np.uint8(h.transpose(1, 2, 0)))
        # image.save('1.png')
        return h
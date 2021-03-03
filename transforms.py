from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random

class Random2DTranslation(object):
    def __init__(self,height,width,p=0.5,interploation=Image.BILINEAR): #选择插值方法
        self.height = height
        self.width = width
        self.p = p
        self.interploation = interploation
    
    def __call__(self,img):
        if random.random() < self.p:
            return img.resize((self.width,self.height),self.interploation)
        # 长和宽都扩大到原来的 9/8
        new_width,new_height = int(round(self.width*1.125)),int(round(self.height*1.125)) # round取上限
        resize_img = img.resize((new_width,new_height),self.interploation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        
        x1 = int(round(random.uniform(0,x_maxrange))) # random.uniform 随机生成一个实数，它在 [x,y) 范围内
        y1 = int(round(random.uniform(0,y_maxrange)))

        croped_img = resize_img.crop((x1,y1,x1+self.width,y1+self.height))

        return croped_img

if __name__ == '__main__':
    img = Image.open('')
    transform = Random2DTranslation(256,128,0.5) 
    img_t = transform(img)
    
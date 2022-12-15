# -*- encoding: utf-8 -*-
#Time        :2020/12/13 19:25:28
#Author      :Chen
#FileName    :load_img.py
#Version     :1.0

import PIL.Image as Image
import torchvision.transforms as transforms

img_size = 256

def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((256, 256))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img

def show_img(img):
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img.show()
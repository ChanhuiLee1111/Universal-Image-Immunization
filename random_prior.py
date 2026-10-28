import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
import torch.storage
#from functions import *
from skimage import filters
from skimage.morphology import disk
#from strategy import *
import random

def get_ranlist(num,min=0,max=225):
    list = [min]
    for i in range(num):
        x = random.randrange(min,max,step=1)
        list.append(x)
    return sorted(list)

def shuffle(img,wide=5,high=7,min=0,max=256):
    #assert mode in [0, 1], 'check shuffle mode'
    _, H, W = img.shape
    wide_list = get_ranlist(wide, max=H + 1)
    high_list = get_ranlist(high, max=W + 1)
    for i in range(len(wide_list)):
        w_start = wide_list[i]
        w_end = wide_list[i + 1] if i < len(wide_list) - 1 else H
        for j in range(len(high_list)):
            h_start = high_list[j]
            h_end = high_list[j + 1] if j < len(high_list) - 1 else W
            for c in range(img.shape[0]):
                img[c, w_start:w_end, h_start:h_end] = random.randrange(min, max, 1)
    return img

def get_jigsaw(img,bs, fre=2,min=0,max=256,filter=False):
    img_shape = torch.zeros_like(img.cpu().detach()).squeeze(0)
    img_batch = torch.zeros_like(img.cpu().detach()).squeeze(0)

    for j in range(bs):
        ximg = shuffle(img_shape, fre+2, fre, min,max)

        if filter == True:
            ximg = ximg.numpy()
            for i in range(len(ximg)):
                ximg[i] = filters.median(ximg[i].astype(np.float32), disk(5))
            ximg = torch.Tensor(ximg)
        ximg = ximg.unsqueeze(0)
        ximg = ximg / 255
        if j == 0:
            img_batch = ximg
        else:
            img_batch = torch.cat([img_batch, ximg], dim=0)
    return img_batch

def curriculum_strategy_jigsaw(iter_num,bs=1, fre=1):
    if iter_num == 400 or iter_num == 800 or iter_num == 1600:
        bs = bs*2
    if iter_num < 1000:
        if iter_num % 200 == 0:
            fre += 1

    if iter_num >= 1000:
        if iter_num % 400 == 0:
            fre += 1

    return bs, fre


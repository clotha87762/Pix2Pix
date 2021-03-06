# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas
import os
import scipy.misc

def inverse_transform(images):
    return ((images+1.)/2.)

def save_imgs( imgs , size ,  path ): # save sample image in to a single large image
    out = merge(inverse_transform(imgs) , size)
    scipy.misc.imsave(path,out)
    #cv2.imwrite(path , out)

def save_img(img, path):
    scipy.misc.imsave(path,out)


def preprocess_img(img , load_size = 286 , crop_size = 256):
    pass

def preprocess_paired_img(img , load_size = 286 , crop_size = 256 , flip = True , isTest = False):
    
    if not isTest:
        w = int(img.shape[1])
        w2 = int(w/2)
        img_A = img[:, 0:w2].astype(np.float32)
        img_B = img[:, w2:w].astype(np.float32)
        
        img_A = scipy.misc.imresize(img_A, (load_size , load_size) )
        img_B = scipy.misc.imresize(img_B, (load_size , load_size) )
        
        res = load_size-crop_size if load_size > crop_size else 0
        img_A = img_A[int(res/2):int(res/2)+crop_size ,  int(res/2):int(res/2)+crop_size]
        img_B = img_B[int(res/2):int(res/2)+crop_size , int(res/2):int(res/2)+crop_size]
        
        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else :
        w = int(img.shape[1])
        w2 = int(w/2)
        img_A = img[:, 0:w2].astype(np.float32)
        img_B = img[:, w2:w].astype(np.float32)
        img_A = scipy.misc.imresize(img_A, (crop_size , crop_size) )
        img_B = scipy.misc.imresize(img_B, (crop_size , crop_size) )


    img_A = (img_A / 127.5) - 1.0
    img_B = (img_B / 127.5) - 1.0
    
    img = np.concatenate((img_A,img_B) , axis = -1)
    
    return img

def merge(images, size):
    
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def pair (img_1 , img_2): # merge in/out pair to a single image
    
    assert img_1.shape[0] == img_2.shape[0] and img_1.shape[2] == img_2.shape[2]
    
    img = np.zeros( (img_1.shape[0] , img_1.shape[1]+img_2.shape[1]) , 3)
    
    img[:,:img_1.shape[1],:] = img_1
    img[:,img_1.shape[1]:,:] = img_2
    
    return img



def load_all_data(batch_in_names, batch_out_names , load_size = 286 , crop_size = 256 , flip = True):
    pass



def load_all_data_pair(datas , load_size = 286 , crop_size = 256 , flip = True , isTest = False ): # facedes
    imgs = []
    for name in datas:
        #img = cv2.imread(name)
        img = scipy.misc.imread(name)
        img =  img.astype(np.float)
        imgs.append(preprocess_paired_img(img))
    
    return imgs

    

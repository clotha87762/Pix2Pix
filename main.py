# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os
from model import pix2pix





parser = argparse.ArgumentParser(description='')

parser.add_argument('--datasets', dest='dataset_name' , default = 'facades' ,help='name of dataset')
parser.add_argument('--lr',dest='init_lr',type=float,default=0.0002,help='learning rate of optimizaer')
parser.add_argument('--beta' , dest='beta',type=float,default=0.5)
parser.add_argument('--direction', dest = 'trans_dir' , default = 'AtoB') #or it can be BtoA
parser.add_argument('--flip',dest = 'flip',type=bool ,default = True)
parser.add_argument('--continue',dest='cont' , type = bool , default = True , help='whether load half-trained model, if exist')
parser.add_argument('--batch',dest='batch_size',type=int , default = 1 , help='batch_size for training')
parser.add_argument('--phase',dest='phase',default='train',help='train or test')
parser.add_argument('--print_freq', dest='print_freq', default=100)
parser.add_argument('--save_freq', dest='save_freq', type = int , default = 400)
parser.add_argument('--epoch',dest='epoch',type=int , default=200)
parser.add_argument('--load_size', dest='load_size' , type=int , default = 286 ,help='first,scale image to this size')
parser.add_argument('--crop_size', dest='crop_size', type=int , default= 256 , help='then crop to this size')
parser.add_argument('--model_dir', dest='model_path' , default='./checkpoint')
parser.add_argument('--log_dir', dest='log_path' , default='./logs')
parser.add_argument('--sample_dir', dest='sample_path' , default='./samples')
parser.add_argument('--test_dir', dest='test_path' , default='./test')
parser.add_argument('--mae_weight', dest='mae_weight',type=float, default=100.0,help='weight of mse loss')
parser.add_argument('--gan_weight', dest='gan_weight' , type = float, default = 1.0)
parser.add_argument('--in_dim', dest='in_dim' , type = int, default = 3)
parser.add_argument('--out_dim', dest='out_dim', type= int, default = 3)
parser.add_argument('--norm_method',dest='norm_method', default = 'bn' , help = 'bn/ins')
parser.add_argument('--en_time_unet',dest='en_time_unet',type=int, default = 8)
parser.add_argument('--en_time_resnet', dest='en_time_resnet', type=int, default = 2)
parser.add_argument('--deconv', dest='deconv', type=bool, default=True ,help = 'use deconvolution or resize-conv')
parser.add_argument('--ur',dest='ur',default='u' , help = 'u or r (unet or res net)')
parser.add_argument('--train_size', dest='train_size' , type = int ,default = 1e6 , help='maximum size of training set')
parser.add_argument('--paired', dest='paired' , type = bool ,default = True , help='whether the in/out is paired')
parser.add_argument('--patch_size' , dest='patch' , type = int , default = 32 , help = 'patch size of the Patch GAN')

args = parser.parse_args()

def main(_):
    
    if not os.path.exists(args.test_path):
        os.makedirs(args.test_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    
   
    with tf.Session() as sess:
                       
            model = pix2pix(sess, args)
            model.build()
         
            if args.phase == 'train':
                model.train()
            else :
                model.test()
    
        
if __name__ == "__main__" :
    tf.app.run()
         
    
    
    
    
    

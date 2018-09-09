# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from time import time
from glob import glob
import os
import loader
import sys
from module import *
import random


class pix2pix(object):
    
    def __init__(self, sess, args , gfdim = 64 , dfdim = 64):
        self.args = args
        self.sess = sess
        
        self.dataset = args.dataset_name
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.img_size = args.crop_size
        self.scale_size = args.load_size
        self.mae_weight = args.mae_weight
        self.gan_weight = args.gan_weight
        self.cont = args.cont
        self.model_path = args.model_path
        self.log_path = args.log_path
        self.sample_path = args.sample_path
        self.test_path = args.test_path
        self.lr = args.init_lr
        self.flip = args.flip
        
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        
        self.beta = args.beta
        
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        
        self.dir = args.trans_dir
        self.phase = args.phase
        
        self.gfdim = gfdim
        self.dfdim = dfdim
        
        self.out_size = args.load_size
        self.isTrain = True if self.phase =='train' else False
        self.norm_method = args.norm_method
        
        self.en_time_unet = args.en_time_unet
        self.en_time_resnet = args.en_time_resnet
        self.deconv = args.deconv
        
        self.ur = args.ur
        self.train_size = args.train_size
        
        self.is_gray = (self.in_dim == 1)
        
        self.paired = args.paired
        
        self.patch_size = args.patch
        
        self.is_build = False
        
        print(self.deconv)
        
        #self.build()
        
    def build(self):
        
        self.norm_layer = batch_norm if self.norm_method=='bn' else instance_norm
        
        self.real_pair = tf.placeholder(tf.float32, shape=[ self.batch_size , self.img_size, self.img_size, self.in_dim+self.out_dim ])
        
        if self.dir == 'AtoB':
            self.realA = self.real_pair[:,:,:, self.in_dim:self.in_dim+self.out_dim ]
            self.realB = self.real_pair[:,:,:, :self.in_dim]
        else :
            self.realB = self.real_pair[:,:,:, self.in_dim:self.in_dim+self.out_dim ]
            self.realA = self.real_pair[:,:,:, :self.in_dim]
        
        #print('a')
        
        if self.ur == 'u':
            self.generator = self.generator_unet
        else:
            self.generator = self.generator_resnet
        
        #print('b')
        self.fakeB =  self.generator(self.realA , reuse = False)
    
        
        self.fake_pair = tf.concat( [self.fakeB, self.realA ] , axis = -1)
        
        self.d_real , self.d_real_logits = self.discriminator(self.real_pair , patch_size = self.patch_size , reuse=False)
        self.d_fake , self.d_fake_logits = self.discriminator(self.fake_pair , patch_size = self.patch_size , reuse=True)
        
        self.d_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_real_logits , labels = tf.ones_like(self.d_real_logits) )  )
        self.d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_fake_logits , labels = tf.zeros_like(self.d_fake_logits) )  )
        
        self.d_loss = self.d_loss_real + self.d_loss_fake
        
        self.g_loss = self.mae_weight * tf.reduce_mean( tf.abs(self.realB - self.fakeB))  \
        + self.gan_weight * tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits= self.d_fake_logits , labels= tf.ones_like(self.d_fake) ) )
        
        g_sums = []
        d_sums  = []
        
        self.g_loss_sum = tf.summary.scalar('g_loss' , tf.reduce_mean(self.g_loss))
        self.d_loss_sum = tf.summary.scalar( 'd_loss' , tf.reduce_mean(self.d_loss))
        
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", tf.reduce_mean(self.d_loss_real))
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", tf.reduce_mean(self.d_loss_fake))
        
        self.fake_b_image = tf.summary.image('fake_image' , self.fakeB)
        self.d_real_sum = tf.summary.histogram( 'd real summary' , self.d_real_logits )
        self.d_fake_sum = tf.summary.histogram( 'd fake summary' , self.d_fake_logits )
        
        g_sums.append(self.g_loss_sum) 
        g_sums.append(self.d_loss_fake_sum)
        g_sums.append(self.d_fake_sum)
        g_sums.append(self.fake_b_image)
        
        d_sums.append(self.d_loss_sum)
        d_sums.append(self.d_real_sum)
        d_sums.append(self.d_fake_sum)
        d_sums.append(self.d_loss_real_sum)
        d_sums.append(self.d_loss_fake_sum)
        
        self.g_sums = tf.summary.merge(g_sums)
        self.d_sums = tf.summary.merge(d_sums)
        
        # or should we remove d_fake_sums here??
        
        self.saver = tf.train.Saver()
        
        tf_vars = tf.trainable_variables()
        
        self.g_vars = [ var for var in tf_vars if 'generator' in var.name ]
        self.d_vars = [ var for var in tf_vars if 'discriminator' in var.name]
        
        self.is_build = True
        
    
        
    def generator_unet(self , input_img , reuse =False):
        
        assert self.en_time_unet > 1
        
        #print('YOOOOO')
        
        with tf.variable_scope('generator'):
            
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else :
                assert tf.get_variable_scope().reuse == False
            
            encs = []
            decs = []
            
            fdim = self.gfdim
            
            t = self.en_time_unet
            sl =[int(self.img_size/2) , int(self.img_size/4) , int(self.img_size/8) , int(self.img_size/16), int(self.img_size/32) , int(self.img_size/64), int(self.img_size/128), int(self.img_size/256)]
            
            
            enc = conv2d(input_img, fdim, name = 'g_e0_conv')
            encs.append(enc)
            
            enc = self.norm_layer(conv2d( lrelu(enc), fdim*2 , name = 'g_e1_conv') , is_train = self.isTrain, name = 'g_e1_bn')
            encs.append(enc)
            enc = self.norm_layer(conv2d( lrelu(enc), fdim*4 , name = 'g_e2_conv') , is_train = self.isTrain, name = 'g_e2_bn')
            encs.append(enc)
            enc = self.norm_layer(conv2d( lrelu(enc), fdim*8 , name = 'g_e3_conv') , is_train = self.isTrain, name = 'g_e3_bn')
            encs.append(enc)
            enc = self.norm_layer(conv2d( lrelu(enc), fdim*8 , name = 'g_e4_conv') , is_train = self.isTrain, name = 'g_e4_bn')
            encs.append(enc)
            enc = self.norm_layer(conv2d( lrelu(enc), fdim*8 , name = 'g_e5_conv') , is_train = self.isTrain, name = 'g_e5_bn')
            encs.append(enc)
            enc = self.norm_layer(conv2d( lrelu(enc), fdim*8 , name = 'g_e6_conv') , is_train = self.isTrain, name = 'g_e6_bn')
            encs.append(enc)
            enc = self.norm_layer(conv2d( lrelu(enc), fdim*8 , name = 'g_e7_conv') , is_train = self.isTrain, name = 'g_e7_bn')
            encs.append(enc)
                
            dec = enc
            
            if self.deconv:
                
                batch_size = tf.shape(dec)[0]
                fdim = self.dfdim
                
                dec = deconv2d(relu(dec), [ batch_size, sl[6], sl[6], fdim*8] , name = 'g_d0_conv')
                dec = self.norm_layer( dropout(dec)  , is_train = self.isTrain, name = 'g_d0_bn')
                decs.append(dec)
                dec = tf.concat( [dec , encs[6]] , axis = -1 )
            
                dec = deconv2d(relu(dec), [ batch_size , sl[5], sl[5], fdim*8] , name = 'g_d1_conv')
                dec = self.norm_layer( dropout(dec), is_train = self.isTrain, name = 'g_d1_bn')
                decs.append(dec)
                dec = tf.concat( [dec , encs[5]] , axis= -1 )
                
                dec = deconv2d(relu(dec), [ batch_size, sl[4], sl[4], fdim*8] , name = 'g_d2_conv')
                dec = self.norm_layer( dropout(dec)  , is_train = self.isTrain, name = 'g_d2_bn')
                decs.append(dec)
                dec = tf.concat( [dec , encs[4]] , axis = -1 )
                
                # without dropout
                
                dec = self.norm_layer(deconv2d(relu(dec), [ batch_size, sl[3], sl[3], fdim*8] , name = 'g_d3_conv') , is_train = self.isTrain, name = 'g_d3_bn')
                decs.append(dec)
                dec = tf.concat( [dec , encs[3]] , axis = -1 )
                
                dec = self.norm_layer(deconv2d(relu(dec), [ batch_size, sl[2], sl[2], fdim*4] , name = 'g_d4_conv') , is_train = self.isTrain, name = 'g_d4_bn')
                decs.append(dec)
                dec = tf.concat( [dec , encs[2]] , axis = -1 )
                
                dec = self.norm_layer(deconv2d(relu(dec), [ batch_size, sl[1], sl[1], fdim*2] , name = 'g_d5_conv') , is_train = self.isTrain, name = 'g_d5_bn')
                decs.append(dec)
                dec = tf.concat( [dec , encs[1]] , axis= -1 )
                
                dec = self.norm_layer( deconv2d(relu(dec), [ batch_size, sl[0], sl[0], fdim*1] , name = 'g_d6_conv') , is_train = self.isTrain, name = 'g_d6_bn')
                decs.append(dec)
                dec = tf.concat( [dec , encs[0]] , axis = -1 )
                
                dec = deconv2d( relu(dec), [batch_size, self.img_size, self.img_size, self.out_dim], name = 'g_d7_conv') 
                decs.append(dec)
               
            else:
                dec = self.norm_layer(deconv2d_resize(relu(dec),fdim*8 , name = 'g_d0_conv') , is_train = self.isTrain, name = 'g_d0_bn')
                dec = dropout(dec)
                decs.append(dec)
                dec = tf.concat( [dec , encs[6]] , axis = -1 )
            
                dec = self.norm_layer(deconv2d_resize(relu(dec),fdim *8 , name = 'g_d1_conv') , is_train = self.isTrain, name = 'g_d1_bn')
                dec = dropout(dec)
                decs.append(dec)
                dec = tf.concat( [dec , encs[5]] , axis = -1 )
                
                dec = self.norm_layer(deconv2d_resize(relu(dec),fdim *8, name = 'g_d2_conv') , is_train = self.isTrain, name = 'g_d2_bn')
                dec = dropout(dec)
                decs.append(dec)
                dec = tf.concat( [dec , encs[4]] , axis = -1 )
                
                # without dropout
                
                dec = self.norm_layer(deconv2d_resize(relu(dec),fdim *8, name = 'g_d3_conv') , is_train = self.isTrain, name = 'g_d3_bn')
                decs.append(dec)
                dec = tf.concat( [dec , encs[3]] , axis = -1 )
                
                dec = self.norm_layer(deconv2d_resize(relu(dec),fdim  *4, name = 'g_d4_conv') , is_train = self.isTrain, name = 'g_d4_bn')
                decs.append(dec)
                dec = tf.concat( [dec , encs[2]] , axis= -1 )
                
                dec = self.norm_layer(deconv2d_resize(relu(dec),fdim  *2, name = 'g_d5_conv') , is_train = self.isTrain, name = 'g_d5_bn')
                decs.append(dec)
                dec = tf.concat( [dec , encs[1]] , axis = -1 )
                
                dec = self.norm_layer(deconv2d_resize(relu(dec),fdim *1 , name = 'g_d6_conv') , is_train = self.isTrain, name = 'g_d6_bn')
                decs.append(dec)
                dec = tf.concat( [dec , encs[0]] , axis = -1 )
                
                dec = conv2d( relu(dec), self.out_dim , name = 'g_d7_conv') 
                decs.append(dec)
                
            return tf.nn.tanh(dec)
          
        
    def generator_resnet(self , input_img , reuse = False ):
        with tf.variable_sope('generator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else :
                assert tf.get_variable_scope().reuse == False
            
            batch_size = tf.shape(input_img)[0]
            
            sl =[self.img_size , int(self.img_size/2) , int(self.img_size/4) , int(self.img_size/8) , int(self.img_size/16), int(self.img_size/32) , int(self.img_size/64)]
            
            # Follow Justin Johnson's implementation
            c0 = tf.pad(input_img, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            c1 = tf.nn.relu( self.norm_layer(conv2d(c0, self.gfdim, kernel = 7, stride=(1,1), padding='VALID' , name='g_e1_c') , is_train = self.isTrain, name = 'g_e1_bn'))
            c2 = tf.nn.relu( self.norm_layer(conv2d(c1, self.gfdim*2, kernel=3, stride=(2,2), name='g_e2_c'),  is_train = self.isTrain, name = 'g_e2_bn'))
            c3 = tf.nn.relu( self.norm_layer(conv2d(c2, self.gfdim*4, kernel = 3, stride = (2,2), name='g_e3_c'),  is_train = self.isTrain, name = 'g_e3_bn'))
            
            r0 = res_block2(c3 , self.gfdim * 4 , stride=(1,1) , name = 'g_r0')
            r1 = res_block2(r0 , self.gfdim * 4 , stride=(1,1) , name = 'g_r1')
            r2 = res_block2(r1 , self.gfdim * 4 , stride=(1,1) , name = 'g_r2')
            r3 = res_block2(r2 , self.gfdim * 4 , stride=(1,1) , name = 'g_r3')
            r4 = res_block2(r3 , self.gfdim * 4 , stride=(1,1) , name = 'g_r4')
            r5 = res_block2(r4 , self.gfdim * 4 , stride=(1,1) , name = 'g_r5')
            r6 = res_block2(r5 , self.gfdim * 4 , stride=(1,1) , name = 'g_r6')
            r7 = res_block2(r6 , self.gfdim * 4 , stride=(1,1) , name = 'g_r7')
            r8 = res_block2(r7 , self.gfdim * 4 , stride=(1,1) , name = 'g_r8')
            
            if self.deconv:
                d0 =  deconv2d( (r8), [ batch_size, sl[1], sl[1], self.gfdim*2] , name = 'g_d0_conv') 
                d0 =  relu( self.norm_layer(  (dec), is_train = self.isTrain, name = 'g_d0_bn') )
                d1 =  deconv2d( (d0), [ batch_size, sl[0], sl[0], self.gfdim] , name = 'g_d1_conv') 
                d1 =  relu( self.norm_layer(  (dec), is_train = self.isTrain, name = 'g_d1_bn') )
                pred = tf.nn.tanh( conv2d(d1, self.out_dim , kernel = 7 , stride=(1,1), padding='VALID', name='g_pred_c'))
                
            else:
                d0 =  deconv2d_resize( (r8), self.gfdim * 2 , name = 'g_d0_conv') 
                d0 =  relu( self.norm_layer(  (dec), is_train = self.isTrain, name = 'g_d0_bn') )
                d1 =  deconv2d( (d0),  self.gfdim , name = 'g_d1_conv') 
                d1 =  relu( self.norm_layer(  (dec), is_train = self.isTrain, name = 'g_d1_bn') )
                pred = tf.nn.tanh( conv2d(d1, self.out_dim , kernel = 7 , stride=(1,1), padding='VALID', name='g_pred_c'))
                
            
            return pred
        
    
    def discriminator(self , input_img , patch_size = 32 ,reuse = False  ):
        
        with tf.variable_scope('discriminator'):
            
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else :
                assert tf.get_variable_scope().reuse == False
            
            init_size = self.img_size
            
            scale_down_time = np.log2(init_size/patch_size) + 1.0 if init_size > patch_size else 0
            
            i = 0.0
            
            
            while i < scale_down_time :
                
                if i < 0.01:
                    dis = conv2d(input_img , self.dfdim, name = 'd_conv_0')
                else:
                    ft = 2**(int(i)) if i < 4.0 else 8
                    strides = (2,2) if i<scale_down_time-1.0 else (1,1)
                    dis = self.norm_layer( conv2d( lrelu(dis), self.dfdim * ft , stride = strides ,name= 'd_conv'+str(int(i))) , is_train = self.isTrain, name = 'd_bn' + str(int(i) ))
                    print(self.dfdim*ft)
                    print(strides)
                i = i + 1.0
            
            dis = lrelu(dis)
            out = dense(tf.reshape(dis, [self.batch_size, -1]), 1, name = 'd_lin')
            #out = conv2d( lrelu(dis) , 1 ,stride=(1,1) , name = 'd_conv_predict')
            #print(out.get_shape())
            return tf.nn.sigmoid(out) , out
        
    def load_model(self):
        
        print('loading model...')
        dir_name = '%s_b%s_o%s' % (self.dataset,self.batch_size,self.out_size)
        path = os.path.join(self.model_path,dir_name)
        ckpt = tf.train.get_checkpoint_state(path)
        #print(ckpt)
        #print(path)
        if ckpt and ckpt.model_checkpoint_path:
            
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print('restoring from...')
            print(os.path.join( path , ckpt_name))
            self.saver.restore(self.sess, os.path.join( path , ckpt_name))
            return True
        else:
            return False
                            
        #print('model loading finished')
        
    def save_model(self , counter):
        
        dir_name = '%s_b%s_o%s' % (self.dataset, self.batch_size,self.out_size)
        checkpoint_dir = os.path.join(self.model_path , dir_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        file_path = os.path.join(checkpoint_dir ,'pix2pix.model')
        self.saver.save(self.sess, file_path , global_step = counter)
        #self.saver.save(self.sess , file_path )
        
    def train(self): # load all data once
        
        assert self.is_build
    
        d_opt = tf.train.AdamOptimizer(learning_rate = self.lr , beta1= self.beta)\
        .minimize(loss = self.d_loss , var_list=self.d_vars)
        g_opt = tf.train.AdamOptimizer(learning_rate = self.lr , beta1 = self.beta )\
        .minimize(loss = self.g_loss , var_list=self.g_vars)
        
        
        self.global_step = tf.Variable( 0  ,name='global_step'  , trainable = False)
        
        self.global_step = tf.assign_add(self.global_step , 1)
       
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.writer = tf.summary.FileWriter( "./logs", self.sess.graph)
        
        
        if self.paired:
            train_names = glob('./datasets/{}/train/*.jpg'.format(self.dataset))
            val_names = glob('./datasets/{}/val/*.jpg'.format(self.dataset))
            total_size = len(train_names)
        else :
            train_in_names = glob('./datasets/{}/train/in/*.jpg'.format(self.dataset))
            train_out_names = glob('./datasets/{}/train/out/*.jpg'.format(self.dataset))
            val_in_names = glob('./datasets/{}/val/in/*.jpg'.format(self.dataset))
            val_out_names = glob('./datasets/{}/val/out/*.jpg'.format(self.dataset))
            total_size = len(train_in_names)
    
        img_data = []
        #img_data = loader.load_all_data_pair(train_names , load_size = self.load_size , crop_size = self.crop_size)
        
        #self.img_data = img_data
        
        
        if self.cont == True and self.load_model():
            print('load model success!')
        else:
            print('start from a new model')
        
        start_time = time()
        
        for i in range(self.epoch):
            
            if self.paired:
                
                #random.shuffle(train_names)
                pass
            else :
                sample = np.random.choice( range(len(train_in_names)) , len(train_in_names) ,replace = False).tolist()
                train_in_names = [ train_in_names[i] for i in sample]
                train_out_names = [ train_out_names[i] for i in sample]
            
            for j in range( int( np.ceil(total_size//self.batch_size))  ):
                
                #print(str(i)+" "+str(j))
                
                if self.paired:
                    batch_names = train_names[j*self.batch_size:(j+1)*self.batch_size] if not j==(np.ceil(total_size//self.batch_size)-1) \
                    else train_names[j*self.batch_size:]
                    
                    input_imgs = loader.load_all_data_pair(batch_names , load_size = self.scale_size , crop_size = self.img_size , flip = self.flip)

                else:
                    batch_in_names = train_in_names[j*self.batch_size:(j+1)*self.batch_size] if not j==(np.ceil(total_size//self.batch_size)-1)\
                    else train_in_names[j*self.batch_size:]
                    
                    batch_out_names = train_out_names[j*self.batch_size:(j+1)*self.batch_size] if not j==(np.ceil(total_size//self.batch_size)-1)\
                    else train_out_names[j*self.batch_size:]
                    
                    input_imgs = loader.load_all_data(batch_in_names, batch_out_names , load_size = self.scale_size , crop_size = self.img_size , flip = self.flip)

                
                img_array = (np.array(input_imgs)[:,:,:,None]) if self.is_gray \
                else (np.array(input_imgs))
                
                
                
                #print(img_array.shape())
                 
                # optimize discriminator
                _ , sum_d_info  = self.sess.run( [d_opt , self.d_sums ] , feed_dict = {self.real_pair:img_array})
                
                # optimize generator
                _ , sum_g_info = self.sess.run( [g_opt , self.g_sums ] , feed_dict={self.real_pair:img_array})
                
                
                
                # optimize generator
                _ , sum_g_info = self.sess.run( [g_opt , self.g_sums ] , feed_dict={self.real_pair:img_array})
                
                gloss = self.g_loss.eval({self.real_pair: img_array})
                dlossreal = self.d_loss_real.eval({self.real_pair: img_array})
                dlossfake = self.d_loss_fake.eval({self.real_pair: img_array})
                
                step = self.sess.run( self.global_step )
                
                #print(step)
                
                self.writer.add_summary(sum_g_info ,  step )
                self.writer.add_summary(sum_d_info ,  step )
                
                
                
                #self.global_step = tf.add(step,1)
                
                
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (i, j ,( (total_size//self.batch_size) + 1), time() - start_time, np.mean(dlossfake) + np.mean(dlossreal), np.mean(gloss)))
                
                if np.mod(step , self.print_freq ) == 1:
                    if self.paired:
                        self.sample_current_generator( self.sample_path, i , j , val_names)
                    else:
                        self.sample_current_generator( self.sample_path, i, j, val_in_names , val_out_names)
                
                if np.mod(step , self.save_freq ) == 2:
                    self.save_model(step)
                    
    
    
    
    def sample_current_generator(self, path , epoch , idx, in_names , out_names = None):
        
        sample = np.random.choice( range(len(in_names)) , self.batch_size ,replace = False).tolist()
        
        if out_names is None:
            names = [ in_names[i] for i in sample ]
            input_img = loader.load_all_data_pair(names , load_size = self.scale_size , crop_size = self.img_size , flip = self.flip)
        else:
            ins = [in_names[i] for i in sample]
            outs = [out_names[i] for i in sample]
            input_img =  loader.load_all_data(ins, outs , load_size = self.scale_size , crop_size = self.img_size , flip = self.flip)
            
        img_array = np.array(input_img)[:,:,:,None] if self.is_gray \
        else np.array(input_img)
        
        fake_imgs , g_loss , d_loss = self.sess.run( [self.fakeB,self.g_loss,self.d_loss] , feed_dict={ self.real_pair: img_array })
        
        loader.save_imgs(fake_imgs,[self.batch_size , 1] , './{}/test_{:03d}_{:04d}.png'.format(path ,epoch, idx))
        
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(np.mean(d_loss), np.mean( g_loss)))
    
    def test(self):
        
        assert self.is_build
        
        self.global_step = tf.Variable( 0  ,name='global_step'  , trainable = False)
        
        if self.paired:
            test_names = glob('./datasets/{}/test/*.jpg'.format(self.dataset))
            n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], test_names)]
            test_names = [x for (y, x) in sorted(zip(n, test_names))]
            input_imgs = loader.load_all_data_pair(test_names , load_size = self.scale_size, crop_size=self.img_size , flip = False)
        else:
            test_in_names = glob('./datasets/{}/test/in/*.jpg'.format(self.dataset))
            test_out_names = glob('./datasets/{}/test/out/*.jpg'.format(self.dataset))
            n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], test_in_names)]
            test_in_names = [x for (y, x) in sorted(zip(n, test_in_names))]
            n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], test_out_names)]
            test_out_names = [x for (y, x) in sorted(zip(n, test_out_names))]
            input_imgs = loader.load_all_data(test_in_names, test_out_names , load_size = self.scale_size, crop_size=self.img_size , flip = False)
        
        input_imgs = [input_imgs[i:i+self.batch_size] for i in range(0, len(input_imgs), self.batch_size)]
        
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
       
        
        if self.load_model():
            print('load model success!')
        else:
            print('there should be a model weight when testing!!')
            sys.exit()
        
        for i , sample in enumerate(input_imgs):
            
            if self.is_gray:
                sample = np.array(sample)[:, :, :, None]
            else:
                sample = np.array(sample)
            
                
            output = self.sess.run( self.fakeB , feed_dict = {self.real_pair : sample})
            
            loader.save_imgs(output , [self.batch_size , 1] , './{}/test_{:04d}.png'.format(self.test_path , i))
        
    
            
        
        
        
        
        
        
    
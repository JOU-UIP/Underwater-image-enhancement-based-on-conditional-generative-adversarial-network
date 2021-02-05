from utils import ( 
  imsave,
  prepare_data
)

import time
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow as tf
import scipy.io as scio
from ops import *
from ssim import *
class T_CNN(object):

  def __init__(self, 
               sess, 
               image_height=128,
               image_width=128,
               label_height=128, 
               label_width=128,
               batch_size=1,
               c_dim=3, 
               c_depth_dim=1,
               checkpoint_dir=None, 
               sample_dir=None,
               test_image_name = None,
               #test_depth_name = None,
               id = None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5
    self.test_image_name = test_image_name
    #self.test_depth_name = test_depth_name
    self.id = id
    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.c_depth_dim=c_depth_dim
    self.new_height=0
    self.new_width=0
    self.new_height_half=0 
    self.new_width_half=0
    self.new_height_half_half=0
    self.new_width_half_half=0  

    image_test =  get_image(self.test_image_name,is_grayscale=False)
    shape = image_test.shape
    RGB=Image.fromarray(np.uint8(image_test*255))
    #RGB1=RGB.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
    image_test = np.asarray(np.float32(RGB)/255)
    shape = image_test.shape



    self.new_height=shape[0]
    self.new_width=shape[1]
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')

    self.pred_h = self.model()



    self.saver = tf.train.Saver()
     
  def train(self, config):


    # Stochastic gradient descent with the standard backpropagation,var_list=self.model_c_vars
    image_test =  get_image(self.test_image_name,is_grayscale=False)
    shape = image_test.shape
    RGB=Image.fromarray(np.uint8(image_test*255))
    #RGB1=RGB.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
    image_test = np.asarray(np.float32(RGB)/255)
    shape = image_test.shape

    expand_test = image_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_image = np.append(expand_test,expand_zero,axis = 0)


    tf.global_variables_initializer().run()
    
    
    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
    start_time = time.time()
    result_h  = self.sess.run(self.pred_h, feed_dict={self.images: batch_test_image})
    all_time = time.time()
    final_time=all_time - start_time
    print(final_time)    


    _,h ,w , c = result_h.shape
    for id in range(0,1):

        result_h0 = result_h[id,:,:,:].reshape(h , w , 3)


        image_path0 = os.path.join(os.getcwd(), config.sample_dir)
        image_path = os.path.join(image_path0, self.test_image_name+'_out.png')
        imsave_lable(result_h0, image_path)


  def model(self):


    with tf.variable_scope("fusion_branch") as scope:
      if self.id > 0: 
        scope.reuse_variables()
#common features
      conv2_cb1_1 = tf.nn.relu(conv2d(self.images, 3,16,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_1"))
      conv2_cb1_2 = tf.nn.relu(conv2d(conv2_cb1_1, 16,32,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_2"))
      conv2_cb1_3 = tf.nn.relu(conv2d(conv2_cb1_2, 32,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_3"))
      conv2_cb1_4 = tf.nn.relu(conv2d(conv2_cb1_3, 64,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_4"))
      conv2_cb1_5 = tf.nn.relu(conv2d(conv2_cb1_4, 128,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_5"))
#
      conv2_cb1_6 = tf.nn.relu(conv2d(self.images, 3,16,k_h=5, k_w=5, d_h=1, d_w=1,name="conv2_cb1_6"))
      conv2_cb1_7 = tf.nn.relu(conv2d(conv2_cb1_6, 16,32,k_h=5, k_w=5, d_h=1, d_w=1,name="conv2_cb1_7"))
      conv2_cb1_8 = tf.nn.relu(conv2d(conv2_cb1_7, 32,64,k_h=5, k_w=5, d_h=1, d_w=1,name="conv2_cb1_8"))
      conv2_cb1_9 = tf.nn.relu(conv2d(conv2_cb1_8, 64,128,k_h=5, k_w=5, d_h=1, d_w=1,name="conv2_cb1_9"))
      conv2_cb1_10 = tf.nn.relu(conv2d(conv2_cb1_9, 128,256,k_h=5, k_w=5, d_h=1, d_w=1,name="conv2_cb1_10"))
#      
      conv2_cb1_11 = tf.nn.relu(conv2d(self.images, 3,16,k_h=7, k_w=7, d_h=1, d_w=1,name="conv2_cb1_11"))
      conv2_cb1_12 = tf.nn.relu(conv2d(conv2_cb1_11, 16,32,k_h=7, k_w=7, d_h=1, d_w=1,name="conv2_cb1_12"))
      conv2_cb1_13 = tf.nn.relu(conv2d(conv2_cb1_12, 32,64,k_h=7, k_w=7, d_h=1, d_w=1,name="conv2_cb1_13"))
      conv2_cb1_14 = tf.nn.relu(conv2d(conv2_cb1_13, 64,128,k_h=7, k_w=7, d_h=1, d_w=1,name="conv2_cb1_14"))
      conv2_cb1_15 = tf.nn.relu(conv2d(conv2_cb1_14, 128,256,k_h=7, k_w=7, d_h=1, d_w=1,name="conv2_cb1_15"))
#     
      concate_multiscale = tf.concat(axis = 3, values = [conv2_cb1_5,conv2_cb1_10,conv2_cb1_15])  
      

# color mask
#
      pool4_cb1=max_pool_2x2(concate_multiscale)
      conv4_cb1_1 = tf.nn.relu(conv2d(pool4_cb1, 512,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv4_cb1_1"))
      conv4_cb1_2 = tf.nn.relu(conv2d(conv4_cb1_1, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv4_cb1_2"))
      conv4_cb1_3 = tf.nn.relu(conv2d(conv4_cb1_2, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv4_cb1_3"))
#
      pool4_cb2=max_pool_2x2(conv4_cb1_3)
      conv4_cb2_1 = tf.nn.relu(conv2d(pool4_cb2, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv4_cb2_1"))
      conv4_cb2_2 = tf.nn.relu(conv2d(conv4_cb2_1, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv4_cb2_2"))
      conv4_cb2_3 = tf.nn.relu(conv2d(conv4_cb2_2, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv4_cb2_3"))
# mask
      color_down = tf.nn.relu(conv2d(conv4_cb2_3, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="color_down"))
      color=tf.image.resize_bilinear(color_down,[self.image_height, self.image_width])

      conv5_cb3_5 = tf.nn.relu(conv2d(color, 256,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv5_cb3_5"))
      conv5_cb3_6 = tf.nn.relu(conv2d(conv5_cb3_5, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv5_cb3_6"))
      conv5_cb3_7 = tf.nn.relu(conv2d(conv5_cb3_6, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv5_cb3_7"))
      conv5_cb3_8 = conv2d(conv5_cb3_7, 64,3,k_h=3, k_w=3, d_h=1, d_w=1,name="conv5_cb3_8")  
      final_results=tf.add(self.images,conv5_cb3_8)
      
    return final_results



  def discriminator(self, image,  reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image,1 ,self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim,self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*2,self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*4,self.df_dim*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

      return tf.nn.sigmoid(h4), h4  


  def save(self, checkpoint_dir, step):
    model_name = "coarse.model"
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

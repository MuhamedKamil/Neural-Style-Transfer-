import tensorflow as tf
import numpy as np 
import matplotlib as cm 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os 
import cv2 
from PIL import Image
from tensorflow.keras import models 
import time
import tensorflow.contrib.eager as tfe
import IPython.display
import math


#Data
#--------------------------------------------------------------------------------------------
content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1','block2_conv1','block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
Channels = 3
IMG_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, Channels)
#--------------------------------------------------------------------------------------------

def get_model():
  vgg = tf.keras.applications.vgg19.VGG19(include_top = False, weights='imagenet')
  vgg.trainable = False
  style_outputs =   [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs
  return models.Model(vgg.input , model_outputs)
#--------------------------------------------------------------------------------------------
def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)

  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]
  x = np.clip(x, 0, 255).astype('uint8')
  return x
#--------------------------------------------------------------------------------------------
def ReadImage (path):
    return plt.imread (path)
#--------------------------------------------------------------------------------------------
def ResizeImage (Img , h, w):
    return cv2.resize (Img , (h,w))
#--------------------------------------------------------------------------------------------    
def generate_noise_image(content_image, noise_ratio , IMAGE_H, IMAGE_W, COLOR_CHANNEL):
    noise_image = np.random.randn(IMAGE_H, IMAGE_W, COLOR_CHANNEL)
    gen_image = noise_image * noise_ratio + content_image * (1.0 - noise_ratio)
    return gen_image
#--------------------------------------------------------------------------------------------
def Compute_Cost_ContentImg (OutOfContet , OutOfGenrated):
  m , n_H , n_W , n_C = OutOfContet.get_shape().as_list()
  return tf.reduce_mean(tf.square (OutOfContet - OutOfGenrated))/(4*n_H*n_W*n_C)
#--------------------------------------------------------------------------------------------
def gram_matrix(OutOfStyle):
    m , H , W , C = OutOfStyle.get_shape().as_list()
    OutOfStyle_R = tf.reshape (OutOfStyle ,[W*H ,C])   
    StyleMatrix =tf.matmul(tf.transpose(OutOfStyle_R) ,OutOfStyle_R) 
    return StyleMatrix

def Compute_Cost_StyleImg_Per_one_Layer (OutOfStyle , OutOfGenrated):
    m , n_H , n_W , n_C = OutOfStyle.get_shape().as_list()
    gram_Style_img = gram_matrix(OutOfStyle)
    gram_Genrated_img = gram_matrix(OutOfGenrated)
    return tf.reduce_mean(tf.square(gram_Style_img-gram_Genrated_img))/ (4 * n_C**2 * (n_W * n_H)**2)

def ComputeTotalLoss(model,  style_weight, content_weight, init_image, InputVGG19_Style_Img, InputVGG19_Content_Img):
  style_weight = style_weight
  content_weight = content_weight

  style_outputs =   model(InputVGG19_Style_Img)       
  content_outputs = model(InputVGG19_Content_Img)
  model_outputs = model(init_image) # 6 layers
    

  OutVGG19_Style_Img = [style_layer for style_layer in style_outputs[:num_style_layers]] 
  OutVGG19_Content_Img = [content_layer for content_layer in content_outputs[num_style_layers:]] 
  
  style_output_features =   model_outputs[:5] 
  content_output_features = model_outputs[5:] 

  style_score = 0
  content_score = 0
  
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(OutVGG19_Content_Img, content_output_features):
    content_score += weight_per_content_layer* Compute_Cost_ContentImg(target_content,comb_content)
  
  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip (OutVGG19_Style_Img, style_output_features):
    style_score += weight_per_style_layer * Compute_Cost_StyleImg_Per_one_Layer(target_style, comb_style)
  
  loss = content_score*content_weight + style_score*style_weight

  return loss, style_score, content_score

def View_Images (content_image , style_image , generated_image , Rows = 1 , Columns = 3):
    fig=plt.figure(figsize=(9, 9))
    fig.add_subplot(Rows, Columns , 1)
    plt.imshow(content_image)
    plt.title("Content Image")

    fig.add_subplot(Rows, Columns, 2)
    plt.imshow(style_image)
    plt.title("Style Image")

    fig.add_subplot(Rows, Columns, 3) 
    plt.imshow(generated_image)
    plt.title("Generated Image")

    plt.show()
    
def Preprocessing_input_Vgg19 (image):
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    img_arr = np.expand_dims(img_arr, axis = 0)
    img_arr = tf.keras.applications.vgg19.preprocess_input(img_arr)
    return img_arr

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = ComputeTotalLoss(**cfg)
  # Compute gradients wrt input image
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss



def Training_Neural_Style_Transfer(InputVGG19_Style_Img ,InputVGG19_Content_Img , init_image ,num_iterations=500,content_weight=10, style_weight=40 ): 
  
  model = get_model() 
  for layer in model.layers:
    layer.trainable = False
  
  init_image = tfe.Variable(InputVGG19_Content_Img, dtype=tf.float32)
  opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
  iter_count = 1
  best_loss, best_img = float('inf'), None
  
  cfg = {
      'model': model,
      'style_weight': style_weight,
      'content_weight':content_weight,
      'init_image': init_image,
      'InputVGG19_Style_Img': InputVGG19_Style_Img,
      'InputVGG19_Content_Img': InputVGG19_Content_Img
  }
  num_rows = 2
  num_cols = 5
  display_interval = num_iterations/(num_rows*num_cols)

  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   
  
  for i in range(num_iterations):
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    
    if loss < best_loss:
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())

    if i % display_interval== 0:
      print('Iteration: {}'.format(i))        
      print('Total loss: {:.4e}, ' 
            'style loss: {:.4e}, '
            'content loss: {:.4e}, '.format(loss, style_score, content_score))
  return best_img, best_loss

#------------------------------------------------------------------------------------
#             Main function
######################################################################################
tf.enable_eager_execution()
tfe = tf.contrib.eager
cont_img = ReadImage("D:/Projects/Style Transfer CNN/content.jpg")
style_img =ReadImage("D:/Projects/Style Transfer CNN/style.jpg")

style_img =ResizeImage (style_img ,IMAGE_HEIGHT,IMAGE_WIDTH)
cont_img = ResizeImage (cont_img ,IMAGE_HEIGHT,IMAGE_WIDTH)
gen_img = generate_noise_image (cont_img ,0.999,IMAGE_WIDTH,IMAGE_HEIGHT,Channels)


InputVGG19_Content_Img = Preprocessing_input_Vgg19(cont_img)
InputVGG19_Style_Img = Preprocessing_input_Vgg19(style_img)
InputVGG19_Generated_Img = Preprocessing_input_Vgg19(gen_img)
#---------------------------------------------------------------------------------------
bestimg , best_loss =  Training_Neural_Style_Transfer(InputVGG19_Style_Img ,InputVGG19_Content_Img , InputVGG19_Generated_Img )
Image.fromarray(bestimg)
View_Images (cont_img , style_img ,bestimg , Rows = 1 , Columns = 3)

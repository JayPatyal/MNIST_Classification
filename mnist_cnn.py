import numpy as np
import numpy as asarray
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

y_train

class Conv_op:
  def __init__(self,num_filters, filter_size):
    self.num_filters=num_filters
    self.filter_size=filter_size
    self.conv_filter=np.random.randn(num_filters,filter_size,filter_size)/(filter_size * filter_size)
  
  def image_region(self, image):
    height, width = image.shape
    self.image = image
    for j in range(height - self.filter_size+1):
      for k in range(width- self.filter_size + 1):
        image_patch=image[j:(j+self.filter_size), k:(k+self.filter_size)]  
        yield image_patch, j, k #striding action please refer Part c for dimensions as the image is fed into the convolution layers. We get image patches after convolutions
  def forward_prop(self, image):
    height, width = image.shape
    conv_out= np.zeros((height - self.filter_size+1,width-self.filter_size + 1, self.num_filters))
    for image_patch, i , j in self.image_region(image):
      conv_out[i,j]= np.sum(image_patch*self.conv_filter, axis=(1,2))
    return conv_out
  def back_prop(self, dL_dout, learning_rate):
    dL_dF_params = np.zeros(self.conv_filter.shape) #initialising it to 0 first
    for image_patch, i , j in self.image_region(self.image):
      for k in range(self.num_filters):
        dL_dF_params[k]+=image_patch*dL_dout[i,j,k] #refer Eqn 9 here image patches are the convolved image blocks as the the filter takes the stride
    self.conv_filter-=learning_rate*dL_dF_params
    return dL_dF_params


class Max_Pool:
  def __init__(self,filter_size):
    self.filter_size=filter_size
  def image_region(self,image):
    new_height=image.shape[0]//self.filter_size
    new_width=image.shape[1]//self.filter_size
    self.image=image

    for i in range(new_height):
      for j in range(new_width):
        image_patch= image[(i*self.filter_size): (i*self.filter_size + self.filter_size), (j*self.filter_size):(j*self.filter_size + self.filter_size)]
        yield image_patch, i, j #Similar to convolution where we store image patches after convolving our image with a filter.
  def forward_prop(self, image):
    height,width, num_filters=image.shape
    output=np.zeros((height // self.filter_size, width // self.filter_size, num_filters))

    for image_patch, i , j in self.image_region(image):
      output[i,j]=np.amax(image_patch, axis=(0,1))

    return output

  def back_prop(self, dL_dout):
    dL_dmax_pool = np.zeros(self.image.shape)
    for image_patch, i, j in self.image_region(self.image):
      height, width, num_filters=image_patch.shape
      maximum_val = np.amax(image_patch, axis=(0,1))
      for i1 in range(height):
        for j1 in range(width):
          for k1 in range(num_filters):
            if image_patch[i1, j1, k1]== maximum_val[k1]: 
              dL_dmax_pool[i*self.filter_size + i1, j*self.filter_size + j1,k1]=dL_dout[i,j,k1] #check Eqn 7 in the above image.Only gives output if value is max.
      return dL_dmax_pool

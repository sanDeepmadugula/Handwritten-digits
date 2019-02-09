#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.utils import np_utils


# In[3]:


import os
os.chdir('D:\\python using jupyter\\Handwritten recognization')


# In[4]:


test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')


# In[5]:


train.head(5)


# In[6]:


test.head(5)


# in test dataset there is no labels

# In[7]:


print(test[:1].shape)
print(train[:1].shape)


# In[10]:


X_train = train.iloc[:,1:].values
y_train = train.iloc[:,0].values
X_test = test.values
print('Size of training data:{}'.format(X_train.shape))
print('Size of testing data:{}'.format(X_test.shape))
print("Size of a single entry in X_train {}".format(X_train[:1].shape))


# In[12]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255


# In[13]:


y_train = np_utils.to_categorical(y_train)
num_classes = y_train.shape[1]


# In[14]:


print(num_classes)


# In[15]:


print(y_train)


# In[16]:


img_width = 28
img_height = 28
img_depth = 1

plt.figure(figsize=(12,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i].reshape(28,28),cmap='gray',interpolation=None)
    plt.title('Label{}'.format(np.where(y_train[i]==1)))


# In[17]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[18]:


seed = 7
np.random.seed(seed)


# In[27]:


def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal',activation='relu'))
    model.add(Dense(num_classes, input_dim=num_pixels, kernel_initializer='normal',activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[28]:


num_pixels = X_train.shape[1]


# In[29]:


# build the model
model = baseline_model()
model.fit(X_train,y_train,epochs=10,batch_size=200,verbose=2)


# In[30]:


y_train[1:10,:]


# In[31]:


from keras.models import load_model
model.save('baseline.h5')
model = load_model('baseline.h5')
results = model.predict_classes(X_test)


# In[32]:


results = pd.Series(results,name='Label')


# In[33]:


submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),results],axis=1)
submission.to_csv('baseline_model.csv',index=False)


# Lets try with Convolutional Neural Network

# In[34]:


from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as k
k.set_image_dim_ordering('th')


# In[35]:


X_train = X_train.reshape(X_train.shape[0],1,28,28)
X_test = X_test.reshape(X_test.shape[0],1,28,28)

print('Size of training data{}'.format(X_train.shape))
print('Size of testing data{}'.format(X_test.shape))


# In[36]:


def cnn_model():
    model = Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(1,28,28),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu')) # passing 128 neurons with relu activation function
    model.add(Dense(num_classes,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


# In[37]:


model = cnn_model()
model.summary()
X_train.reshape((-1,1))
model.fit(X_train,y_train,epochs=3,batch_size=50,verbose=2)


# In[40]:


# now to predict the model's accuracy on the test data set
from keras.models import load_model
model.save('model.h5')
model=load_model('model.h5')
labels=model.predict_classes(X_test)


# In[42]:


labels = pd.Series(labels,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),labels],axis = 1)

submission.to_csv("cnn_model.csv",index=False)
# print(check_output(["ls", "."]).decode("utf8"))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[ ]:


import numpy as np
import tensorflow as tf

from data_gen import DataGenerator
from model import MyCL_Model


# ### Paths

# In[ ]:


videos_path = './Videos_MERL_Shopping_Dataset/'


# In[ ]:


x_train_path = videos_path+'train/'
y_train_path = 'train_y.pkl'


# In[ ]:


x_val_path = videos_path + '/val/'
y_val_path = 'val_y.pkl'


# ### Create Train and Validation Data Generator objects

# In[ ]:


train_data = DataGenerator(x_train_path ,y_path = y_train_path)
val_data = DataGenerator(x_val_path ,y_path = y_val_path)


# ### Define and Compile Model

# In[ ]:


model = MyCL_Model()
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])


# ### Train and Evaluate Model

# In[ ]:


epochs = 20
for i in range(epochs):

    for j in range(len(train_data)):
        model.fit_generator(generator = train_data[j])
    val_loss = 0.0
    val_acc = 0.0
    for k in range(len(val_data)):
        l, a = model.evaluate_generator(generator = val_data[k])
        val_loss += l
        val_acc += a
    val_loss /= k
    val_acc /=k
    
    model.save('mycl.h5')
    print("Epoch: ", i, ", Validation Loss: ",  val_loss, ", Validation Per-Frame Accuracy: ", val_acc)


# ### Testing Data Paths

# In[ ]:


x_test_path = videos_path + 'test/'
y_test_path = 'test_y.pkl'


# ### Create Test Data Generator Object

# In[ ]:


test_data = DataGenerator(x_test_path ,y_path = y_test_path)


# ### Test the model

# In[ ]:


test_acc = 0.0
for i in range(len(test_data)):
    _, a = model.evaluate_generator(testing_generator)
    test_acc += a
test_acc/=i
print("Per Frame Accuracy for Test Data = ", test_acc)


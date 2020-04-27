#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('600519历史数据.csv')


# In[3]:


df.head()


# In[4]:


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


# In[5]:


TRAN_SPLIT=1500


# In[6]:


uni_data=df['收盘'][-1::-1]
uni_data.index=df['日期'][-1::-1]
uni_data.tail()


# In[7]:


uni_data=pd.DataFrame(uni_data)


# In[8]:


uni_data['收盘']=uni_data['收盘'].apply(lambda x : x.replace(',',''))


# In[9]:


uni_data['收盘']=uni_data['收盘'].astype('float')


# In[10]:


uni_data.plot(subplots=True)


# In[11]:


uni_data=uni_data.values


# In[12]:


uni_train_mean=uni_data[:TRAN_SPLIT].mean()
uni_train_std=uni_data[:TRAN_SPLIT].std()


# In[13]:


uni_data=(uni_data-uni_train_mean)/uni_train_std


# In[14]:


uni_data


# In[15]:


univariate_past_history=20
univariate_future_target=0
x_train_uni,y_train_uni=univariate_data(uni_data,0,TRAN_SPLIT,
                                        univariate_past_history,
                                        univariate_future_target)
x_val_uni,y_val_uni=univariate_data(uni_data,TRAN_SPLIT,None,
                                    univariate_past_history,
                                    univariate_future_target)


# In[18]:


def create_time_steps(length):
    return list(range(-length, 0))


# In[19]:


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


# In[20]:


show_plot([x_train_uni[0],y_train_uni[0]],0,'SE')


# In[21]:


def baseline(history):
    return np.mean(history)


# In[22]:


show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
           'Baseline Prediction Example')


# In[23]:


BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()


# In[24]:


simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')


# In[25]:


for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)


# In[26]:


EVALUATION_INTERVAL = 200
EPOCHS = 10
simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)


# In[27]:


for x, y in val_univariate.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                    simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()


# In[ ]:





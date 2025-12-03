import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array

from setting import *

def calc_fading():
    theta = np.random.rand(L) * 2 * np.pi
    phi = np.random.rand(L) * 2 * np.pi
    
    fading_data_list=[]
    x=np.zeros(L)
    for _ in range(DATA_NUM):
        x+=np.random.uniform(-DELTA_D,DELTA_D,L)
        fading_data = np.sum(R * np.exp(1j * (theta + (2 * np.pi / LAMBDA_0) * x * np.cos(phi))))
        fading_data /= np.sqrt(L)
        fading_data_list.append(fading_data)
 
    return np.array(fading_data_list)
    
def calc_nakagami_rice_fading(k_rice=K_RICE):
    theta0 = np.random.rand() * 2 * np.pi
    scattered_data_list=calc_fading()
    scattered_data_list = scattered_data_list / np.sqrt(np.mean(np.abs(scattered_data_list)**2))*np.sqrt(1 / (k_rice + 1))
    direct_data_list=[]
    x=0
    for _ in range(DATA_NUM):
        x+=np.random.uniform(-DELTA_D,DELTA_D)
        direct_data = np.sqrt(k_rice / (k_rice + 1.0)) * np.exp(1j * ((2 * np.pi / LAMBDA_0) * x + theta0))
        direct_data_list.append(direct_data)
    direct_data_list=np.array(direct_data_list)
    
    nakagami_rice_data_list=scattered_data_list+direct_data_list

    return nakagami_rice_data_list
        
### シミュレーション用のデータセット(入力と答え)の配列を用意する関数
def generate_fading_dataset(input_len,data_set_num=DATA_SET_NUM):
    
    fading_data_list_list=[]
    for _ in range(data_set_num):
        fading_data_list_list.append(calc_nakagami_rice_fading())
    fading_data_list_list=np.array(fading_data_list_list)

    power = np.abs(fading_data_list_list) ** 2
    #power_db = 10 * np.log10(power)
    #data_normalized = normalize(power_db)
    #data_normalized=power_db
    data_normalized=power
    dataset_arr=[]
    for i in range(data_set_num):
        targets = data_normalized[i][input_len:] 

        #datasetの中身はtensorflow特有のオブジェクトで入力と出力(入力に対する答え)のセットが入っている    
        dataset_i=timeseries_dataset_from_array(
            data=data_normalized[i],
            targets=targets,
            sequence_length=input_len,
            batch_size=None,
            shuffle=None
        )
        dataset_arr.append(dataset_i)
    # dataset_arrの中身をすべてつなげる
    dataset = dataset_arr[0]
    for ds in dataset_arr[1:]:
        dataset = dataset.concatenate(ds)
    return dataset

### シミュレーションデータセットを訓練用、検証用と用意する関数
def load_fading_data(batch_size,input_len):
    train_dataset = generate_fading_dataset(input_len)
    train_dataset = (
        train_dataset
        .shuffle(buffer_size=10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = generate_fading_dataset(input_len,DATA_SET_NUM//4)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset,val_dataset

### RNNで使えるようにデータ配列を入力と答えに分離する関数(再帰予測する際にしか使いません)
def make_data_set(changed_data,input_len):
    data,target=[],[]
    
    for i in range(len(changed_data)-input_len):
        data.append(changed_data[i:i + input_len])
        target.append(changed_data[i + input_len])

    # RNN用に3次元のデータに変更する
    re_data = np.array(data).reshape(len(data), input_len, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return (re_data, re_target)
    
### フェージングデータをプロットする用のコード ###
'''
#dataset=calc_nakagami_rice_fading()
dataset=calc_fading()
plt.figure()
#plt.scatter(range(NUM),dataset,color="r",alpha=0.5,label="fading_data")
#plt.scatter(range(NUM),dataset,color="r",alpha=0.5,label="fading_data",s=1)
plt.plot(range(DATA_NUM),dataset,color="r",label="fading_data")
plt.legend()
plt.show()

'''
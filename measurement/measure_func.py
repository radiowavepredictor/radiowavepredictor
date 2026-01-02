import numpy as np
import os
path = os.path.dirname(__file__)
import pandas as pd
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array

from common.common_func import normalize

#### ここから↓結構分かりづらいかも ごめんなさい ####
def csv_to_dataset(csv_path,input_len,in_features,out_features,start_cut_index,end_cut_index): 
    csv_data = pd.read_csv(csv_path, usecols=in_features)
    csv_data = csv_data.iloc[int(len(csv_data)*start_cut_index):int(len(csv_data) * end_cut_index)]

    data_normalized = normalize(csv_data)

    data = data_normalized.values.astype(np.float64)
    targets = data_normalized[out_features] 
    targets = targets[input_len:] 
    targets = targets.values.astype(np.float64)
    
    #datasetの中身はtensorflow特有のオブジェクトで入力と出力(入力に対する答え)のセットが入っている    
    dataset=timeseries_dataset_from_array(
        data=data,
        targets=targets,
        sequence_length=input_len,
        batch_size=None,
        shuffle=None
    )
    return dataset 

def multiple_csv_to_dataset(read_cources,input_len,learn_mode,in_features,out_features,start_cut_index,end_cut_index):
    dataset_arr=[]
    for cource in read_cources:
        csv_path = f"{path}/result/WAVE{cource:04d}/result_n{learn_mode}-001.csv" 
        train_dataset_i=csv_to_dataset(csv_path,input_len,in_features,out_features,start_cut_index,end_cut_index)

        dataset_arr.append(train_dataset_i)

    # dataset_arrの中身をすべてつなげる
    dataset = dataset_arr[0]
    for ds in dataset_arr[1:]:
        dataset = dataset.concatenate(ds)
    return dataset

#csvからデータを読み込んで機械学習の訓練データセットと検証データセットを返す関数
def load_training_data(training_cources,validation_cources,learn_mode,batch_size,input_len,in_features,out_features,start_cut_index,end_cut_index):
    train_dataset=multiple_csv_to_dataset(training_cources,input_len,learn_mode,in_features,out_features,start_cut_index,end_cut_index)
    train_dataset = (
        train_dataset
        .shuffle(buffer_size=10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    val_dataset =multiple_csv_to_dataset(validation_cources,input_len,learn_mode,in_features,out_features,start_cut_index,end_cut_index)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset,val_dataset

#dataからRNN用のデータセットを生成する関数
#timeseries_dataset_from_array()と使い分ける
#make_data_setはnumpy配列を返すので、後から加工しやすい
#timeseries_dataset_from_arrayはtf.data.Datasetオブジェクトを返すので、prefetchなどtensorflow専用の関数が使える
def make_data_set(changed_data,input_len):
    data,target=[],[]
    
    for i in range(len(changed_data)-input_len):
        data.append(changed_data[i:i + input_len])
        target.append(changed_data[i + input_len])

    # RNN用に3次元のデータに変更する
    re_data = np.array(data).reshape(len(data), input_len, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return (re_data, re_target)

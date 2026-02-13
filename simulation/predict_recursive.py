### 再帰予測用に書いていたコード 多分そのままじゃ動かないので参考程度に ###

#(x,y)=make_data_set(normalized_data,INPUT_LEN)
'''
##### ここから再帰予測 #####
input_data=normalized_data[-PREDICT_LEN-INPUT_LEN:-PREDICT_LEN]
future_result = np.empty((0,)) # (0,)で空の配列になる

for i in range(PREDICT_LEN):
    print(f'step:{i+1}')
    test_data=np.reshape(input_data,(1,INPUT_LEN,1))
    predicted=model.predict(test_data)

    input_data=np.delete(input_data,0)
    input_data=np.append(input_data,predicted)

    future_result=np.append(future_result,predicted)
#true_data = np.array(data_csv[OUT_FEATURES]).reshape(len(data_csv[OUT_FEATURES]))
#true_data=power_db
true_data=power
#denormalized_predicted = denormalize(predicted,true_data)
denormalized_predicted = future_result
denormalized_predicted = 10 * np.log10(denormalized_predicted)
reshape_denormalied_predeicted = np.array(denormalized_predicted).reshape(len(denormalized_predicted))
true_data = 10 * np.log10(true_data)
mse=np.mean((future_result - true_data[-PREDICT_LEN:])**2)
rmse=np.sqrt(np.mean((future_result - true_data[-PREDICT_LEN:])**2))
print(rmse)

'''

#plt.plot(range(len(true_data)-PREDICT_LEN, len(true_data)), future_result, color="g", label="future_predict") #再帰用
#plt.plot(range(len(true_data)-PLOT_RANGE-PREDICT_LEN,len(true_data)),true_data[-PLOT_RANGE-PREDICT_LEN:],color="r",alpha=0.5,label="true_data") #再帰用

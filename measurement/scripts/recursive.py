'''
    ##### ここから再帰予測 #####
    input_data=normalized_true_data[-PREDICT_LEN-INPUT_LEN:-PREDICT_LEN]
    future_result = np.empty((0,)) # (0,)で空の配列になる

    for i in range(PREDICT_LEN):
        print(f'step:{i+1}')
        test_data=np.reshape(input_data,(1,INPUT_LEN,1))
        predicted=model.predict(test_data)
        predicted =  predicted * true_data.std() + true_data.mean()

        input_data=np.delete(input_data,0)
        input_data=np.append(input_data,predicted)

        future_result=np.append(future_resul,predicted)
    mse=np.mean((future_result - true_data[-PREDICT_LEN:])**2)
    rmse=np.sqrt(np.mean((future_result - true_data[-PREDICT_LEN:])**2))
    print(f"2乗誤差:{rmse}")
    '''


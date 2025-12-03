from keras_tuner.tuners import Hyperband
from keras.callbacks import EarlyStopping

from setting import *
from func import load_training_data
from hyperparam_tuning.build_model import RNNHyperModel

# プログラムの実行はルートディレクトリにあるhyperparam.pyから行ってください
def layer_unit_tuning():
    train_dataset,validation_dataset=load_training_data(TRAINING_COURCES,VALIDATION_COURCES,LEARN_MODE,BATCH_SIZE,INPUT_LEN)

    tuner = Hyperband(
        RNNHyperModel(),
        objective='val_loss',
        max_epochs=EPOCHS,
        directory=f'{path}/hyperparam_tuning',
        project_name='tuner_result'
    )

    tuner.search(
        train_dataset,
        validation_data=validation_dataset,
        callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience=20)],
    )

    tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters()[0]
    print(f"一番良かったパラメータ:{best_hps.values}")

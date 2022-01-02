import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from model import get_model

scan_types    = ['FLAIR','T1w','T1wCE','T2w']

def train_model():
    for scan_type in scan_types:
        # load train_dataset dataset
        tf_data_path = f'./datasets/{scan_type}_train_dataset'
        with open(tf_data_path + '/element_spec', 'rb') as in_:
            es = pickle.load(in_)
        train_dataset = tf.data.experimental.load(tf_data_path, es, compression='GZIP')
        
        # load validation_dataset
        tf_data_path = f'./datasets/{scan_type}_validation_dataset'
        with open(tf_data_path + '/element_spec', 'rb') as in_:
            es = pickle.load(in_)
        validation_dataset = tf.data.experimental.load(tf_data_path, es, compression='GZIP')

        # Get Model
        model = get_model(width=128, height=128, depth=64,name=scan_type)
        
        # Define callbacks.
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            f'{scan_type}_3d_image_classification.h5', save_best_only=True
        )
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

        epochs = 100
        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            shuffle=True,
            verbose=2,
            callbacks=[checkpoint_cb, early_stopping_cb],
        )
        
        #save model
        model.save(f'./models/{scan_type}')
        
        # show metrics
        fig, ax = plt.subplots(1, 2, figsize=(20, 3))
        ax = ax.ravel()

        for i, metric in enumerate(["acc", "loss"]):
            ax[i].plot(model.history.history[metric])
            ax[i].plot(model.history.history["val_" + metric])
            ax[i].set_title("{} Model {}".format(scan_type, metric))
            ax[i].set_xlabel("epochs")
            ax[i].set_ylabel(metric)
            ax[i].legend(["train", "val"])
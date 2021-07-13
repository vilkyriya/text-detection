from __future__ import print_function

import os
import tensorflow as tf
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import *
from keras.metrics import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 384
img_cols = 384

# img_rows_orig = 480
# img_cols_orig = 640

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss,
                  metrics=[dice_coef, tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.Precision(name='precision')])

    model.summary()

    return model


def preprocess_RGB(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 3), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols, 3), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def f_metric(precision, recall, b=1):
    if type(precision) == float:
        if precision == 0. and recall == 0.:
            return 0.
        else:
            return (1 + b ** 2) * precision * recall / (b ** 2 * precision + recall)
    else:
        isNull = False
        for i, j in zip(precision, recall):
            if i == 0 and j == 0:
                isNull = True

        if isNull is False:
            f_metric = []
            for i in range(len(precision)):
                f_metric.append((1 + b ** 2) * precision[i] * recall[i] / (b ** 2 * precision[i] + recall[i]))
            return f_metric
        else:
            f_metric = []
            for i in range(len(precision)):
                f_metric.append(0)
            return f_metric


def plot_accuracy(history, cur_dir):
    fig = plt.figure()
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    fig.savefig(os.path.join(cur_dir, 'model_accuracy.png'), dpi=fig.dpi)


def plot_PRF_metrics(history, train_f_metric, val_f_metric, cur_dir):
    fig = plt.figure()
    plt.plot(history.history['precision'], color='lightsteelblue')
    plt.plot(history.history['recall'], color='royalblue')
    plt.plot(train_f_metric, color='navy', linestyle="--")
    plt.plot(history.history['val_precision'], color='navajowhite')
    plt.plot(history.history['val_recall'], color='gold')
    plt.plot(val_f_metric, color='darkorange', linestyle="--")
    plt.title('Precision, Recall and F metrics')
    plt.ylabel('Metrics')
    plt.xlabel('Epoch')
    plt.legend(['Train Precision', 'Train Recall', 'Train F',
                'Val Precision', 'Val Recall', 'Val F'], loc='lower right')
    fig.savefig(os.path.join(cur_dir, 'PRF_metrics.png'), dpi=fig.dpi)


def train_and_predict():
    cur_dir = str(datetime.datetime.now()).replace(":", "-")[:16]
    os.mkdir(str(cur_dir))

    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train, imgs_id_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    imgs_train /= 255.

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    # from sklearn.model_selection import train_test_split
    # imgs_train, imgs_test, imgs_mask_train, imgs_mask_test, \
    # id_train, id_test = train_test_split(imgs_train, imgs_mask_train, imgs_id, test_size=0.1, random_state=1)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()
    with open(os.path.join(cur_dir, 'model_summary.txt'), 'w') as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(os.path.join(cur_dir, 'weights.h5'))  

    model_checkpoint = ModelCheckpoint(os.path.join(cur_dir, 'weights.h5'), monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger(os.path.join(cur_dir, 'train_history.csv'), separator=';', append=True)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=1, verbose=1, shuffle=True,
                        validation_split=0.15,
                        callbacks=[model_checkpoint, csv_logger])

    print('-' * 30)
    print('Saving metric images...')
    print('-' * 30)

    plot_accuracy(history, cur_dir)

    train_f_metric = f_metric(history.history['precision'], history.history['recall'])
    val_f_metric = f_metric(history.history['val_precision'], history.history['val_recall'])

    plot_PRF_metrics(history, train_f_metric, val_f_metric, cur_dir)

    hist_df = pd.DataFrame(history.history)
    f_metrics = pd.DataFrame(data={'f_metric': train_f_metric, 'val_f_metric': val_f_metric})
    data = pd.merge(hist_df, f_metrics, left_index=True, right_index=True)
    data.to_csv(os.path.join(cur_dir, 'train_history.csv'), sep=';')

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test, imgs_mask_test, imgs_id_test = load_test_data()

    imgs_test = preprocess(imgs_test)
    imgs_mask_test = preprocess(imgs_mask_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test /= 255.

    imgs_mask_test = imgs_mask_test.astype('float32')
    imgs_mask_test /= 255.  # scale masks to [0, 1]

    print('-' * 30)
    print("Evaluate on test data...")
    print('-' * 30)
    results = model.evaluate(imgs_test, imgs_mask_test, batch_size=16)

    results.append(f_metric(results[4], results[3]))
    test_metrics = pd.DataFrame(data={'test_loss': results[0], 'test_dice_coef': results[1], 'test_auc': results[2],
                                      'test_recall': results[3], 'test_precision': results[4],
                                      'test_f_metric': [results[5]]})
    test_metrics.to_csv(os.path.join(cur_dir, 'test_history.csv'), sep=';')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    pred_imgs_mask_test = model.predict(imgs_test, verbose=1)
    # np.save(os.path.join(cur_dir, 'pred_imgs_mask_test_p_and_w.npy'), pred_imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    os.mkdir(os.path.join(cur_dir, pred_dir))
    for image, id_image in zip(pred_imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        image = resize(image, (img_rows, img_cols), preserve_range=True)
        imsave(os.path.join(cur_dir, pred_dir, str(id_image) + '_pred.png'), image)


if __name__ == '__main__':
    train_and_predict()

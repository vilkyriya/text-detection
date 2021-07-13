from __future__ import print_function

import os
import numpy as np
import cv2

from math import sqrt, exp

from skimage.io import imsave, imread

data_path = 'raw/'

image_rows_orig = 480
image_cols_orig = 640

img_rows = 384
img_cols = 384
dist = ""
n = ""


def img_preprocess(img):
    normal_img = np.uint8((img - np.min(img)) / (np.max(img) - np.min(img)) * 255)
    blur_img = cv2.bilateralFilter(normal_img, 5, 50, 50)

    return blur_img


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    print('Train before: ' + str(len(images) // 2))
    total = len(images) // 2 * 5
    print('Train after: ' + str(total))

    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=np.int32)

    flag = 1
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)

    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)
        img_id = int(flag)

        img *= 255
        img = img_preprocess(img)
        img_mask *= 255

        imgs[5 * (img_id - 1)] = cv2.resize(img, (img_rows, img_cols))
        imgs_mask[5 * (img_id - 1)] = cv2.resize(img_mask, (img_rows, img_cols))
        imgs_id[5 * (img_id - 1)] = 5 * (img_id - 1)

        for i in range(0, 2):
            for j in range(0, 2):
                sm_img = np.array(img[i * (image_rows_orig - img_rows):i * image_rows_orig + (1 - i) * img_rows,
                                  j * (image_cols_orig - img_cols):j * image_cols_orig + (1 - j) * img_cols])
                sm_img_mask = np.array(
                    img_mask[i * (image_rows_orig - img_rows):i * image_rows_orig + (1 - i) * img_rows,
                    j * (image_cols_orig - img_cols):j * image_cols_orig + (1 - j) * img_cols])

                imgs[4 * (img_id - 1) + i * 2 + j + img_id] = sm_img

                imgs_mask[4 * (img_id - 1) + i * 2 + j + img_id] = sm_img_mask
                imgs_id[4 * (img_id - 1) + i * 2 + j + img_id] = 4 * (img_id - 1) + i * 2 + j + img_id

        if flag * 5 % 200 == 0:
            print('Done: {0}/{1} images'.format(flag * 5, total))
        flag += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    np.save('imgs_id_train.npy', imgs_id)
    print('Saving to ' + str(dist) + str(n) + ' .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    imgs_id = np.load('imgs_id_train.npy')

    print('imgs_train.npy')
    print('imgs_mask_train.npy')
    print('imgs_id_train.npy')
    return imgs_train, imgs_mask_train, imgs_id


def create_test_data():
    test_data_path = os.path.join(data_path, 'test')
    images = os.listdir(test_data_path)
    print('Test before: ' + str(len(images) // 2))
    total = len(images) // 2 * 5
    print('Test after: ' + str(total))

    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=np.int32)

    flag = 1
    print('-' * 30)
    print('Creating test images...')
    print('-' * 30)

    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        img = imread(os.path.join(test_data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(test_data_path, image_mask_name), as_gray=True)
        img_id = int(flag)

        img *= 255
        img = img_preprocess(img)
        img_mask *= 255

        imgs[5 * (img_id - 1)] = cv2.resize(img, (img_rows, img_cols))
        imgs_mask[5 * (img_id - 1)] = cv2.resize(img_mask, (img_rows, img_cols))
        imgs_id[5 * (img_id - 1)] = 5 * (img_id - 1)

        for i in range(0, 2):
            for j in range(0, 2):
                sm_img = np.array(img[i * (image_rows_orig - img_rows):i * image_rows_orig + (1 - i) * img_rows,
                                  j * (image_cols_orig - img_cols):j * image_cols_orig + (1 - j) * img_cols])
                sm_img_mask = np.array(
                    img_mask[i * (image_rows_orig - img_rows):i * image_rows_orig + (1 - i) * img_rows,
                    j * (image_cols_orig - img_cols):j * image_cols_orig + (1 - j) * img_cols])

                imgs[4 * (img_id - 1) + i * 2 + j + img_id] = sm_img

                imgs_id[4 * (img_id - 1) + i * 2 + j + img_id] = 4 * (img_id - 1) + i * 2 + j + img_id

        if flag * 5 % 200 == 0:
            print('Done: {0}/{1} images'.format(flag * 5, total))
        flag += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_mask_test.npy', imgs_mask)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_mask = np.load('imgs_mask_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_mask, imgs_id


if __name__ == '__main__':
    create_test_data()
    create_train_data()


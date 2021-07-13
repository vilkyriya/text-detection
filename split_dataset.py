import os
import random
import shutil

from skimage.io import imsave, imread

data_path = 'raw/'

train_data_path = os.path.join(data_path, 'old_train')
new_train_data_path = os.path.join(data_path, 'train')

new_test_data_path = os.path.join(data_path, 'test')

print(train_data_path)
images = os.listdir(train_data_path)
total = len(images) // 2

test_size = int(total * 0.1)

randomlist = random.sample(range(1, total+1), test_size)
randomlist.sort()
print(randomlist)


with open('raw/train_test_split.txt', 'w') as f:
    for item in randomlist:
        f.write("%s\n" % item)

for image_name in images:
    if 'mask' in image_name:
        continue
    image_mask_name = image_name.split('.')[0] + '_mask.png'

    if int(image_name.split('.')[0]) in randomlist:
        original = os.path.join(train_data_path, image_name)
        target = os.path.join(new_test_data_path, image_name)
        shutil.copyfile(original, target)

        original_mask = os.path.join(train_data_path, image_mask_name)
        target_mask = os.path.join(new_test_data_path, image_mask_name)
        shutil.copyfile(original_mask, target_mask)

    else:
        original = os.path.join(train_data_path, image_name)
        target = os.path.join(new_train_data_path, image_name)
        shutil.copyfile(original, target)

        original_mask = os.path.join(train_data_path, image_mask_name)
        target_mask = os.path.join(new_train_data_path, image_mask_name)
        shutil.copyfile(original_mask, target_mask)

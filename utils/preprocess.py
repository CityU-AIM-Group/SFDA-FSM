import os
import random
import numpy as np
from PIL import Image

def revise_label():
    dir = '/home/cyang/SFDA/data/EndoScene/labels'
    dir_list = os.listdir(dir)
    for i in dir_list:
        label = Image.open(os.path.join(dir, i)).convert('L')
        label = np.asarray(label, np.uint8)
        label_copy = label.copy()
        label_copy[np.where(label > 128)] = 255
        label_copy[np.where(label <= 128)] = 0
        label_copy = Image.fromarray(label_copy)
        label_copy.save(os.path.join(dir, i))

    label = Image.open('/home/cyang/SFDA/data/EndoScene/labels/2.png').convert('L')
    label = np.asarray(label, np.uint8)
    for i in range(256):
        print(np.where(label == i))


def split_dataset(dir):
    dir_list = os.listdir(dir)
    random.shuffle(dir_list)
    train_file = open(os.path.join(dir[:-7].replace('data', 'dataset') + '_list', 'train.lst'), 'w')
    test_file = open(os.path.join(dir[:-7].replace('data', 'dataset') + '_list', 'test.lst'), 'w')
    for i in range(len(dir_list)):
        if i < 4/5 * len(dir_list):
            train_file.write(dir_list[i])
            train_file.write('\n')
        else:
            test_file.write(dir_list[i])
            test_file.write('\n')
    train_file.close()
    test_file.close()
    
if __name__ == '__main__':
    split_dataset('/home/cyang/SFDA/data/ETIS-Larib/images')
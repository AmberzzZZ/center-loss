import numpy as np
import glob
import cv2
import random
from keras.utils import to_categorical


def load_mnist():

    data_pt = "/Users/amber/dataset/mnist"
    x_train = []
    y_train = []
    for sub_folder in glob.glob(data_pt+"/*"):
        num = int(sub_folder.split("/")[-1][-1])
        if num not in [0,1,2]:
            continue
        for file in glob.glob(sub_folder + "/*")[:10000]:
            img = cv2.imread(file, 0)
            x_train.append(img)
            y_train.append(num)
    idx = [i for i in range(len(y_train))]
    random.shuffle(idx)
    x_train = np.array(x_train)[idx[:30000]]
    y_train = np.array(y_train)[idx[:30000]]

    return x_train, y_train


if __name__ == '__main__':

    # data
    x_train, y_train = load_mnist()
    y_train_onehot = to_categorical(y_train)
    print(x_train.shape)
    print(y_train.shape)
    print(y_train_onehot.shape)





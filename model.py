from keras.datasets import mnist
from keras.layers import Input,Conv2D,Activation,Dense,Lambda,Flatten,Embedding,PReLU,BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.utils import to_categorical
from keras.optimizers import SGD
import numpy as np
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import plot_model


def center_loss_model():

    # branch 1
    inputs = Input((28,28,1))

    x = Conv2D(32,(3,3))(inputs)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(32,(3,3))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(64,(5,5))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(64,(5,5))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(128,(7,7))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(128,(7,7))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Flatten()(x)
    x = Dense(2)(x)
    out1 = PReLU(name="out1")(x)                     # 2 dimension for coord represention
    out2 = Dense(3,activation="softmax")(out1)      # 10 dimension for classification

    # branch 2
    lambda_c = 1
    input_ = Input(shape=(1,))
    centers = Embedding(3,2)(input_)    # (None, 1, 2)
    intra_loss = Lambda(lambda x:K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True))([out1,centers])

    # multi-input, multi-output model
    model_center_loss = Model([inputs,input_],[out2,intra_loss])

    model_center_loss.compile(optimizer=SGD(lr=5e-3, momentum=0.9, decay=0.01, nesterov=True),
                              loss=["categorical_crossentropy",lambda y_true,y_pred:y_pred],
                              loss_weights=[1,lambda_c/2.],
                              metrics=["acc"])

    # plot_model(model_center_loss, to_file='model.png', show_shapes=True, show_layer_names=True)

    return model_center_loss


def load_mnist():
    import glob
    import cv2
    import random
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
    x_train = np.expand_dims(x_train,axis=-1)       # 28,28,1
    dummy_matrix = np.zeros((x_train.shape[0],1))

    # model
    model_center_loss = center_loss_model()

    # train
    filepath = "./centerloss_weights_{epoch:02d}_val_acc_{val_dense_2_acc:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, monitor="val_dense_2_loss", mode='min', save_best_only=True)
    model_center_loss.fit(x=[x_train, y_train], 
                          y=[y_train_onehot, dummy_matrix], 
                          batch_size=512, epochs=100, verbose=1, 
                          validation_split=0.2, 
                          callbacks=[checkpoint, EarlyStopping(monitor="val_dense_2_loss",patience=20)])




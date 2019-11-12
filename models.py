from keras.layers import Input,Conv2D,Activation,Dense,Lambda,Flatten,\
                         Embedding,PReLU,BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.utils import to_categorical
from keras.optimizers import SGD
import numpy as np
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import plot_model
from customLayer import CenterLossLayer
from dataLoader import load_mnist


def raw_cls_model():

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
    out2 = Dense(3,activation="softmax")(out1)      # 10 dimension for classification   kernel_regularizer=l2(0.0005)

    model = Model(inputs, out2)

    # plot_model(model, to_file='images/raw_cls_model.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer=SGD(lr=3e-3, momentum=0.9, decay=0.01, nesterov=True),
                  loss="categorical_crossentropy", metrics=["acc"])

    return model


def center_loss_model_embedding():

    # branch 1
    basic_model = raw_cls_model()
    out1 = basic_model.get_layer(name="out1").output   # dense2 output

    # branch 2
    lambda_c = 1
    input_ = Input(shape=(1,))        # raw GT label in [0,1,2,...,9]
    centers = Embedding(3,2)(input_)    # (None, 1, 2)
    intra_loss = Lambda(lambda x:K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True))([out1,centers])

    # multi-input, multi-output model
    model = Model([basic_model.input,input_],[basic_model.output,intra_loss])

    # plot_model(model, to_file='images/center_loss_model_embedding.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer=SGD(lr=5e-3, momentum=0.9, decay=0.01, nesterov=True),
                  loss=["categorical_crossentropy",lambda y_true,y_pred:y_pred],
                  loss_weights=[1,lambda_c/2.], metrics=["acc"])

    return model


def center_loss_model_custom():

    # branch 1
    basic_model = raw_cls_model()
    out1 = basic_model.get_layer(name="out1").output

    # branch 2
    input_ = Input(shape=(3,))      # one-hot GT label
    intra_loss = CenterLossLayer(alpha=0.5, name='centerlosslayer')([out1, input_])

    model = Model([basic_model.input,input_],[basic_model.output,intra_loss])

    # plot_model(model, to_file='images/center_loss_model_custom.png', show_shapes=True, show_layer_names=True)

    lambda_c = 1
    model.compile(optimizer=SGD(lr=3e-4, momentum=0.9, decay=0.01, nesterov=True),
                  loss=["categorical_crossentropy",lambda y_true,y_pred:y_pred],
                  loss_weights=[1,lambda_c/2.], metrics=["acc"])

    return model


if __name__ == '__main__':

    # data
    x_train, y_train = load_mnist()
    y_train_onehot = to_categorical(y_train)
    x_train = np.expand_dims(x_train,axis=-1)       # 28,28,1
    dummy_matrix = np.zeros((x_train.shape[0],1))


    # train raw cls model
    # model = raw_cls_model()
    # filepath = "./rawmodel_weights_{epoch:02d}_val_acc_{val_acc:.3f}.h5"
    # checkpoint = ModelCheckpoint(filepath, verbose=1, monitor="val_loss", mode='min', save_best_only=True)
    # model.fit(x_train, y_train_onehot,
    #           batch_size=512, epochs=10, verbose=1, 
    #           validation_split=0.2, 
    #           callbacks=[checkpoint, EarlyStopping(monitor="val_loss",patience=20)])


    # train embedding model
    # model = center_loss_model_embedding()
    # filepath = "./embedding_weights_{epoch:02d}_val_acc_{val_dense_2_acc:.3f}.h5"
    # checkpoint = ModelCheckpoint(filepath, verbose=1, monitor="val_dense_2_loss", mode='min', save_best_only=True)
    # model.fit(x=[x_train, y_train], 
    #                       y=[y_train_onehot, dummy_matrix], 
    #                       batch_size=512, epochs=100, verbose=1, 
    #                       validation_split=0.2, 
    #                       callbacks=[checkpoint, EarlyStopping(monitor="val_dense_2_loss",patience=20)])


    # train custom model
    model = center_loss_model_custom()
    model.load_weights("custom_weights_08_val_acc_0.937.h5", by_name=True)
    filepath = "./custom_weights_{epoch:02d}_val_acc_{val_dense_2_acc:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, monitor="val_dense_2_loss", mode='min', save_best_only=True)
    model.fit(x=[x_train, y_train_onehot], 
              y=[y_train_onehot, dummy_matrix], 
              batch_size=512, epochs=100, verbose=1, 
              validation_split=0.2, 
              callbacks=[checkpoint, EarlyStopping(monitor="val_dense_2_loss",patience=20)])




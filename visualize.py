from keras.utils import to_categorical
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
from model import load_mnist, center_loss_model


x_test, y_test = load_mnist()
x_test = np.expand_dims(x_test[:1000],axis=-1)
y_test = y_test[:1000]
y_test_one_hot = to_categorical(y_test)


# model
model = center_loss_model()
model.load_weights("./centerloss_weights_31_val_acc_0.942.h5", by_name=True)

# predict
func = K.function([model.input[0]],[model.get_layer('out1').output])   # one of the multi-input
test_features = func([x_test])[0]
print(test_features.shape)

# centers
test_centers = np.dot(np.transpose(y_test_one_hot),test_features)
test_centers_count = np.sum(np.transpose(y_test_one_hot),axis=1,keepdims=True)
test_centers /= test_centers_count
print(test_centers.shape)

# visualize
plt.scatter(test_features[:,0],test_features[:,1],c=y_test,edgecolor="none",s=5)
plt.scatter(test_centers[:,0],test_centers[:,1],c="black",marker="*",edgecolor="none",s=50)
plt.show()

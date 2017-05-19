import numpy as np
import time

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from sklearn.metrics import roc_auc_score

from data import read_data_from_disk

NB_EPOCHS = 20
BATCH_SIZE = 32
NB_INPUTS = 36*9+2+4+1+1+1
NB_TRAIN = 3200
NB_VALID = 400
NB_TEST = 4000 - NB_TRAIN - NB_VALID


class AurocCallback(Callback):
    def __init__(self, model, X, Y, batch_size=32):
        self.model = model
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        Y_prob = model.predict(self.X, batch_size=self.batch_size)
        auroc = roc_auc_score(self.Y, Y_prob, average=None)
        if len(self.Y.shape) == 1 or Y.shape[1] == 1:
            auroc = np.array([auroc])
        self.time = int(time.time() - self.time)
        msg = 'epoch {0:3d} ({1:6d}s): {2:.4f}'.format(epoch, self.time, np.mean(auroc))
        msg += ' ['
        for i in range(auroc.shape[0]):
            msg += ' {0:.4f}'.format(auroc[i])
        msg += ' ]'
        print('\n\n' + msg)

X, Y = read_data_from_disk('../src/main/resources/physionet2012/features',
                           target_dirs='../src/main/resources/physionet2012/mortality',
                           end=4000)
Y = Y[0].ravel()

X_train = X[:NB_TRAIN]
Y_train = Y[:NB_TRAIN]
X_valid = X[NB_TRAIN:NB_TRAIN+NB_VALID]
Y_valid = Y[NB_TRAIN:NB_TRAIN+NB_VALID]
X_test  = X[NB_TRAIN+NB_VALID:]
Y_test  = Y[NB_TRAIN+NB_VALID:]

input = Input((NB_INPUTS,))
H = Dense(500, activation='relu')(input)
output = Dense(1, activation='sigmoid')(H)
model = Model(input, output)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCHS,
          validation_data=(X_valid, Y_valid),
          callbacks=[TensorBoard(), AurocCallback(model, X_valid, Y_valid), ModelCheckpoint('ex1_keras_mortality_mlp.h5')])

model = load_model('ex1_keras_mortality_mlp.h5')

Y_prob = model.predict(X_train)
print('Final training AUC:', roc_auc_score(Y_train, Y_prob))

Y_prob = model.predict(X_test)
print('Final test AUC:    ', roc_auc_score(Y_test, Y_prob))

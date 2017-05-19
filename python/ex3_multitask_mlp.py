import numpy as np

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.utils import np_utils
from sklearn.metrics import roc_auc_score

from data import read_data_from_disk
from util import AurocCallback

NB_EPOCHS = 20
BATCH_SIZE = 32
NB_INPUTS = 36*9+2+4+1+1+1
NB_MORTALITY_TARGETS = 1
NB_LOS_TARGETS = 8
NB_SURVIVAL_TARGETS = 5
NB_TRAIN = 3200
NB_VALID = 400
NB_TEST = 4000 - NB_TRAIN - NB_VALID


X, Y = read_data_from_disk('../src/main/resources/physionet2012/features',
                           target_dirs=[ '../src/main/resources/physionet2012/mortality',
                                         '../src/main/resources/physionet2012/los_bucket',
                                         '../src/main/resources/physionet2012/survival_bucket' ],
                           end=4000)
Y[1] = np_utils.to_categorical(Y[1], NB_LOS_TARGETS)
Y[2] = np_utils.to_categorical(Y[2], NB_SURVIVAL_TARGETS)

X_train = X[:NB_TRAIN]
Y_train = [ y[:NB_TRAIN] for y in Y ]
X_valid = X[NB_TRAIN:NB_TRAIN+NB_VALID]
Y_valid = [ y[NB_TRAIN:NB_TRAIN+NB_VALID] for y in Y ]
X_test  = X[NB_TRAIN+NB_VALID:]
Y_test  = [ y[NB_TRAIN+NB_VALID:] for y in Y ]

input = Input((NB_INPUTS,))
H = Dense(1000, activation='relu')(input)
mortality = Dense(NB_MORTALITY_TARGETS, activation='sigmoid')(H)
los = Dense(NB_LOS_TARGETS, activation='sigmoid')(H)
survival = Dense(NB_SURVIVAL_TARGETS, activation='sigmoid')(H)
model = Model(input, [mortality, los, survival])
model.compile(optimizer='adam', loss=['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'])

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCHS,
          validation_data=(X_valid, Y_valid),
          callbacks=[TensorBoard(), AurocCallback(model, X_valid, Y_valid), ModelCheckpoint('ex3_keras_multitask_mlp.h5')])

model = load_model('ex3_keras_multitask_mlp.h5')

Y_prob = model.predict(X_train, batch_size=BATCH_SIZE)
auroc = np.array([ roc_auc_score(y, yp) for y, yp in zip(Y_train, Y_prob) ])
msg = 'Final training AUROC: {0:.4f}'.format(np.mean(auroc))
msg += ' ['
for i in range(auroc.shape[0]):
    msg += ' {0:.4f}'.format(auroc[i])
msg += ' ]'
print('\n' + msg)

Y_prob = model.predict(X_test, batch_size=BATCH_SIZE)
auroc = np.array([ roc_auc_score(y, yp) for y, yp in zip(Y_test, Y_prob) ])
msg = 'Final training AUROC: {0:.4f}'.format(np.mean(auroc))
msg += ' ['
for i in range(auroc.shape[0]):
    msg += ' {0:.4f}'.format(auroc[i])
msg += ' ]'
print('\n' + msg)

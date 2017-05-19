import numpy as np
import time

from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


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
        Y_prob = self.model.predict(self.X, batch_size=self.batch_size)
        if type(Y_prob) is list:
            auroc = np.array([ roc_auc_score(y, yp) for y, yp in zip(self.Y, Y_prob) ])
        else:
            auroc = np.array([ roc_auc_score(self.Y, Y_prob) ])
        self.time = int(time.time() - self.time)
        msg = 'epoch {0:3d} ({1:6d}s): {2:.4f}'.format(epoch, self.time, np.mean(auroc))
        msg += ' ['
        for i in range(auroc.shape[0]):
            msg += ' {0:.4f}'.format(auroc[i])
        msg += ' ]'
        print('\n\n' + msg)

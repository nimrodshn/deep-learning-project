import numpy as np

class DiceMetric():
    def dice(self,y_true, y_pred):
        y_t = np.reshape(y_true, -1)
        y_p = np.reshape(y_pred, -1)
        y_int = y_t*y_p
        return (2*(np.sum(y_int)))/((np.sum(y_t)) + (np.sum(y_p)))

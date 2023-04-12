from sklearn.metrics import brier_score_loss
import numpy as np


def calculate_BS(y_target, y_pred, n_class):
    '''
    input: 要計算的 2D-array (已經取過 idx 的)
    '''
    n_BS = []
    for i in range(n_class):
        n_BS.append(brier_score_loss(y_target[:, i], y_pred[:, i]))

    mean_BS = np.mean(n_BS)
    return {'BS': n_BS, 'mean_BS': mean_BS}
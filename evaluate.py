from sklearn.metrics import brier_score_loss
import numpy as np
from keras import backend as K


def calculate_BS(y_target, y_pred, n_class):
    '''
    input: 要計算的 2D-array (已經取過 idx 的)
    '''
    n_BS = []
    for i in range(n_class):
        n_BS.append(brier_score_loss(y_target[:, i], y_pred[:, i]))

    mean_BS = np.mean(n_BS)
    return {'BS': n_BS, 'mean_BS': mean_BS}



def f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
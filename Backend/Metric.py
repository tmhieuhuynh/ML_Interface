import numpy as np
import math

def PearsonCoefficient(X, Y):
    r = np.sum((X - np.average(X)) * (Y - np.average(Y))) / math.sqrt(np.sum((X - np.average(X)) ** 2) * np.sum((Y - np.average(Y)) ** 2))
    
    return r

def RMSE(X, Y):
    rmse = math.sqrt(np.sum((Y - X) ** 2) / len(Y))
    
    return rmse

def MAPE(X,Y):
    mape = np.average(abs((X - Y) / Y)) * 100
    
    return mape

def neg_MAE(X,Y):
    mae = np.average(abs(X - Y))
    
    return -mae

def PerOut(X, Y, cutoff):    
    return (np.count_nonzero(abs(X - Y) > cutoff) / len(Y)) * 100


def get_metrics(y_pred_list, y_exp_list, cutoff1, cutoff2):

	y_pred_list = np.array(y_pred_list)
	y_exp_list = np.array(y_exp_list)
	MAPE_ = MAPE(y_pred_list, y_exp_list)
	neg_MAE_ = neg_MAE(y_pred_list, y_exp_list)
	RMSE_ = RMSE(y_pred_list, y_exp_list)
	r_ = PearsonCoefficient(y_pred_list, y_exp_list)
	P1 = PerOut(y_pred_list, y_exp_list, cutoff1)
	P2 = PerOut(y_pred_list, y_exp_list, cutoff2)

	metrics = np.array([MAPE_,neg_MAE_, RMSE_, r_, P1, P2])

	return metrics
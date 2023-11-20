import pandas as pd
import numpy as np
from numpy import mean, std
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import os
from collections import OrderedDict

#############
# LOAD DATA #
#############
def boolean(a):
    if a=='Yes':
        return True
    else:
        return False
def scaler(a):
    if a=='MinMax':
        return MinMaxScaler()
    elif a=='Standard':
        return StandardScaler()
    elif a=='Robust':
        return RobustScaler()
    else:
        return 'passthrough'

def hiddenlayer(a,b):
    if int(b)==0:
        return [int(a)]
    else:
        return [int(a),int(b)]
    
# Get a list of models to evaluate
def get_models(control):

    models = OrderedDict()
	
    models['lr'] = Pipeline([('scaler',scaler(control.RidgeParameters['Scaler'].GetValue())),
                             ('lr',LinearRegression())])
	
    models['ridge'] = Pipeline([('scaler',scaler(control.RidgeParameters['Scaler'].GetValue())),
                                ('ridge', Ridge(solver=control.RidgeParameters['Solver'].GetValue(),
                                                alpha=10**int(control.RidgeParameters['Alpha'].GetValue()),
                                                fit_intercept=boolean(control.RidgeParameters['FitIntercept'].GetValue())))])
	
    models['svr'] =Pipeline([('scaler', scaler(control.SvrParameters['Scaler'].GetValue())),
                             ('svr', SVR(C= int(control.SvrParameters['C'].GetValue()),
                                         epsilon=int(control.SvrParameters['Epsilon'].GetValue())/100,
                                         gamma= control.SvrParameters['Gamma'].GetValue(),
                                         kernel= control.SvrParameters['Kernel'].GetValue()))])
    
    models['knn'] = Pipeline([('scaler', scaler(control.KnnParameters['Scaler'].GetValue())), 
							  ('reduce_dim', PCA(n_components=int(control.KnnParameters['Dim'].GetValue()))), 
							  ('knn', KNeighborsRegressor(n_neighbors=int(control.KnnParameters['Neighbor'].GetValue()), 
							  							  weights=control.KnnParameters['Weight'].GetValue(), 
							  							  algorithm=control.KnnParameters['Algorithm'].GetValue(), 
							  							  leaf_size=int(control.KnnParameters['Leaf'].GetValue())))])
    

    models['ann'] = Pipeline([('scaler', scaler(control.AnnParameters['Scaler'].GetValue())), 
							  ('reduce_dim', PCA(n_components=int(control.AnnParameters['Dim'].GetValue()))), 
							  ('mlp', MLPRegressor(solver=control.AnnParameters['Solver'].GetValue(),
							  					   hidden_layer_sizes=hiddenlayer(control.AnnParameters['HiddenLayer1'].GetValue(),control.AnnParameters['HiddenLayer2'].GetValue()), 
							  					   activation=control.AnnParameters['Activation'].GetValue(), 							  					    
							  					   alpha=10**int(control.AnnParameters['Alpha'].GetValue()), 
							  					   max_iter=10**int(control.AnnParameters['Iter'].GetValue()), 
							  					   momentum=int(control.AnnParameters['Momentum'].GetValue())/10, 
							  					   nesterovs_momentum=boolean(control.AnnParameters['Nesterovs'].GetValue())))])

    models['forest'] = RandomForestRegressor(n_estimators=int(control.RfParameters['Estimator'].GetValue()), 
											 criterion=control.RfParameters['Criterion'].GetValue(), 
											 min_samples_split=int(control.RfParameters['MinSplit'].GetValue()), 
											 max_depth=int(control.RfParameters['Depth'].GetValue()), 
											 min_samples_leaf=int(control.RfParameters['MinLeaf'].GetValue()), 
											 max_features=control.RfParameters['MaxFeature'].GetValue(),
											 bootstrap=boolean(control.RfParameters['Bootstrap'].GetValue()))


    models['gbrt'] = GradientBoostingRegressor(n_estimators=int(control.GbParameters['Estimator'].GetValue()), 
											   max_depth=int(control.GbParameters['Depth'].GetValue()), 
											   learning_rate=int(control.GbParameters['Learning'].GetValue())/1000, 
											   criterion=control.GbParameters['Criterion'].GetValue(), 
											   min_samples_split=int(control.GbParameters['MinSplit'].GetValue()), 
											   min_samples_leaf=int(control.GbParameters['MinLeaf'].GetValue()), 
											   max_features=control.GbParameters['MaxFeature'].GetValue())
    
    return models

# Evaluate a given model using cross-validation
def evaluate_model(control, model, X, y):
	cv = RepeatedKFold(n_splits=int(control.RepeatedParameters['NSplit'].GetValue()),
                       n_repeats=int(control.RepeatedParameters['NRepeat'].GetValue()),
                       random_state=int(control.RepeatedParameters['RandomState'].GetValue()))
	scores1 = cross_val_score(model, X, y, scoring='neg_mean_absolute_percentage_error', cv=cv, n_jobs=-1, error_score='raise')
	scores2 = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
	scores3 = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')
	#scores = cross_val_score(model, X, y, scoring='explained_variance', cv=cv, n_jobs=-1, error_score='raise')	

	return scores1, scores2, scores3
	#return scores


def cross_validation(control, train, test, trainX, trainy, testX, testy, traintest, traintestX, traintesty, cutoff1, cutoff2, Yfeatures, Xfeatures, Subfeatures, InputName):
	# INITIALIZE MACHINE LEARNING MODELS
    models = get_models(control)
	# Evaluate the models and store results
    MAPE, RMSE, r_2, names = list(), list(), list(), list()
    results, names = list(), list()
    
    included = []
    if control.lr.GetValue()==True:
        included.append('lr')
    if control.ridge.GetValue()==True:
        included.append('ridge')
    if control.svr.GetValue()==True:
        included.append('svr')
    if control.knn.GetValue()==True:
        included.append('knn')
    if control.ann.GetValue()==True:
        included.append('ann')
    if control.rf.GetValue()==True:
        included.append('forest')
    if control.gb.GetValue()==True:
        included.append('gbrt')

    for name, model in models.items():
        if name in included:
            scores1, scores2, scores3 = evaluate_model(control, model, traintestX, traintesty)
            MAPE.append(scores1)
            RMSE.append(scores2)
            r_2.append(scores3)
            temp = np.sqrt(scores3)
            FOM = (mean(scores1), std(scores1), mean(scores2), std(scores2), mean(temp), std(temp))
            results.append(FOM)
    		
            names.append(name)
    		#print('> %s %.3f (%.3f)' % (name, mean(scores), std(scores)))
            print('> %s %.5f %.5f' % (name, mean(scores1), std(scores1)),'%.5f %.5f' % (mean(scores2), std(scores2)),
    									'%.3f %.3f' % (mean(temp), std(temp)))

	
    cross_validation_summary = pd.DataFrame(results, index=names)
	#print(cr_val_stat)
	# Save to csv
    cross_validation_summary.to_csv(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Cross_Validation','cross_validation_summary.csv'))

    MAPE = np.transpose(MAPE)
    MAPE = pd.DataFrame(MAPE,columns=names)
	#print(MAPE)
	# Save to csv
    MAPE.to_csv(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Cross_Validation','MAPE.csv'))

    RMSE = np.transpose(RMSE)
    RMSE = pd.DataFrame(RMSE,columns=names)
	#print(RMSE)
	# Save to csv
    RMSE.to_csv(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Cross_Validation','RMSE.csv'))

    r_2 = np.transpose(r_2)
    r_2 = pd.DataFrame(r_2,columns=names)
    Pearson = np.sqrt(r_2)
	#print(Pearson)
	# Save to csv
    Pearson.to_csv(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Cross_Validation','Pearson.csv'))

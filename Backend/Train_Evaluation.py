import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import os
from collections import OrderedDict

from Backend.Metric import get_metrics


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
							  ('reduce_dim', PCA(n_components=int(control.KnnParameters['Dim'].GetValue()))), #this is best hyperparameters for 497 descriptors; for 67 descriptors n_components=5
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
    
    voting_model, stacking_model = get_ensemble_models(control,models)

    models['voting'] = voting_model
    models['stacking'] = stacking_model	
	
    return models
 

# Helper function for `get_models()`. Get an ensemble model from a dictionary of models.

def get_ensemble_models(control,models):
	
	# Base models
    level0 = list()

	# Exclude any model that you think might negatively contribute to the ensemble
    excluded = []
    if control.EnsembleParameters['lr']==False:
        excluded.append('lr')
    if control.EnsembleParameters['ridge']==False:
        excluded.append('ridge')
    if control.EnsembleParameters['svr']==False:
        excluded.append('svr')
    if control.EnsembleParameters['knn']==False:
        excluded.append('knn')
    if control.EnsembleParameters['ann']==False:
        excluded.append('ann')
    if control.EnsembleParameters['rf']==False:
        excluded.append('forest')
    if control.EnsembleParameters['gb']==False:
        excluded.append('gbrt')

    for model_name in models.keys():
        if model_name not in excluded:
            level0.append((model_name, models[model_name]))
			
	# Define meta learner model
    level1 = LinearRegression()

    voting_model = VotingRegressor(estimators=level0)	

	# Define the stacking ensemble
    stacking_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)	
		
    return voting_model, stacking_model

def train_evaluate(control,train,test,trainX,trainy,testX,testy,traintest,traintestX,traintesty,cutoff1,cutoff2,Yfeatures,Xfeatures,Subfeatures,name):

	# INITIALIZE MACHINE LEARNING MODELS
    models = get_models(control)
    lr = models['lr']
    ridge = models['ridge']
    knn = models['knn']
    ann = models['ann']
    forest = models['forest']
    gbrt = models['gbrt']
    svr = models['svr']
    voting = models['voting']
    stacking = models['stacking']

    prediction=[list(testy)]
    pred_tabs = [Yfeatures[0]+'_DATA']
    columns_FI = ['Decriptor']
    data_FI=[Xfeatures]
    test_metrics=[]
	#####################
	# LINEAR REGRESSION #
	#####################
    if control.lr.GetValue() == True:
        lr.fit(trainX,trainy)
        test_pred_LR = lr.predict(testX)
        prediction.append(test_pred_LR)
        test_metrics_LR = get_metrics(test_pred_LR, testy, cutoff1, cutoff2)
    
        print("\nLinear Regression: ", test_metrics_LR)
        importances_LR = lr.named_steps['lr'].coef_
        data_FI.append(importances_LR)

	#########
	# RIDGE #
	#########
    if control.ridge.GetValue() == True:
        ridge.fit(trainX, trainy)
        test_pred_RIDGE = ridge.predict(testX)
        prediction.append(list(test_pred_RIDGE))
        test_metrics_RIDGE = get_metrics(test_pred_RIDGE, testy, cutoff1, cutoff2)
    
        print("\nRidge Regression: ", test_metrics_RIDGE)
        importances_RIDGE = ridge.named_steps['ridge'].coef_
        data_FI.append(importances_RIDGE)
	
	##################
	# SVR            #
	##################
    if control.svr.GetValue() == True:
        svr.fit(trainX, trainy)
        test_pred_SVR = svr.predict(testX)
        prediction.append(list(test_pred_SVR))
        test_metrics_SVR = get_metrics(test_pred_SVR, testy, cutoff1, cutoff2)
    
        print("\nSVR: ", test_metrics_SVR)

	#######################
	# K-NEAREST NEIGHBORS #
	#######################
    if control.knn.GetValue() == True:
        knn.fit(trainX, trainy)
        test_pred_KNN = knn.predict(testX)
        prediction.append(list(test_pred_KNN))
        test_metrics_KNN = get_metrics(test_pred_KNN, testy, cutoff1, cutoff2)
    
        print("\nK-Nearest Neighbors: ", test_metrics_KNN)

	##################
	# NEURAL NETWORK #
	##################
    if control.ann.GetValue() == True:
        ann.fit(trainX, trainy)
        test_pred_ANN = ann.predict(testX)
        prediction.append(list(test_pred_ANN))
        test_metrics_ANN = get_metrics(test_pred_ANN, testy, cutoff1, cutoff2)
        print("\nNeural Network: ", test_metrics_ANN)

	#################
	# RANDOM FOREST #
	#################	
    if control.rf.GetValue() == True:
        forest.fit(trainX, trainy)
        test_pred_RF = forest.predict(testX)
        prediction.append(list(test_pred_RF))
        test_metrics_RF = get_metrics(test_pred_RF, testy, cutoff1, cutoff2)
        print("\nRandom Forest: ", test_metrics_RF)
        importances_RF = forest.feature_importances_
        data_FI.append(importances_RF)

	#####################
	# GRADIENT BOOSTING #
	#####################
    if control.gb.GetValue() == True:	
        gbrt.fit(trainX, trainy)
        test_pred_GB = gbrt.predict(testX)
        prediction.append(list(test_pred_GB))
        test_metrics_GB = get_metrics(test_pred_GB, testy, cutoff1, cutoff2)
    
        print("\nGradient Boosting: ", test_metrics_GB)
        importances_GB = gbrt.feature_importances_	
        data_FI.append(importances_GB)

	##################
	# STACKING MODEL #
	##################	
    if control.st.GetValue() == True:
        stacking.fit(trainX, trainy)
        test_pred_STACKING = stacking.predict(testX)
        prediction.append(list(test_pred_STACKING))
        test_metrics_STACKING = get_metrics(test_pred_STACKING, testy, cutoff1, cutoff2)
    
        print("\nStacking Ensemble: ", test_metrics_STACKING)	

	################
	# VOTING MODEL #
	################
    if control.vt.GetValue() == True:
        voting.fit(trainX, trainy)
        test_pred_VOTING = voting.predict(testX)
        prediction.append(list(test_pred_VOTING))
        test_metrics_VOTING = get_metrics(test_pred_VOTING, testy, cutoff1, cutoff2)
    
        print("\nVoting Ensemble: ", test_metrics_VOTING)	

	# SAVE TEST RESULTS
    columns = ['MAPE','MAE', 'RMSE', 'r', 'P1Sigma', 'P2Sigma']
    index = []
    if control.lr.GetValue() == True:
        index.append('LIN-REG')
        test_metrics.append(test_metrics_LR)
        pred_tabs.append(Yfeatures[0]+'_ML_LR')
        columns_FI.append('FI_LR')
    if control.ridge.GetValue() == True:
        index.append('RIDGE')
        test_metrics.append(test_metrics_RIDGE)
        pred_tabs.append(Yfeatures[0]+'_ML_RIDGE')
        columns_FI.append('FI_RIDGE')
    if control.svr.GetValue() == True:
        index.append('SVR')
        test_metrics.append(test_metrics_SVR)
        pred_tabs.append(Yfeatures[0]+'_ML_SVR')
    if control.knn.GetValue() == True:
        index.append('KNN')
        test_metrics.append(test_metrics_KNN)
        pred_tabs.append(Yfeatures[0]+'_ML_KNN')
    if control.ann.GetValue() == True:
        index.append('ANN')
        test_metrics.append(test_metrics_ANN)
        pred_tabs.append(Yfeatures[0]+'_ML_ANN')
    if control.rf.GetValue() == True:
        index.append('RF')
        test_metrics.append(test_metrics_RF)
        pred_tabs.append(Yfeatures[0]+'_ML_RF')
        columns_FI.append('FI_RF')
    if control.gb.GetValue() == True:
        index.append('GB')
        test_metrics.append(test_metrics_GB)
        pred_tabs.append(Yfeatures[0]+'_ML_GB')
        columns_FI.append('FI_GB')
    if control.vt.GetValue() == True:
        index.append('VOTING')
        test_metrics.append(test_metrics_VOTING)
        pred_tabs.append(Yfeatures[0]+'_ML_VOTING')
    if control.st.GetValue() == True:
        index.append('STACKING')
        test_metrics.append(test_metrics_STACKING)
        pred_tabs.append(Yfeatures[0]+'_ML_STACKING')
        
    df = pd.DataFrame(data=test_metrics, index=index, columns=columns)
    print(df)
    
    output_name = name[:name.index('.')]+'_test_result'

	# Save to csv
    df.to_csv(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Train_Evaluation', output_name + '.csv'), float_format='%2.2f')
	
	# Save to text
    df = df.round(decimals=2)
    with open(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Train_Evaluation', output_name + '.txt'), 'w') as f:
        f.write(df.to_string())	

	#Save all the predicted charge transfer into an csv file	
	#Putting all the d_xxx arrays into dataframe; 
    sub1=[]
    for x in range (0,len(prediction[0])):
        part=[]
        for y in range (0,len(prediction)):
            part.append(prediction[y][x])
        sub1.append(part)
        
    df_all = pd.DataFrame(data=sub1, columns=pred_tabs)
    df_all=pd.concat([test.iloc[:,:len(Subfeatures)].reset_index().iloc[:,1:],df_all],axis=1)

    np.set_printoptions(precision=5)
    df_all.to_csv(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Train_Evaluation', 'Allpredicted_'+Yfeatures[0]+'_test.csv'),index=False)

	#SAVE FEATURE IMPORTANCES
    sub2=[]
    for x in range (0,len(data_FI[0])):
        part=[]
        for y in range (0,len(data_FI)):
            part.append(data_FI[y][x])
        sub2.append(part)
    df_FI = pd.DataFrame(data=sub2, columns = columns_FI)
	#np.set_printoptions(precision=5)
    df_FI.to_csv(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Train_Evaluation','Feature.Importance.csv'))
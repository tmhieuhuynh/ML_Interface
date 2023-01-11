import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from Backend.Metric import get_metrics
import os

def GridSearch(main):
    if main.Analyzing.tabOne.ridge.GetValue() == True:
        Ridge_grid_search(main, main.Analyzing.tabOne.RidgeParameters, main.Analyzing.tabOne.MetricBox.GetValue(), main.traintest, main.train, main.test, main.trainX, main.trainy, main.testX, main.testy, main.traintestX, main.traintesty,main.cutoff1,main.cutoff2)
    if main.Analyzing.tabOne.knn.GetValue() == True:
        KNN_grid_search(main, main.Analyzing.tabOne.KnnParameters, main.Analyzing.tabOne.MetricBox.GetValue(), main.traintest, main.train, main.test, main.trainX, main.trainy, main.testX, main.testy, main.traintestX, main.traintesty,main.cutoff1,main.cutoff2)
    if main.Analyzing.tabOne.svr.GetValue() == True:
        SVR_grid_search(main, main.Analyzing.tabOne.SvrParameters, main.Analyzing.tabOne.MetricBox.GetValue(), main.traintest, main.train, main.test, main.trainX, main.trainy, main.testX, main.testy, main.traintestX, main.traintesty,main.cutoff1,main.cutoff2)
    if main.Analyzing.tabOne.ann.GetValue() == True:
        ANN_grid_search(main, main.Analyzing.tabOne.AnnParameters, main.Analyzing.tabOne.MetricBox.GetValue(), main.traintest, main.train, main.test, main.trainX, main.trainy, main.testX, main.testy, main.traintestX, main.traintesty,main.cutoff1,main.cutoff2)
    if main.Analyzing.tabOne.rf.GetValue() == True:
        RF_grid_search(main, main.Analyzing.tabOne.RfParameters, main.Analyzing.tabOne.MetricBox.GetValue(), main.traintest, main.train, main.test, main.trainX, main.trainy, main.testX, main.testy, main.traintestX, main.traintesty, main.cutoff1,main.cutoff2)
    if main.Analyzing.tabOne.gb.GetValue() == True:
        GB_grid_search(main, main.Analyzing.tabOne.GbParameters, main.Analyzing.tabOne.MetricBox.GetValue(), main.traintest, main.train, main.test, main.trainX, main.trainy, main.testX, main.testy, main.traintestX, main.traintesty, main.cutoff1,main.cutoff2)

def grid_search(main, model, metric, trainX, trainy, param_grid, save_result_path=None, verbose=False):
	# Define evaluation
    cv = RepeatedKFold(n_splits=int(main.Analyzing.tabOne.RepeatedParameters['NSplit'].GetValue()),
                       n_repeats=int(main.Analyzing.tabOne.RepeatedParameters['NRepeat'].GetValue()),
                       random_state=1)

	# Define search
    search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=metric, n_jobs=None, cv=cv, verbose=verbose)

	# Run grid search
    result = search.fit(trainX, trainy)

    if save_result_path:
        with open(save_result_path, 'a') as f:			
            f.write("\nBest score: %s" % result.best_score_)
            f.write("\nBest hyperparameters: %s \n" % result.best_params_)

    if verbose:
		# Summarize result
        print("Best score: %s" % result.best_score_)
        print("Best hyperparameters: %s" % result.best_params_)

    return result


def fit_model(model, model_name, traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2):

	# Fit model
    model.fit(trainX, trainy)

	# Test model
    test_pred = model.predict(testX)
    test_metrics = get_metrics(test_pred, testy, cutoff1, cutoff2)

    print("\n{}: {} ".format(model_name, test_metrics))

    return test_pred, test_metrics


def Ridge_grid_search(main, parameters, metric, traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2):

    model = Pipeline([('scaler', 'passthrough'), 
					  ('ridge', Ridge())])

	# Define search space
    param_grid = dict()
    param_grid['scaler'] = []
    if parameters['ScalerMinMax'].GetValue() == True:
        param_grid['scaler'].append(MinMaxScaler())
    if parameters['ScalerStandard'].GetValue() == True:
        param_grid['scaler'].append(StandardScaler())
    if parameters['ScalerRobust'].GetValue() == True:
        param_grid['scaler'].append(RobustScaler())
    if parameters['ScalerNone'].GetValue() == True:
        param_grid['scaler'].append('passthrough')
        
    param_grid['ridge__solver'] = []
    if parameters['SolverSvd'].GetValue() == True:
        param_grid['ridge__solver'].append('svd')
    if parameters['SolverCholesky'].GetValue() == True:
        param_grid['ridge__solver'].append('cholesky')
    if parameters['SolverLsqr'].GetValue() == True:
        param_grid['ridge__solver'].append('lsqr')
    if parameters['SolverSag'].GetValue() == True:
        param_grid['ridge__solver'].append('sag')
        
    param_grid['ridge__alpha'] = []
    for i in range (int(parameters['AlphaStart'].GetValue()), int(parameters['AlphaEnd'].GetValue())+1):
        param_grid['ridge__alpha'].append(10**i)
    
    param_grid['ridge__fit_intercept'] = []
    if parameters['FitInterceptYes'].GetValue() == True:
        param_grid['ridge__fit_intercept'].append(True)
    if parameters['FitInterceptNo'].GetValue() == True:
        param_grid['ridge__fit_intercept'].append(False)
    
    save_result_path = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Grid_Search','ridge_best_hyperparamaters.txt')
    result = grid_search(main, model, metric, trainX, trainy, param_grid, save_result_path=save_result_path, verbose=True)
	
	# Define the best model based on grid search result
    ridge = Pipeline([('scaler',result.best_params_['scaler']),
                      ('ridge', Ridge(solver=result.best_params_['ridge__solver'], 
				                      alpha=result.best_params_['ridge__alpha'], 
				                      fit_intercept=result.best_params_['ridge__fit_intercept']))])

    _, test_metrics = fit_model(ridge, 'Ridge Regression',traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2)


def KNN_grid_search(main, parameters, metric, traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2):

    model = Pipeline([('scaler', 'passthrough'), 
					  ('reduce_dim', PCA()), 
					  ('knn', KNeighborsRegressor())])

	# Define search space
    param_grid = dict()
    param_grid['scaler'] = []
    if parameters['ScalerMinMax'].GetValue() == True:
        param_grid['scaler'].append(MinMaxScaler())
    if parameters['ScalerStandard'].GetValue() == True:
        param_grid['scaler'].append(StandardScaler())
    if parameters['ScalerRobust'].GetValue() == True:
        param_grid['scaler'].append(RobustScaler())
    
    param_grid['reduce_dim__n_components'] = np.arange(int(parameters['DimStart'].GetValue()), int(parameters['DimEnd'].GetValue())+1)
	#param_grid['reduce_dim__k'] = np.arange(1, 11)	
    param_grid['knn__n_neighbors'] = np.arange(int(parameters['NeighborStart'].GetValue()), int(parameters['NeighborEnd'].GetValue())+int(parameters['NeighborStep'].GetValue()), int(parameters['NeighborStep'].GetValue()))
    param_grid['knn__weights'] = ['distance']
    param_grid['knn__algorithm'] = ['auto']
    param_grid['knn__leaf_size'] = np.arange(int(parameters['LeafStart'].GetValue()), int(parameters['LeafEnd'].GetValue())+int(parameters['LeafStep'].GetValue()), int(parameters['LeafStep'].GetValue()))

    save_result_path = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Grid_Search','knn_best_hyperparamaters.txt')

    result = grid_search(main, model, metric, trainX, trainy, param_grid, save_result_path=save_result_path, verbose=True)

	# Define the best model based on grid search result
    knn = Pipeline([('scaler', result.best_params_['scaler']),
					('knn', KNeighborsRegressor(n_neighbors=result.best_params_['knn__n_neighbors'],
							  					weights=result.best_params_['knn__weights'],
							  					algorithm=result.best_params_['knn__algorithm'],
							  					leaf_size=result.best_params_['knn__leaf_size']))])

    _, test_metrics = fit_model(knn, 'KNN', traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2)

def SVR_grid_search(main, parameters, metric, traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2):

    model = Pipeline([('scaler', 'passthrough'), 
					  ('svr', SVR())])

	# Define search space
    param_grid = dict()
    param_grid['scaler'] = []
    if parameters['ScalerMinMax'].GetValue() == True:
        param_grid['scaler'].append(MinMaxScaler())
    if parameters['ScalerStandard'].GetValue() == True:
        param_grid['scaler'].append(StandardScaler())
    if parameters['ScalerRobust'].GetValue() == True:
        param_grid['scaler'].append(RobustScaler())
    if parameters['ScalerNone'].GetValue() == True:
        param_grid['scaler'].append('passthrough')
    
    param_grid['svr__kernel'] = []
    if parameters['KernelLinear'].GetValue() == True:
        param_grid['svr__kernel'].append('linear')
    if parameters['KernelPoly'].GetValue() == True:
        param_grid['svr__kernel'].append('poly')
    if parameters['KernelRbf'].GetValue() == True:
        param_grid['svr__kernel'].append('rbf')
    if parameters['KernelSigmoid'].GetValue() == True:
        param_grid['svr__kernel'].append('sigmoid')
    if parameters['KernelPrecomputed'].GetValue() == True:
        param_grid['svr__kernel'].append('precomputed')
    
    param_grid['svr__gamma'] = []
    if parameters['GammaScale'].GetValue() == True:
        param_grid['svr__gamma'].append('scale')
    if parameters['GammaAuto'].GetValue() == True:
        param_grid['svr__gamma'].append('auto')
    
    param_grid['svr__epsilon'] = np.arange(int(parameters['EpsilonStart'].GetValue()), int(parameters['EpsilonEnd'].GetValue())+int(parameters['EpsilonStep'].GetValue()), int(parameters['EpsilonStep'].GetValue()))/100
    param_grid['svr__C'] = np.arange(int(parameters['CStart'].GetValue()), int(parameters['CEnd'].GetValue())+1, 1)

    save_result_path = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Grid_Search','svr_best_hyperparamaters.txt')

    result = grid_search(main, model, metric, trainX, trainy, param_grid, save_result_path=save_result_path, verbose=True)

	# Define the best model based on grid search result
    knn = Pipeline([('scaler', result.best_params_['scaler']),
					('svr', SVR(kernel=result.best_params_['svr__kernel'],
							  			gamma=result.best_params_['svr__gamma'],
							  			epilson=result.best_params_['svr__epsilon'],
							  			C=result.best_params_['svr__C']))])

    _, test_metrics = fit_model(knn, 'KNN', traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2)


def ANN_grid_search(main, parameters, metric, traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2):

    model = Pipeline([('scaler', 'passthrough'), 
					  ('reduce_dim', PCA()), 
					  #('reduce_dim', SelectKBest(f_regression)), 
					  ('mlp', MLPRegressor())])	

	# Define search space
    param_grid = dict()
    param_grid['scaler'] = []
    if parameters['ScalerMinMax'].GetValue() == True:
        param_grid['scaler'].append(MinMaxScaler())
    if parameters['ScalerStandard'].GetValue() == True:
        param_grid['scaler'].append(StandardScaler())
    if parameters['ScalerRobust'].GetValue() == True:
        param_grid['scaler'].append(RobustScaler())
        
    param_grid['reduce_dim__n_components'] = np.arange(int(parameters['DimStart'].GetValue()), int(parameters['DimEnd'].GetValue())+1)
	#param_grid['reduce_dim__k'] = np.arange(1, 11)
    
    param_grid['mlp__solver'] = []
    if parameters['SolverAdam'].GetValue() == True:
        param_grid['mlp__solver'].append('adam')
    if parameters['SolverLbfgs'].GetValue() == True:
        param_grid['mlp__solver'].append('lbfgs')

    param_grid['mlp__max_iter'] = [10000]
    param_grid['mlp__hidden_layer_sizes'] = np.arange(int(parameters['HiddenLayerStart'].GetValue()), int(parameters['HiddenLayerEnd'].GetValue())+int(parameters['HiddenLayerStep'].GetValue()), int(parameters['HiddenLayerStep'].GetValue()))
    
    param_grid['mlp__activation'] = []
    if parameters['ActivationRelu'].GetValue() == True:
        param_grid['mlp__activation'].append('relu')
    if parameters['ActivationTanh'].GetValue() == True:
        param_grid['mlp__activation'].append('tanh')
        
    param_grid['mlp__momentum'] = [0.9]
    param_grid['mlp__nesterovs_momentum'] = [True]
    
    param_grid['mlp__alpha'] = []
    for i in range (int(parameters['AlphaStart'].GetValue()), int(parameters['AlphaEnd'].GetValue())+1):
        param_grid['mlp__alpha'].append(10**i)

    save_result_path = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Grid_Search','ann_best_hyperparamaters.txt')

    result = grid_search(main, model, metric, trainX, trainy, param_grid, save_result_path=save_result_path, verbose=True)

	# Define the best model based on grid search result
    ann = Pipeline([('scaler', result.best_params_['scaler']), 
					('ann', MLPRegressor(solver=result.best_params_['mlp__solver'],
							  			 hidden_layer_sizes=result.best_params_['mlp__hidden_layer_sizes'],
							  			 activation=result.best_params_['mlp__activation'],
							  			 momentum=result.best_params_['mlp__momentum'],
							  			 nesterovs_momentum=result.best_params_['mlp__nesterovs_momentum'],
							  			 alpha=result.best_params_['mlp__alpha']))])

    _, test_metrics = fit_model(ann, 'ANN', traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2)


def RF_grid_search(main, parameters, metric, traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2):
	
    model = RandomForestRegressor()

	# Define search space
    param_grid = dict()
    param_grid['n_estimators'] = np.arange(int(parameters['EstimatorStart'].GetValue()), int(parameters['EstimatorEnd'].GetValue())+int(parameters['EstimatorStep'].GetValue()), int(parameters['EstimatorStep'].GetValue()))
    
    param_grid['criterion'] = []
    if parameters['CriterionMse'].GetValue() == True:
        param_grid['criterion'].append('mse')
    if parameters['CriterionMae'].GetValue() == True:
        param_grid['criterion'].append('mae')
        
    param_grid['min_samples_split'] = np.arange(int(parameters['MinSplitStart'].GetValue()), int(parameters['MinSplitEnd'].GetValue())+int(parameters['MinSplitStep'].GetValue()), int(parameters['MinSplitStep'].GetValue()))
    param_grid['max_depth'] = np.arange(int(parameters['DepthStart'].GetValue()), int(parameters['DepthEnd'].GetValue())+int(parameters['DepthStep'].GetValue()), int(parameters['DepthStep'].GetValue()))
    param_grid['min_samples_leaf'] = np.arange(int(parameters['MinLeafStart'].GetValue()), int(parameters['MinLeafEnd'].GetValue())+int(parameters['MinLeafStep'].GetValue()), int(parameters['MinLeafStep'].GetValue()))
    
    param_grid['max_features'] = []
    if parameters['MaxFeatureAuto'].GetValue() == True:
        param_grid['max_features'].append('auto')
    if parameters['MaxFeatureLog2'].GetValue() == True:
        param_grid['max_features'].append('log2')
    if parameters['MaxFeatureSqrt'].GetValue() == True:
        param_grid['max_features'].append('sqrt')
    
    param_grid['bootstrap'] = [True, False]
    if parameters['BootstrapYes'].GetValue() == True:
        param_grid['bootstrap'].append(True)
    if parameters['BootstrapNo'].GetValue() == True:
        param_grid['bootstrap'].append(False)

    save_result_path = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Grid_Search','rf_best_hyperparamaters.txt')

    result = grid_search(main, model, metric, trainX, trainy, param_grid, save_result_path=save_result_path, verbose=True)
	
    print(result)
	# Define the best model based on grid search result
    rf = RandomForestRegressor(n_estimators=result.best_params_['n_estimators'], 
							   criterion=result.best_params_['criterion'], 
							   min_samples_split=result.best_params_['min_samples_split'], 
							   max_depth=result.best_params_['max_depth'], 
							   min_samples_leaf=result.best_params_['min_samples_leaf'], 
							   max_features=result.best_params_['max_features'],
							   bootstrap=result.best_params_['bootstrap'])


    _, test_metrics = fit_model(rf, 'Random Forest', traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2)


def GB_grid_search(main, parameters, metric, traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2):
	
    model = GradientBoostingRegressor()

	# Define search space
    param_grid = dict()
    param_grid['n_estimators'] = np.arange(int(parameters['EstimatorStart'].GetValue()), int(parameters['EstimatorEnd'].GetValue())+int(parameters['EstimatorStep'].GetValue()), int(parameters['EstimatorStep'].GetValue()))
    param_grid['criterion'] = ['friedman_mse']
    
    param_grid['min_samples_split'] = np.arange(int(parameters['MinSplitStart'].GetValue()), int(parameters['MinSplitEnd'].GetValue())+int(parameters['MinSplitStep'].GetValue()), int(parameters['MinSplitStep'].GetValue()))
    param_grid['max_depth'] = np.arange(int(parameters['DepthStart'].GetValue()), int(parameters['DepthEnd'].GetValue())+int(parameters['DepthStep'].GetValue()), int(parameters['DepthStep'].GetValue()))
    param_grid['min_samples_leaf'] = np.arange(int(parameters['MinLeafStart'].GetValue()), int(parameters['MinLeafEnd'].GetValue())+int(parameters['MinLeafStep'].GetValue()), int(parameters['MinLeafStep'].GetValue()))
    
    param_grid['max_features'] = []
    if parameters['MaxFeatureAuto'].GetValue() == True:
        param_grid['max_features'].append('auto')
    if parameters['MaxFeatureLog2'].GetValue() == True:
        param_grid['max_features'].append('log2')
    if parameters['MaxFeatureSqrt'].GetValue() == True:
        param_grid['max_features'].append('sqrt')	

    save_result_path = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Grid_Search','gb_best_hyperparamaters.txt')

    result = grid_search(main, model, metric, trainX, trainy, param_grid, save_result_path=save_result_path, verbose=True)
	
    print(result)

	# Define the best model based on grid search result
    rf = RandomForestRegressor(n_estimators=result.best_params_['n_estimators'], 
							   criterion=result.best_params_['criterion'], 
							   min_samples_split=result.best_params_['min_samples_split'], 
							   max_depth=result.best_params_['max_depth'], 
							   min_samples_leaf=result.best_params_['min_samples_leaf'], 
							   max_features=result.best_params_['max_features'])


    _, test_metrics = fit_model(rf, 'Gradient Boosting', traintest, train, test, trainX, trainy, testX, testy, traintestX, traintesty, cutoff1,cutoff2)

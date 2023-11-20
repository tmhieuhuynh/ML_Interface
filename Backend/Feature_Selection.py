from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import os
from Backend.Metric import get_metrics

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

def get_models(control):

    if control.lr.GetValue() == True:
        return Pipeline([('scaler',scaler(control.RidgeParameters['Scaler'].GetValue())),
                                 ('lr',LinearRegression())])
	
    if control.ridge.GetValue() == True:
        return Pipeline([('scaler',scaler(control.RidgeParameters['Scaler'].GetValue())),
                                    ('ridge', Ridge(solver=control.RidgeParameters['Solver'].GetValue(),
                                                    alpha=10**int(control.RidgeParameters['Alpha'].GetValue()),
                                                    fit_intercept=boolean(control.RidgeParameters['FitIntercept'].GetValue())))])
	
    if control.svr.GetValue() == True:
        return Pipeline([('scaler', scaler(control.SvrParameters['Scaler'].GetValue())),
                                 ('svr', SVR(C= int(control.SvrParameters['C'].GetValue()),
                                             epsilon=int(control.SvrParameters['Epsilon'].GetValue())/100,
                                             gamma= control.SvrParameters['Gamma'].GetValue(),
                                             kernel= control.SvrParameters['Kernel'].GetValue()))])
	
    if control.knn.GetValue() == True:
        return Pipeline([('scaler', scaler(control.KnnParameters['Scaler'].GetValue())), 
    							  ('reduce_dim', PCA(n_components=int(control.KnnParameters['Dim'].GetValue()))), 
    							  ('knn', KNeighborsRegressor(n_neighbors=int(control.KnnParameters['Neighbor'].GetValue()), 
    							  							  weights=control.KnnParameters['Weight'].GetValue(), 
    							  							  algorithm=control.KnnParameters['Algorithm'].GetValue(), 
    							  							  leaf_size=int(control.KnnParameters['Leaf'].GetValue())))])
    
    if control.ann.GetValue() == True:
        return Pipeline([('scaler', scaler(control.AnnParameters['Scaler'].GetValue())), 
							  ('reduce_dim', PCA(n_components=int(control.AnnParameters['Dim'].GetValue()))), 
							  ('mlp', MLPRegressor(solver=control.AnnParameters['Solver'].GetValue(),
							  					   hidden_layer_sizes=hiddenlayer(control.AnnParameters['HiddenLayer1'].GetValue(),control.AnnParameters['HiddenLayer2'].GetValue()), 
							  					   activation=control.AnnParameters['Activation'].GetValue(), 							  					    
							  					   alpha=10**int(control.AnnParameters['Alpha'].GetValue()), 
							  					   max_iter=10**int(control.AnnParameters['Iter'].GetValue()), 
							  					   momentum=int(control.AnnParameters['Momentum'].GetValue())/10, 
							  					   nesterovs_momentum=boolean(control.AnnParameters['Nesterovs'].GetValue())))])

    if control.rf.GetValue() == True:
        return RandomForestRegressor(n_estimators=int(control.RfParameters['Estimator'].GetValue()), 
											 criterion=control.RfParameters['Criterion'].GetValue(), 
											 min_samples_split=int(control.RfParameters['MinSplit'].GetValue()), 
											 max_depth=int(control.RfParameters['Depth'].GetValue()), 
											 min_samples_leaf=int(control.RfParameters['MinLeaf'].GetValue()), 
											 max_features=control.RfParameters['MaxFeature'].GetValue(),
											 bootstrap=boolean(control.RfParameters['Bootstrap'].GetValue()))

    if control.gb.GetValue() == True:
        return GradientBoostingRegressor(n_estimators=int(control.GbParameters['Estimator'].GetValue()), 
											   max_depth=int(control.GbParameters['Depth'].GetValue()), 
											   learning_rate=int(control.GbParameters['Learning'].GetValue())/1000, 
											   criterion=control.GbParameters['Criterion'].GetValue(), 
											   min_samples_split=int(control.GbParameters['MinSplit'].GetValue()), 
											   min_samples_leaf=int(control.GbParameters['MinLeaf'].GetValue()), 
											   max_features=control.GbParameters['MaxFeature'].GetValue())
	
def Choose(array):
    return array.count(-999999)

def Metric(a):
    return ['MAPE','negative MAE', 'RMSE', 'Pearson R', 'P1 Sigma', 'P2 Sigma'].index(a)

def Feature_Backward_Selection(control, trainX,trainy,testX,testy,traintest,cutoff1,cutoff2,Yfeatures,Xfeatures,Subfeatures,InputName):
    n_step=int(control.SelectionParameters['NStep'].GetValue())
    model=get_models(control)
    features=Yfeatures+Xfeatures    
    ban=['#']
    high=[-999999]
    data=[['-']]
    k=0
    if control.ann.GetValue() == True:
        k = int(control.AnnParameters['Dim'].GetValue())-1
    elif control.knn.GetValue() == True:
        k = int(control.KnnParameters['Dim'].GetValue())-1
    
    
    for i in range (0,len(Xfeatures)):
        data[0].append(Xfeatures[i])
        
    while len(ban)+n_step<len(features)-k:
        metric=[]
        for i in range(0,len(features)):
            if features[i] in ban:
                metric.append(-999999)
            else:
                print(i,'/',len(features)-1-k,' ',len(ban),'/',len(features)-1-k)
                if i!=0:
                    model.fit(trainX.drop(ban[1:]+[features[i]],axis=1), trainy)
                    test_pred = model.predict(testX.drop(ban[1:]+[features[i]],axis=1))
                else:
                    model.fit(trainX.drop(ban[1:],axis=1), trainy)
                    test_pred = model.predict(testX.drop(ban[1:],axis=1))
                
                test_metrics = get_metrics(test_pred, testy, cutoff1, cutoff2)
                metric.append(test_metrics[Metric(control.SelectionParameters['Metric'].GetValue())])
                
        data.append(metric.copy())
        
        for i in range (0,n_step):       
            highest=max(metric)
            if features[metric.index(highest)]==Yfeatures[0]:
                metric[metric.index(highest)]=-999999
                highest=max(metric)
            high.append(highest)
            ban.append(features[metric.index(highest)])
            metric[metric.index(highest)]=-999999

    print(max(high))
    temp= traintest.sort_index().drop(ban[1:high.index(max(high))+1],axis=1)
    
    temp.to_csv(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Feature_Selection','Backward_Selected_Feature_'+InputName),index=False)
    
    sub=[]
    for x in range (0,len(data[0])):
        part=[]
        for y in range (0,len(data)):
            part.append(data[y][x])
        sub.append(part)
    
    sub.sort(key=Choose)
    
    for x in range (0,len(sub)):
        for y in range (0,len(sub[0])):
            if sub[x][y]==-999999:
                sub[x][y]='-'
              
    label=['#','-']
    for i in range (1,len(ban)-n_step,n_step):
        part=ban[i]
        for x in range (i+1,i+n_step):
            part+=' - '+ban[x]
        label.append(part)       
    table=DataFrame(sub,columns=label)
    
    table.to_excel(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Feature_Selection', 'FBS_Metric_Table.xlsx'),index=False)


def Feature_Forward_Selection(control, trainX,trainy,testX,testy,traintest,cutoff1,cutoff2,Yfeatures,Xfeatures,Subfeatures,InputName):
    n_step=int(control.SelectionParameters['NStep'].GetValue())
    model=get_models(control)
    features=Yfeatures+Xfeatures    
    chosen=['#']
    high=[-999999]
    data=[['-']]
    k=0
    if control.ann.GetValue() == True:
        k = int(control.AnnParameters['Dim'].GetValue())-1
    elif control.knn.GetValue() == True:
        k = int(control.KnnParameters['Dim'].GetValue())-1
        
    for i in range (0,len(Xfeatures)):
        data[0].append(Xfeatures[i])
        
    while len(chosen)+n_step<len(features)-k:
        metric=[]
        for i in range(0,len(features)):
            if features[i] in chosen:
                metric.append(-999999)
            else:
                print(i,'/',len(features)-1-k,' ',len(chosen),'/',len(features)-1-k)
                if i==0 and len(chosen)==1:
                    metric.append(-99999)
                else:
                    if i!=0:
                        model.fit(trainX[chosen[1:]+[features[i]]], trainy)
                        test_pred = model.predict(testX[chosen[1:]+[features[i]]])
                    else:
                        model.fit(trainX[chosen[1:]], trainy)
                        test_pred = model.predict(testX[chosen[1:]])
                    
                    test_metrics = get_metrics(test_pred, testy, cutoff1, cutoff2)
                    metric.append(test_metrics[Metric(control.SelectionParameters['Metric'].GetValue())])
                
        data.append(metric.copy())
        
        for i in range (0,n_step):       
            highest=max(metric)
            if features[metric.index(highest)]==Yfeatures[0]:
                metric[metric.index(highest)]=-999999
                highest=max(metric)
            high.append(highest)
            chosen.append(features[metric.index(highest)])
            metric[metric.index(highest)]=-999999

    print(max(high))
    temp= traintest.sort_index()[Subfeatures+Yfeatures+chosen[1:high.index(max(high))+1]]
    
    temp.to_csv(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Feature_Selection','Forward_Selected_Feature_'+InputName),index=False)
    
    sub=[]
    for x in range (0,len(data[0])):
        part=[]
        for y in range (0,len(data)):
            part.append(data[y][x])
        sub.append(part)
    
    sub.sort(key=Choose)
    
    for x in range (0,len(sub)):
        for y in range (0,len(sub[0])):
            if sub[x][y]==-999999 or sub[x][y]==-99999:
                sub[x][y]='-'
              
    label=['#','-']
    for i in range (1,len(chosen)-n_step,n_step):
        part=chosen[i]
        for x in range (i+1,i+n_step):
            part+=' - '+chosen[x]
        label.append(part)       
    table=DataFrame(sub,columns=label)
    
    table.to_excel(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Result','Feature_Selection', 'FFS_Metric_Table.xlsx'),index=False)

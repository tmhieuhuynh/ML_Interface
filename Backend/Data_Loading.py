import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def binvalue(data_path,target,split_ratio):
    a=np.sort(pd.read_csv(data_path)[target])
    variables=[a[0]-abs(a[0])/100]
    for i in range (1,split_ratio):
        variables.append(a[int(len(a)*i/split_ratio)])
    variables.append(np.inf)
    return variables

def DataLoading(data_path, split_ratio, Yfeatures, Xfeatures,Subfeatures):
    temp= pd.concat([pd.read_csv(data_path, usecols =[x for x in Subfeatures]),pd.read_csv(data_path, usecols =[x for x in Yfeatures])],axis=1)
    for i in Xfeatures:
        temp= pd.concat([temp,pd.read_csv(data_path, usecols = [x for x in [i]])],axis=1)
    sub=pd.cut(temp[Yfeatures[0]],bins=binvalue(data_path,Yfeatures[0],split_ratio),labels=np.arange(1,split_ratio+1,1))
    split=StratifiedShuffleSplit(n_splits=1,test_size=1/split_ratio,random_state=22)
    for train_index,test_index in split.split(temp,sub):
        strat_train_set=temp.loc[train_index]
        strat_test_set=temp.loc[test_index]
    
    train = strat_train_set
    test = strat_test_set
    
    trainX = train.iloc[:,len(Subfeatures)+1:]
    trainy = train.iloc[:,len(Subfeatures)]
    
    testX = test.iloc[:,len(Subfeatures)+1:]
    testy = test.iloc[:,len(Subfeatures)]	
    
    traintest = pd.concat((train,test),axis=0)
    traintestX = traintest.iloc[:, len(Subfeatures)+1:]
    traintesty = traintest.iloc[:, len(Subfeatures)]
    
    cutoff1 = temp[Yfeatures[0]].std(axis=0)
    cutoff2 = cutoff1*2
    
    return train, test, trainX, trainy, testX, testy, traintest, traintestX, traintesty, cutoff1, cutoff2

def DataLoading2(train_path, test_path, Yfeatures, Xfeatures,Subfeatures):
    train= pd.concat([pd.read_csv(train_path, usecols =[x for x in Subfeatures]),pd.read_csv(train_path, usecols =[x for x in Yfeatures])],axis=1)
    for i in Xfeatures:
        train= pd.concat([train,pd.read_csv(train_path, usecols = [x for x in [i]])],axis=1)
    test= pd.concat([pd.read_csv(test_path, usecols =[x for x in Subfeatures]),pd.read_csv(test_path, usecols =[x for x in Yfeatures])],axis=1)
    for i in Xfeatures:
        test= pd.concat([test,pd.read_csv(test_path, usecols = [x for x in [i]])],axis=1)
        
    trainX = train.iloc[:,len(Subfeatures)+1:]
    trainy = train.iloc[:,len(Subfeatures)]
    
    testX = test.iloc[:,len(Subfeatures)+1:]
    testy = test.iloc[:,len(Subfeatures)]	
    
    traintest = pd.concat((train,test),axis=0)	
    traintestX = traintest.iloc[:, len(Subfeatures)+1:]
    traintesty = traintest.iloc[:, len(Subfeatures)]
    
    cutoff1 = traintest[Yfeatures[0]].std(axis=0)
    cutoff2 = cutoff1*2
    
    return train, test, trainX, trainy, testX, testy, traintest, traintestX, traintesty, cutoff1, cutoff2

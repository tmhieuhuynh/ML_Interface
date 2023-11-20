import wx

def Show(dic,boolean):
    for i in dic:
        dic[i].Show(boolean)

class Grid_Search(wx.Panel):
    
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        
        wx.StaticBox(self, pos=(5, 5), size=(485, 50))
        wx.StaticText(self,label='Metric:',pos=(10,25))
        metric=['explained_variance','max_error','neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error','neg_mean_squared_log_error','neg_median_absolute_error','r2','neg_mean_poisson_deviance','neg_mean_gamma_deviance','neg_mean_absolute_percentage_error']
        self.MetricBox = wx.ComboBox(self,value='neg_mean_absolute_error', pos=(50, 20), choices=metric)
        
        self.RepeatedParameters={}
        wx.StaticText(self, label='n_splits:', pos=(283, 25))
        self.RepeatedParameters['NSplit'] = wx.SpinCtrl(self, value='5', pos=(328, 20), size=(50, -1), min=0, max=100)
        
        wx.StaticText(self, label='n_repeats:', pos=(383, 25))
        self.RepeatedParameters['NRepeat'] = wx.SpinCtrl(self, value='6', pos=(438, 20), size=(50, -1), min=0, max=100)
        
        wx.StaticBox(self, label='Model Selection', pos=(5, 60), size=(140, 230))
        
        #Ridge
        self.ridge = wx.CheckBox(self, label='Ridge', pos=(15, 85))
        self.ridge.Bind(wx.EVT_CHECKBOX,self.Ridge)
        
        self.RidgeParameters={}
        self.RidgeParameters['ScalerText'] = wx.StaticBox(self, label='Scaler:', pos=(180, 55),size=(150,50))
        self.RidgeParameters['ScalerMinMax'] = wx.CheckBox(self, label='MinMax', pos=(190, 70))
        self.RidgeParameters['ScalerMinMax'].Value=True
        self.RidgeParameters['ScalerStandard'] = wx.CheckBox(self, label='Standard', pos=(260, 70))
        self.RidgeParameters['ScalerStandard'].Value=True
        self.RidgeParameters['ScalerRobust'] = wx.CheckBox(self, label='Robust', pos=(190, 85))
        self.RidgeParameters['ScalerRobust'].Value=True
        self.RidgeParameters['ScalerNone'] = wx.CheckBox(self, label='Nọne', pos=(260, 85))
        self.RidgeParameters['ScalerNone'].Value=True
        
        self.RidgeParameters['SolverText'] = wx.StaticBox(self, label='Solver:', pos=(180, 105),size=(150,50))
        self.RidgeParameters['SolverSvd'] = wx.CheckBox(self, label='svd', pos=(190, 120))
        self.RidgeParameters['SolverSvd'].Value=True
        self.RidgeParameters['SolverCholesky'] = wx.CheckBox(self, label='cholesky', pos=(260, 120))
        self.RidgeParameters['SolverCholesky'].Value=True
        self.RidgeParameters['SolverLsqr'] = wx.CheckBox(self, label='lsqr', pos=(190, 135))
        self.RidgeParameters['SolverLsqr'].Value=True
        self.RidgeParameters['SolverSag'] = wx.CheckBox(self, label='sag', pos=(260, 135))
        self.RidgeParameters['SolverSag'].Value=True
        
        self.RidgeParameters['AlphaText'] = wx.StaticBox(self, label='Logarit of Alpha:', pos=(180, 155),size=(150,60))
        self.RidgeParameters['AlphaStart'] = wx.SpinCtrl(self, value='-5', pos=(190, 180), size=(50, -1), min=-100, max=100)
        self.RidgeParameters['AlphaSubText'] = wx.StaticText(self, label='__', pos=(250, 177))
        self.RidgeParameters['AlphaEnd'] = wx.SpinCtrl(self, value='2', pos=(270, 180), size=(50, -1), min=-100, max=100)
        
        self.RidgeParameters['FitInterceptText'] = wx.StaticBox(self, label='Fit Interception', pos=(180, 215),size=(150,35))
        self.RidgeParameters['FitInterceptYes'] = wx.CheckBox(self, label='Yes', pos=(190, 230))
        self.RidgeParameters['FitInterceptYes'].Value=True
        self.RidgeParameters['FitInterceptNo'] = wx.CheckBox(self, label='No', pos=(260, 230))
        self.RidgeParameters['FitInterceptNo'].Value=True
        
        Show(self.RidgeParameters,False)
        
        #KNN
        self.knn = wx.CheckBox(self, label='KNN', pos=(15, 120))
        self.knn.Bind(wx.EVT_CHECKBOX,self.Knn)
        
        self.KnnParameters={}
        self.KnnParameters['ScalerText'] = wx.StaticBox(self, label='Scaler:', pos=(180, 55),size=(150,50))
        self.KnnParameters['ScalerMinMax'] = wx.CheckBox(self, label='MinMax', pos=(190, 70))
        self.KnnParameters['ScalerMinMax'].Value=True
        self.KnnParameters['ScalerStandard'] = wx.CheckBox(self, label='Standard', pos=(260, 70))
        self.KnnParameters['ScalerStandard'].Value=True
        self.KnnParameters['ScalerRobust'] = wx.CheckBox(self, label='Robust', pos=(190, 85))
        self.KnnParameters['ScalerRobust'].Value=True
        self.KnnParameters['ScalerNone'] = wx.CheckBox(self, label='None', pos=(260, 85))
        self.KnnParameters['ScalerNone'].Value=True
        
        self.KnnParameters['DimText'] = wx.StaticBox(self, label='Reduce Dim:', pos=(180, 105),size=(150,50))
        self.KnnParameters['DimStart'] = wx.SpinCtrl(self, value='4', pos=(190, 125), size=(50, -1), min=0, max=100)
        self.KnnParameters['DimSubText'] = wx.StaticText(self, label='__', pos=(250, 122))
        self.KnnParameters['DimEnd'] = wx.SpinCtrl(self, value='20', pos=(270, 125), size=(50, -1), min=0, max=100)
        
        self.KnnParameters['NeighborText'] = wx.StaticBox(self, label='Number of Neighbors:', pos=(180, 155),size=(150,75))
        self.KnnParameters['NeighborStart'] = wx.SpinCtrl(self, value='7', pos=(190, 175), size=(50, -1), min=1, max=100)
        self.KnnParameters['NeighborSubText1'] = wx.StaticText(self, label='__', pos=(250, 172))
        self.KnnParameters['NeighborEnd'] = wx.SpinCtrl(self, value='24', pos=(270, 175), size=(50, -1), min=1, max=100)
        self.KnnParameters['NeighborSubText2'] = wx.StaticText(self, label='Step:', pos=(190, 205))
        self.KnnParameters['NeighborStep'] = wx.SpinCtrl(self, value='1', pos=(230, 200), size=(50, -1), min=1, max=100)
        
        self.KnnParameters['LeafText'] = wx.StaticBox(self, label='Leaf Size:', pos=(180, 230),size=(150,75))
        self.KnnParameters['LeafStart'] = wx.SpinCtrl(self, value='20', pos=(190, 250), size=(50, -1), min=1, max=100)
        self.KnnParameters['LeafSubText1'] = wx.StaticText(self, label='__', pos=(250, 247))
        self.KnnParameters['LeafEnd'] = wx.SpinCtrl(self, value='40', pos=(270, 250), size=(50, -1), min=1, max=100)
        self.KnnParameters['LeafSubText2'] = wx.StaticText(self, label='Step:', pos=(190, 280))
        self.KnnParameters['LeafStep'] = wx.SpinCtrl(self, value='10', pos=(230, 275), size=(50, -1), min=1, max=100)

        Show(self.KnnParameters,False)
        
        #SVR
        self.svr = wx.CheckBox(self, label='SVR', pos=(15, 155))
        self.svr.Bind(wx.EVT_CHECKBOX,self.Svr)
        
        self.SvrParameters={}
        self.SvrParameters['ScalerText'] = wx.StaticBox(self, label='Scaler:', pos=(180, 55),size=(150,50))
        self.SvrParameters['ScalerMinMax'] = wx.CheckBox(self, label='MinMax', pos=(190, 70))
        self.SvrParameters['ScalerMinMax'].Value=True
        self.SvrParameters['ScalerStandard'] = wx.CheckBox(self, label='Standard', pos=(260, 70))
        self.SvrParameters['ScalerStandard'].Value=True
        self.SvrParameters['ScalerRobust'] = wx.CheckBox(self, label='Robust', pos=(190, 85))
        self.SvrParameters['ScalerRobust'].Value=True
        self.SvrParameters['ScalerNone'] = wx.CheckBox(self, label='None', pos=(260, 85))
        self.SvrParameters['ScalerNone'].Value=True
        
        self.SvrParameters['KernelText'] = wx.StaticBox(self, label='Kernel:', pos=(180, 105),size=(150,65))
        self.SvrParameters['KernelLinear'] = wx.CheckBox(self, label='Linear', pos=(190, 120))
        self.SvrParameters['KernelLinear'].Value=True
        self.SvrParameters['KernelPoly'] = wx.CheckBox(self, label='Poly', pos=(260, 120))
        self.SvrParameters['KernelPoly'].Value=True
        self.SvrParameters['KernelRbf'] = wx.CheckBox(self, label='Rbf', pos=(190, 135))
        self.SvrParameters['KernelRbf'].Value=True
        self.SvrParameters['KernelSigmoid'] = wx.CheckBox(self, label='Sigmoid', pos=(260, 135))
        self.SvrParameters['KernelSigmoid'].Value=True
        self.SvrParameters['KernelPrecomputed'] = wx.CheckBox(self, label='Precomputed', pos=(190, 150))
        self.SvrParameters['KernelPrecomputed'].Value=True
        
        self.SvrParameters['GammaText'] = wx.StaticBox(self, label='Gamma:', pos=(180, 170),size=(150,35))
        self.SvrParameters['GammaScale'] = wx.CheckBox(self, label='Scale', pos=(190, 185))
        self.SvrParameters['GammaScale'].Value=True
        self.SvrParameters['GammaAuto'] = wx.CheckBox(self, label='Auto', pos=(260, 185))
        self.SvrParameters['GammaAuto'].Value=True
        
        self.SvrParameters['CText'] = wx.StaticBox(self, label='C:', pos=(340, 55),size=(150,50))
        self.SvrParameters['CStart'] = wx.SpinCtrl(self, value='1', pos=(350, 75), size=(50, -1), min=-100, max=100)
        self.SvrParameters['CSubText'] = wx.StaticText(self, label='__', pos=(410, 72))
        self.SvrParameters['CEnd'] = wx.SpinCtrl(self, value='5', pos=(430, 75), size=(50, -1), min=-100, max=100)
        
        self.SvrParameters['EpsilonText'] = wx.StaticBox(self, label='100 x Epsilon:', pos=(180, 205),size=(150,75))
        self.SvrParameters['EpsilonStart'] = wx.SpinCtrl(self, value='5', pos=(190, 225), size=(50, -1), min=1, max=100)
        self.SvrParameters['EpsilonSubText1'] = wx.StaticText(self, label='__', pos=(250, 222))
        self.SvrParameters['EpsilonEnd'] = wx.SpinCtrl(self, value='20', pos=(270, 225), size=(50, -1), min=1, max=100)
        self.SvrParameters['EpsilonSubText2'] = wx.StaticText(self, label='Step:', pos=(190, 255))
        self.SvrParameters['EpsilonStep'] = wx.SpinCtrl(self, value='5', pos=(230, 250), size=(50, -1), min=1, max=100)
        
        Show(self.SvrParameters,False)
        
        #ANN
        self.ann = wx.CheckBox(self, label='ANN', pos=(15, 190))
        self.ann.Bind(wx.EVT_CHECKBOX,self.Ann)
        
        self.AnnParameters={}
        self.AnnParameters['ScalerText'] = wx.StaticBox(self, label='Scaler:', pos=(180, 55),size=(150,50))
        self.AnnParameters['ScalerMinMax'] = wx.CheckBox(self, label='MinMax', pos=(190, 70))
        self.AnnParameters['ScalerMinMax'].Value=True
        self.AnnParameters['ScalerStandard'] = wx.CheckBox(self, label='Standard', pos=(260, 70))
        self.AnnParameters['ScalerStandard'].Value=True
        self.AnnParameters['ScalerRobust'] = wx.CheckBox(self, label='Robust', pos=(190, 85))
        self.AnnParameters['ScalerRobust'].Value=True
        self.AnnParameters['ScalerNone'] = wx.CheckBox(self, label='None', pos=(260, 85))
        self.AnnParameters['ScalerNone'].Value=True
        
        self.AnnParameters['DimText'] = wx.StaticBox(self, label='Reduce Dim:', pos=(180, 105),size=(150,50))
        self.AnnParameters['DimStart'] = wx.SpinCtrl(self, value='4', pos=(190, 125), size=(50, -1), min=0, max=100)
        self.AnnParameters['DimSubText'] = wx.StaticText(self, label='__', pos=(250, 122))
        self.AnnParameters['DimEnd'] = wx.SpinCtrl(self, value='20', pos=(270, 125), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['SolverText'] = wx.StaticBox(self, label='Solver:', pos=(180, 155),size=(150,35))        
        self.AnnParameters['SolverAdam'] = wx.CheckBox(self, label='adam', pos=(190, 170))
        self.AnnParameters['SolverAdam'].Value=True
        self.AnnParameters['SolverLbfgs'] = wx.CheckBox(self, label='lbfgs', pos=(260, 170))
        self.AnnParameters['SolverLbfgs'].Value=True
        
        self.AnnParameters['HiddenLayerText'] = wx.StaticBox(self, label='Hidden Layer Size:', pos=(180, 190),size=(150,75))
        self.AnnParameters['HiddenLayerStart'] = wx.SpinCtrl(self, value='15', pos=(190, 210), size=(50, -1), min=1, max=100)
        self.AnnParameters['HiddenLayerSubText1'] = wx.StaticText(self, label='__', pos=(250, 207))
        self.AnnParameters['HiddenLayerEnd'] = wx.SpinCtrl(self, value='30', pos=(270, 210), size=(50, -1), min=1, max=100)
        self.AnnParameters['HiddenLayerSubText2'] = wx.StaticText(self, label='Step:', pos=(190, 240))
        self.AnnParameters['HiddenLayerStep'] = wx.SpinCtrl(self, value='5', pos=(230, 235), size=(50, -1), min=1, max=100)
        
        self.AnnParameters['ActivationText'] = wx.StaticBox(self, label='Activation:', pos=(180, 265),size=(150,35))        
        self.AnnParameters['ActivationRelu'] = wx.CheckBox(self, label='Relu', pos=(190, 280))
        self.AnnParameters['ActivationRelu'].Value=True
        self.AnnParameters['ActivationTanh'] = wx.CheckBox(self, label='Tanh', pos=(260, 280))
        self.AnnParameters['ActivationTanh'].Value=True
        
        self.AnnParameters['AlphaText'] = wx.StaticBox(self, label='Logarit of Alpha:', pos=(340, 55),size=(150,50))
        self.AnnParameters['AlphaStart'] = wx.SpinCtrl(self, value='-3', pos=(350, 75), size=(50, -1), min=-100, max=100)
        self.AnnParameters['AlphaSubText'] = wx.StaticText(self, label='__', pos=(410, 72))
        self.AnnParameters['AlphaEnd'] = wx.SpinCtrl(self, value='-1', pos=(430, 75), size=(50, -1), min=-100, max=100)
        
        Show(self.AnnParameters,False)
        
        #Random Forest
        self.rf = wx.CheckBox(self, label='RF', pos=(15, 225))
        self.rf.Bind(wx.EVT_CHECKBOX,self.Rf)
        
        self.RfParameters={}
        self.RfParameters['EstimatorText'] = wx.StaticBox(self, label='Number of Estimator:', pos=(180, 55),size=(150,75))
        self.RfParameters['EstimatorStart'] = wx.SpinCtrl(self, value='140', pos=(190, 75), size=(50, -1), min=1, max=100)
        self.RfParameters['EstimatorSubText1'] = wx.StaticText(self, label='__', pos=(250, 72))
        self.RfParameters['EstimatorEnd'] = wx.SpinCtrl(self, value='350', pos=(270, 75), size=(50, -1), min=1, max=100)
        self.RfParameters['EstimatorSubText2'] = wx.StaticText(self, label='Step:', pos=(190, 105))
        self.RfParameters['EstimatorStep'] = wx.SpinCtrl(self, value='50', pos=(230, 100), size=(50, -1), min=1, max=100)
        
        self.RfParameters['CriterionText'] = wx.StaticBox(self, label='Criterion:', pos=(180, 130),size=(150,35))        
        self.RfParameters['CriterionMse'] = wx.CheckBox(self, label='mse', pos=(190, 145))
        self.RfParameters['CriterionMse'].Value=True
        self.RfParameters['CriterionMae'] = wx.CheckBox(self, label='mae', pos=(260, 145))
        self.RfParameters['CriterionMae'].Value=True
        
        self.RfParameters['MinSplitText'] = wx.StaticBox(self, label='Min Sample Split:', pos=(180, 165),size=(150,75))
        self.RfParameters['MinSplitStart'] = wx.SpinCtrl(self, value='2', pos=(190, 185), size=(50, -1), min=1, max=100)
        self.RfParameters['MinSplitSubText1'] = wx.StaticText(self, label='__', pos=(250, 182))
        self.RfParameters['MinSplitEnd'] = wx.SpinCtrl(self, value='6', pos=(270, 185), size=(50, -1), min=1, max=100)
        self.RfParameters['MinSplitSubText2'] = wx.StaticText(self, label='Step:', pos=(190, 215))
        self.RfParameters['MinSplitStep'] = wx.SpinCtrl(self, value='1', pos=(230, 210), size=(50, -1), min=1, max=100)
        
        self.RfParameters['DepthText'] = wx.StaticBox(self, label='Max Depth:', pos=(180, 240),size=(150,75))
        self.RfParameters['DepthStart'] = wx.SpinCtrl(self, value='6', pos=(190, 260), size=(50, -1), min=1, max=100)
        self.RfParameters['DepthSubText1'] = wx.StaticText(self, label='__', pos=(250, 257))
        self.RfParameters['DepthEnd'] = wx.SpinCtrl(self, value='10', pos=(270, 260), size=(50, -1), min=1, max=100)
        self.RfParameters['DepthSubText2'] = wx.StaticText(self, label='Step:', pos=(190, 290))
        self.RfParameters['DepthStep'] = wx.SpinCtrl(self, value='1', pos=(230, 285), size=(50, -1), min=1, max=100)
        
        self.RfParameters['MinLeafText'] = wx.StaticBox(self, label='Min Sample Leaf:', pos=(340, 55),size=(150,75))
        self.RfParameters['MinLeafStart'] = wx.SpinCtrl(self, value='1', pos=(350, 75), size=(50, -1), min=1, max=100)
        self.RfParameters['MinLeafSubText1'] = wx.StaticText(self, label='__', pos=(410, 72))
        self.RfParameters['MinLeafEnd'] = wx.SpinCtrl(self, value='4', pos=(430, 75), size=(50, -1), min=1, max=100)
        self.RfParameters['MinLeafSubText2'] = wx.StaticText(self, label='Step:', pos=(350, 105))
        self.RfParameters['MinLeafStep'] = wx.SpinCtrl(self, value='1', pos=(390, 100), size=(50, -1), min=1, max=100)
        
        self.RfParameters['MaxFeatureText'] = wx.StaticBox(self, label='Max Features:', pos=(340, 130),size=(150,50))
        self.RfParameters['MaxFeatureAuto'] = wx.CheckBox(self, label='Auto', pos=(350, 145))
        self.RfParameters['MaxFeatureAuto'].Value=True
        self.RfParameters['MaxFeatureLog2'] = wx.CheckBox(self, label='Log2', pos=(420, 145))
        self.RfParameters['MaxFeatureLog2'].Value=True
        self.RfParameters['MaxFeatureSqrt'] = wx.CheckBox(self, label='Sqrt', pos=(350, 160))
        self.RfParameters['MaxFeatureSqrt'].Value=True
        
        self.RfParameters['BootstrapText'] = wx.StaticBox(self, label='Bootstrap:', pos=(340, 180),size=(150,35))
        self.RfParameters['BootstrapYes'] = wx.CheckBox(self, label='Yes', pos=(350, 195))
        self.RfParameters['BootstrapYes'].Value=True
        self.RfParameters['BootstrapNo'] = wx.CheckBox(self, label='No', pos=(420, 195))
        self.RfParameters['BootstrapNo'].Value=True
        
        Show(self.RfParameters,False)
        
        #Gradient Boosting
        self.gb = wx.CheckBox(self, label='GB', pos=(15, 260))
        self.gb.Bind(wx.EVT_CHECKBOX,self.Gb)
        
        self.GbParameters={}
        self.GbParameters['EstimatorText'] = wx.StaticBox(self, label='Number of Estimator:', pos=(180, 55),size=(150,75))
        self.GbParameters['EstimatorStart'] = wx.SpinCtrl(self, value='100', pos=(190, 75), size=(50, -1), min=1, max=100)
        self.GbParameters['EstimatorSubText1'] = wx.StaticText(self, label='__', pos=(250, 72))
        self.GbParameters['EstimatorEnd'] = wx.SpinCtrl(self, value='300', pos=(270, 75), size=(50, -1), min=1, max=100)
        self.GbParameters['EstimatorSubText2'] = wx.StaticText(self, label='Step:', pos=(190, 105))
        self.GbParameters['EstimatorStep'] = wx.SpinCtrl(self, value='20', pos=(230, 100), size=(50, -1), min=1, max=100)
        
        self.GbParameters['MinSplitText'] = wx.StaticBox(self, label='Min Sample Split:', pos=(180, 130),size=(150,75))
        self.GbParameters['MinSplitStart'] = wx.SpinCtrl(self, value='2', pos=(190, 150), size=(50, -1), min=1, max=100)
        self.GbParameters['MinSplitSubText1'] = wx.StaticText(self, label='__', pos=(250, 147))
        self.GbParameters['MinSplitEnd'] = wx.SpinCtrl(self, value='6', pos=(270, 150), size=(50, -1), min=1, max=100)
        self.GbParameters['MinSplitSubText2'] = wx.StaticText(self, label='Step:', pos=(190, 180))
        self.GbParameters['MinSplitStep'] = wx.SpinCtrl(self, value='1', pos=(230, 175), size=(50, -1), min=1, max=100)
        
        self.GbParameters['DepthText'] = wx.StaticBox(self, label='Max Depth:', pos=(180, 205),size=(150,75))
        self.GbParameters['DepthStart'] = wx.SpinCtrl(self, value='6', pos=(190, 225), size=(50, -1), min=1, max=100)
        self.GbParameters['DepthSubText1'] = wx.StaticText(self, label='__', pos=(250, 222))
        self.GbParameters['DepthEnd'] = wx.SpinCtrl(self, value='13', pos=(270, 225), size=(50, -1), min=1, max=100)
        self.GbParameters['DepthSubText2'] = wx.StaticText(self, label='Step:', pos=(190, 255))
        self.GbParameters['DepthStep'] = wx.SpinCtrl(self, value='1', pos=(230, 250), size=(50, -1), min=1, max=100)
        
        self.GbParameters['MinLeafText'] = wx.StaticBox(self, label='Min Sample Leaf:', pos=(340, 55),size=(150,75))
        self.GbParameters['MinLeafStart'] = wx.SpinCtrl(self, value='1', pos=(350, 75), size=(50, -1), min=1, max=100)
        self.GbParameters['MinLeafSubText1'] = wx.StaticText(self, label='__', pos=(410, 72))
        self.GbParameters['MinLeafEnd'] = wx.SpinCtrl(self, value='4', pos=(430, 75), size=(50, -1), min=1, max=100)
        self.GbParameters['MinLeafSubText2'] = wx.StaticText(self, label='Step:', pos=(350, 105))
        self.GbParameters['MinLeafStep'] = wx.SpinCtrl(self, value='1', pos=(390, 100), size=(50, -1), min=1, max=100)
        
        self.GbParameters['MaxFeatureText'] = wx.StaticBox(self, label='Max Features:', pos=(340, 130),size=(150,50))
        self.GbParameters['MaxFeatureAuto'] = wx.CheckBox(self, label='Auto', pos=(350, 145))
        self.GbParameters['MaxFeatureAuto'].Value=True
        self.GbParameters['MaxFeatureLog2'] = wx.CheckBox(self, label='Log2', pos=(420, 145))
        self.GbParameters['MaxFeatureLog2'].Value=True
        self.GbParameters['MaxFeatureSqrt'] = wx.CheckBox(self, label='Sqrt', pos=(350, 160))
        self.GbParameters['MaxFeatureSqrt'].Value=True
        
        Show(self.GbParameters,False)
        
        
    #Function
    def Ridge(self,e):
        Show(self.RidgeParameters,True)
        Show(self.KnnParameters,False)
        Show(self.SvrParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
    def Knn(self,e):
        Show(self.RidgeParameters,False)
        Show(self.KnnParameters,True)
        Show(self.SvrParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
    def Svr(self,e):
        Show(self.RidgeParameters,False)
        Show(self.KnnParameters,False)
        Show(self.SvrParameters,True)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
       
    def Ann(self,e):
        Show(self.RidgeParameters,False)
        Show(self.KnnParameters,False)
        Show(self.SvrParameters,False)
        Show(self.AnnParameters,True)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
    def Rf(self,e):
        Show(self.RidgeParameters,False)
        Show(self.KnnParameters,False)
        Show(self.SvrParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,True)
        Show(self.GbParameters,False)
        
    def Gb(self,e):
        Show(self.RidgeParameters,False)
        Show(self.KnnParameters,False)
        Show(self.SvrParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,True)

class Train_Evaluation(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        
        wx.StaticBox(self, label='Model', pos=(5, 5), size=(180, 300))
        
        #Linear Regression
        self.lr = wx.CheckBox(self, label='Linear Regression', pos=(15, 30))
        self.lr.Value=False
        self.lr.Bind(wx.EVT_CHECKBOX,self.Lr)
        
        self.LrParameters={}
        self.LrParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 5), size=(120, 50))
        self.LrParameters['Scaler'] = wx.ComboBox(self,value='None', pos=(250, 25), choices=['MinMax','Standard','Robust','None'])
        
        Show(self.LrParameters,False)
        
        #Ridge
        self.ridge = wx.CheckBox(self, label='Ridge Regression', pos=(15, 60))
        self.ridge.Value=False
        self.ridge.Bind(wx.EVT_CHECKBOX,self.Ridge)
        
        self.RidgeParameters={}
        self.RidgeParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 5), size=(120, 50))
        self.RidgeParameters['Scaler'] = wx.ComboBox(self,value='None', pos=(250, 25), choices=['MinMax','Standard','Robust','None'])
        
        self.RidgeParameters['SolverText'] = wx.StaticBox(self, label='Solver', pos=(240, 55), size=(120, 50))
        self.RidgeParameters['Solver'] = wx.ComboBox(self,value='svd', pos=(250, 75), choices=['svd','cholesky','lsqr','sag'])
        
        self.RidgeParameters['AlphaText'] = wx.StaticBox(self, label='Logarit of Alpha', pos=(240, 105), size=(120, 50))
        self.RidgeParameters['Alpha'] = wx.SpinCtrl(self, value='10', pos=(250, 125), size=(50, -1), min=-100, max=100)
        
        self.RidgeParameters['FitInterceptText'] = wx.StaticBox(self, label='Fit Intercept', pos=(240, 155), size=(120, 50))
        self.RidgeParameters['FitIntercept'] = wx.ComboBox(self,value='No', pos=(250, 175), choices=['Yes','No'])
        
        Show(self.RidgeParameters,False)
        
        #SVR
        self.svr = wx.CheckBox(self, label='Support Vector Regression', pos=(15, 90))
        self.svr.Value=False
        self.svr.Bind(wx.EVT_CHECKBOX,self.Svr)
        
        self.SvrParameters={}
        self.SvrParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 5), size=(120, 50))
        self.SvrParameters['Scaler'] = wx.ComboBox(self,value='MinMax', pos=(250, 25), choices=['MinMax','Standard','Robust','None'])
        
        self.SvrParameters['KernelText'] = wx.StaticBox(self, label='Kernel', pos=(240, 55), size=(120, 50))
        self.SvrParameters['Kernel'] = wx.ComboBox(self,value='poly', pos=(250, 75), choices=['linear','poly','rbf','sigmoid','precomputed'])
        
        
        self.SvrParameters['GammaText'] = wx.StaticBox(self, label='Gamma', pos=(240, 105), size=(120, 50))
        self.SvrParameters['Gamma'] = wx.ComboBox(self,value='auto', pos=(250, 125), choices=['scale','auto'])
        
        self.SvrParameters['CText'] = wx.StaticBox(self, label='C', pos=(240, 155), size=(120, 50))
        self.SvrParameters['C'] = wx.SpinCtrl(self, value='4', pos=(250, 175), size=(50, -1), min=1, max=10)
        
        self.SvrParameters['EpsilonText'] = wx.StaticBox(self, label='100 x Epsilon', pos=(240, 205), size=(120, 50))
        self.SvrParameters['Epsilon'] = wx.SpinCtrl(self, value='5', pos=(250, 225), size=(50, -1), min=0, max=100)
        
        Show(self.SvrParameters,False)
        
        #KNN
        self.knn = wx.CheckBox(self, label='K Nearest Neighbors', pos=(15, 120))
        self.knn.Value=False
        self.knn.Bind(wx.EVT_CHECKBOX,self.Knn)
        
        self.KnnParameters={}
        self.KnnParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 5), size=(120, 50))
        self.KnnParameters['Scaler'] = wx.ComboBox(self,value='MinMax', pos=(250, 25), choices=['MinMax','Standard','Robust','None'])
        
        self.KnnParameters['DimText'] = wx.StaticBox(self, label='Reduce Dim', pos=(240, 55), size=(120, 50))
        self.KnnParameters['Dim'] = wx.SpinCtrl(self, value='6', pos=(250, 75), size=(50, -1), min=0, max=100)
        
        self.KnnParameters['NeighborText'] = wx.StaticBox(self, label='Number of Neighbors', pos=(240, 105), size=(120, 50))
        self.KnnParameters['Neighbor'] = wx.SpinCtrl(self, value='7', pos=(250, 125), size=(50, -1), min=0, max=100)
        
        self.KnnParameters['WeightText'] = wx.StaticBox(self, label='Weight', pos=(240, 155), size=(120, 50))
        self.KnnParameters['Weight'] = wx.ComboBox(self,value='distance', pos=(250, 175), choices=['distance','uniform'])
        
        self.KnnParameters['AlgorithmText'] = wx.StaticBox(self, label='Algorithm', pos=(240, 205), size=(120, 50))
        self.KnnParameters['Algorithm'] = wx.ComboBox(self,value='auto', pos=(250, 225), choices=['auto','ball_tree','kd_tree','brute'])
        
        self.KnnParameters['LeafText'] = wx.StaticBox(self, label='Leaf Size', pos=(240, 255), size=(120, 50))
        self.KnnParameters['Leaf'] = wx.SpinCtrl(self, value='20', pos=(250, 275), size=(50, -1), min=0, max=100)
        
        Show(self.KnnParameters,False)
        
        #ANN
        self.ann = wx.CheckBox(self, label='Artificial Neural Network', pos=(15, 150))
        self.ann.Value=False
        self.ann.Bind(wx.EVT_CHECKBOX,self.Ann)
        
        self.AnnParameters={}
        self.AnnParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 5), size=(120, 50))
        self.AnnParameters['Scaler'] = wx.ComboBox(self,value='MinMax', pos=(250, 25), choices=['MinMax','Standard','Robust','None'])
        
        self.AnnParameters['DimText'] = wx.StaticBox(self, label='Reduce Dim', pos=(240, 55), size=(120, 50))
        self.AnnParameters['Dim'] = wx.SpinCtrl(self, value='17', pos=(250, 75), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['SolverText'] = wx.StaticBox(self, label='Solver', pos=(240, 105), size=(120, 50))
        self.AnnParameters['Solver'] = wx.ComboBox(self,value='lbfgs', pos=(250, 125), choices=['lbfgs','adam'])
        
        self.AnnParameters['HiddenLayerText'] = wx.StaticBox(self, label='Hidden Layer', pos=(240, 155), size=(120, 50))
        self.AnnParameters['HiddenLayer1'] = wx.SpinCtrl(self, value='30', pos=(250, 175), size=(50, -1), min=0, max=100)
        self.AnnParameters['HiddenLayer2'] = wx.SpinCtrl(self, value='20', pos=(305, 175), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['ActivationText'] = wx.StaticBox(self, label='Activation', pos=(240, 205), size=(120, 50))
        self.AnnParameters['Activation'] = wx.ComboBox(self,value='relu', pos=(250, 225), choices=['relu','tanh'])
        
        self.AnnParameters['AlphaText'] = wx.StaticBox(self, label='Logarit of Alpha', pos=(240, 255), size=(120, 50))
        self.AnnParameters['Alpha'] = wx.SpinCtrl(self, value='-2', pos=(250, 275), size=(50, -1), min=-100, max=100)
        
        self.AnnParameters['IterText'] = wx.StaticBox(self, label='Logarit of Max_iter', pos=(370, 5), size=(120, 50))
        self.AnnParameters['Iter'] = wx.SpinCtrl(self, value='4', pos=(380, 25), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['MomentumText'] = wx.StaticBox(self, label='10 x Momentum', pos=(370, 55), size=(120, 50))
        self.AnnParameters['Momentum'] = wx.SpinCtrl(self, value='9', pos=(380, 75), size=(50, -1), min=0, max=10)
        
        self.AnnParameters['NesterovsText'] = wx.StaticBox(self, label='NesterovMomentum', pos=(370, 105), size=(120, 50))
        self.AnnParameters['Nesterovs'] = wx.ComboBox(self,value='Yes', pos=(380, 125), choices=['Yes','No'])
        
        Show(self.AnnParameters,False)
        
        #Random Forest
        self.rf = wx.CheckBox(self, label='Random Forest', pos=(15, 180))
        self.rf.Value=False
        self.rf.Bind(wx.EVT_CHECKBOX,self.Rf)
        
        self.RfParameters={}
        self.RfParameters['EstimatorText'] = wx.StaticBox(self, label='Number of Estimator', pos=(240, 5), size=(120, 50))
        self.RfParameters['Estimator'] = wx.SpinCtrl(self, value='240', pos=(250, 25), size=(50, -1), min=0, max=1000)
        
        self.RfParameters['CriterionText'] = wx.StaticBox(self, label='Criterion', pos=(240, 55), size=(120, 50))
        self.RfParameters['Criterion'] = wx.ComboBox(self,value='absolute_error', pos=(250, 75), choices=['absolute_error','squared_error'])
        
        self.RfParameters['MinSplitText'] = wx.StaticBox(self, label='Min Sample Split', pos=(240, 105), size=(120, 50))
        self.RfParameters['MinSplit'] = wx.SpinCtrl(self, value='2', pos=(250, 125), size=(50, -1), min=0, max=100)
        
        self.RfParameters['DepthText'] = wx.StaticBox(self, label='Max Depth', pos=(240, 155), size=(120, 50))
        self.RfParameters['Depth'] = wx.SpinCtrl(self, value='9', pos=(250, 175), size=(50, -1), min=0, max=100)
        
        self.RfParameters['MinLeafText'] = wx.StaticBox(self, label='Min Sample Leaf', pos=(240, 205), size=(120, 50))
        self.RfParameters['MinLeaf'] = wx.SpinCtrl(self, value='1', pos=(250, 225), size=(50, -1), min=0, max=100)
        
        self.RfParameters['MaxFeatureText'] = wx.StaticBox(self, label='Max Feature', pos=(240, 255), size=(120, 50))
        self.RfParameters['MaxFeature'] = wx.ComboBox(self,value='sqrt', pos=(250, 275), choices=['sqrt','log2','auto'])
        
        self.RfParameters['BootstrapText'] = wx.StaticBox(self, label='Bootstrap', pos=(370, 5), size=(120, 50))
        self.RfParameters['Bootstrap'] = wx.ComboBox(self,value='No', pos=(380, 25), choices=['Yes','No'])
        
        Show(self.RfParameters,False)
        
        #Gradient Boosting
        self.gb = wx.CheckBox(self, label='Gradient Boosting', pos=(15, 210))
        self.gb.Value=False
        self.gb.Bind(wx.EVT_CHECKBOX,self.Gb)
        
        self.GbParameters={}
        self.GbParameters['EstimatorText'] = wx.StaticBox(self, label='Number of Estimator', pos=(240, 5), size=(120, 50))
        self.GbParameters['Estimator'] = wx.SpinCtrl(self, value='120', pos=(250, 25), size=(50, -1), min=0, max=1000)
        
        self.GbParameters['DepthText'] = wx.StaticBox(self, label='Max Depth', pos=(240, 55), size=(120, 50))
        self.GbParameters['Depth'] = wx.SpinCtrl(self, value='7', pos=(250, 75), size=(50, -1), min=0, max=100)
        
        self.GbParameters['LearningText'] = wx.StaticBox(self, label='1000 x Learning Rate', pos=(240, 105), size=(120, 50))
        self.GbParameters['Learning'] = wx.SpinCtrl(self, value='88', pos=(250, 125), size=(50, -1), min=0, max=1000)
        
        self.GbParameters['CriterionText'] = wx.StaticBox(self, label='Criterion', pos=(240, 155), size=(120, 50))
        self.GbParameters['Criterion'] = wx.ComboBox(self,value='friedman_mse', pos=(250, 175), choices=['friedman_mse','mae','mse'])
        
        self.GbParameters['MinSplitText'] = wx.StaticBox(self, label='Min Sample Split', pos=(240, 205), size=(120, 50))
        self.GbParameters['MinSplit'] = wx.SpinCtrl(self, value='4', pos=(250, 225), size=(50, -1), min=0, max=100)
        
        self.GbParameters['MinLeafText'] = wx.StaticBox(self, label='Min Sample Leaf', pos=(240, 255), size=(120, 50))
        self.GbParameters['MinLeaf'] = wx.SpinCtrl(self, value='2', pos=(250, 275), size=(50, -1), min=0, max=100)
        
        self.GbParameters['MaxFeatureText'] = wx.StaticBox(self, label='Max Feature', pos=(370, 5), size=(120, 50))
        self.GbParameters['MaxFeature'] = wx.ComboBox(self,value='sqrt', pos=(380, 25), choices=['sqrt','log2','auto'])
        
        Show(self.GbParameters,False)
        
        #Ensemble Models
        self.vt = wx.CheckBox(self, label='Voting Model', pos=(15, 240))
        self.vt.Value=False
        self.vt.Bind(wx.EVT_CHECKBOX,self.Ensemble)
        self.st = wx.CheckBox(self, label='Stacking Model', pos=(15, 270))
        self.st.Value=False
        self.st.Bind(wx.EVT_CHECKBOX,self.Ensemble)
        
        self.EnsembleParameters={}
        self.EnsembleParameters['ModelText'] = wx.StaticBox(self, label='Include Models', pos=(240, 5),size=(170,200))
        self.EnsembleParameters['lr'] = wx.CheckBox(self, label='Linear Regression', pos=(250, 25))
        self.EnsembleParameters['ridge'] = wx.CheckBox(self, label='Ridge Regression', pos=(250, 50))
        self.EnsembleParameters['svr'] = wx.CheckBox(self, label='Support Vector Machine', pos=(250, 75))
        self.EnsembleParameters['knn'] = wx.CheckBox(self, label='K Nearest Neighbors', pos=(250, 100))
        self.EnsembleParameters['ann'] = wx.CheckBox(self, label='Artificial Neural Network', pos=(250, 125))
        self.EnsembleParameters['rf'] = wx.CheckBox(self, label='Random Forest', pos=(250, 150))
        self.EnsembleParameters['rf'].Value=True
        self.EnsembleParameters['gb'] = wx.CheckBox(self, label='Gradient Boosting', pos=(250, 175))
        self.EnsembleParameters['gb'].Value=True

        Show(self.EnsembleParameters,False)

    #Function
    def Lr(self,e):
        Show(self.LrParameters,True)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        Show(self.EnsembleParameters,False)
    
    def Ridge(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,True)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        Show(self.EnsembleParameters,False)
        
    def Svr(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,True)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        Show(self.EnsembleParameters,False)
    
    def Knn(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,True)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        Show(self.EnsembleParameters,False)
    
    def Ann(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,True)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        Show(self.EnsembleParameters,False)
    
    def Rf(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,True)
        Show(self.GbParameters,False)
        Show(self.EnsembleParameters,False)
    
    def Gb(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,True)
        Show(self.EnsembleParameters,False)
        
    def Ensemble(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        Show(self.EnsembleParameters,True)

class Cross_Validation(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        
        #Cross Validation
        wx.StaticBox(self, label='Cross Validation', pos=(5, 5), size=(480, 50))
        
        self.RepeatedParameters={}
        wx.StaticText(self, label='n_splits:', pos=(15, 30))
        self.RepeatedParameters['NSplit'] = wx.SpinCtrl(self, value='5', pos=(60, 25), size=(50, -1), min=0, max=100)
        
        wx.StaticText(self, label='n_repeats:', pos=(175, 30))
        self.RepeatedParameters['NRepeat'] = wx.SpinCtrl(self, value='6', pos=(230, 25), size=(50, -1), min=0, max=100)
        
        wx.StaticText(self, label='random_state:', pos=(345, 30))
        self.RepeatedParameters['RandomState'] = wx.SpinCtrl(self, value='22', pos=(425, 25), size=(50, -1), min=0, max=100)
        
        #Models
        wx.StaticBox(self, label='Model', pos=(5, 55), size=(180, 240))
        
        #Linear Regression
        self.lr = wx.CheckBox(self, label='Linear Regression', pos=(15, 80))
        self.lr.Value=False
        self.lr.Bind(wx.EVT_CHECKBOX,self.Lr)
        
        self.LrParameters={}
        self.LrParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 55), size=(120, 50))
        self.LrParameters['Scaler'] = wx.ComboBox(self,value='None', pos=(250, 75), choices=['MinMax','Standard','Robust','None'])
        
        Show(self.LrParameters,False)
        
        #Ridge
        self.ridge = wx.CheckBox(self, label='Ridge Regression', pos=(15, 110))
        self.ridge.Value=False
        self.ridge.Bind(wx.EVT_CHECKBOX,self.Ridge)
        
        self.RidgeParameters={}
        self.RidgeParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 55), size=(120, 50))
        self.RidgeParameters['Scaler'] = wx.ComboBox(self,value='None', pos=(250, 75), choices=['MinMax','Standard','Robust','None'])
        
        self.RidgeParameters['SolverText'] = wx.StaticBox(self, label='Solver', pos=(240, 105), size=(120, 50))
        self.RidgeParameters['Solver'] = wx.ComboBox(self,value='svd', pos=(250, 125), choices=['svd','cholesky','lsqr','sag'])
        
        self.RidgeParameters['AlphaText'] = wx.StaticBox(self, label='Logarit of Alpha', pos=(240, 155), size=(120, 50))
        self.RidgeParameters['Alpha'] = wx.SpinCtrl(self, value='10', pos=(250, 175), size=(50, -1), min=-100, max=100)
        
        self.RidgeParameters['FitInterceptText'] = wx.StaticBox(self, label='Fit Intercept', pos=(240, 205), size=(120, 50))
        self.RidgeParameters['FitIntercept'] = wx.ComboBox(self,value='No', pos=(250, 225), choices=['Yes','No'])
        
        Show(self.RidgeParameters,False)
        
        #SVR
        self.svr = wx.CheckBox(self, label='Support Vector Regression', pos=(15, 140))
        self.svr.Value=False
        self.svr.Bind(wx.EVT_CHECKBOX,self.Svr)
        
        self.SvrParameters={}
        self.SvrParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 55), size=(120, 50))
        self.SvrParameters['Scaler'] = wx.ComboBox(self,value='MinMax', pos=(250, 75), choices=['MinMax','Standard','Robust','None'])
        
        self.SvrParameters['KernelText'] = wx.StaticBox(self, label='Kernel', pos=(240, 105), size=(120, 50))
        self.SvrParameters['Kernel'] = wx.ComboBox(self,value='poly', pos=(250, 125), choices=['linear','poly','rbf','sigmoid','precomputed'])
        
        
        self.SvrParameters['GammaText'] = wx.StaticBox(self, label='Gamma', pos=(240, 155), size=(120, 50))
        self.SvrParameters['Gamma'] = wx.ComboBox(self,value='auto', pos=(250, 175), choices=['scale','auto'])
        
        self.SvrParameters['CText'] = wx.StaticBox(self, label='C', pos=(240, 205), size=(120, 50))
        self.SvrParameters['C'] = wx.SpinCtrl(self, value='4', pos=(250, 225), size=(50, -1), min=1, max=10)
        
        self.SvrParameters['EpsilonText'] = wx.StaticBox(self, label='100 x Epsilon', pos=(240, 255), size=(120, 50))
        self.SvrParameters['Epsilon'] = wx.SpinCtrl(self, value='5', pos=(250, 275), size=(50, -1), min=0, max=100)
        
        Show(self.SvrParameters,False)
        
        #KNN
        self.knn = wx.CheckBox(self, label='K Nearest Neighbors', pos=(15, 170))
        self.knn.Value=False
        self.knn.Bind(wx.EVT_CHECKBOX,self.Knn)
        
        self.KnnParameters={}
        self.KnnParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 55), size=(120, 50))
        self.KnnParameters['Scaler'] = wx.ComboBox(self,value='MinMax', pos=(250, 75), choices=['MinMax','Standard','Robust','None'])
        
        self.KnnParameters['DimText'] = wx.StaticBox(self, label='Reduce Dim', pos=(240, 105), size=(120, 50))
        self.KnnParameters['Dim'] = wx.SpinCtrl(self, value='6', pos=(250, 125), size=(50, -1), min=0, max=100)
        
        self.KnnParameters['NeighborText'] = wx.StaticBox(self, label='Number of Neighbors', pos=(240, 155), size=(120, 50))
        self.KnnParameters['Neighbor'] = wx.SpinCtrl(self, value='7', pos=(250, 175), size=(50, -1), min=0, max=100)
        
        self.KnnParameters['WeightText'] = wx.StaticBox(self, label='Weight', pos=(240, 205), size=(120, 50))
        self.KnnParameters['Weight'] = wx.ComboBox(self,value='distance', pos=(250, 225), choices=['distance','uniform'])
        
        self.KnnParameters['AlgorithmText'] = wx.StaticBox(self, label='Algorithm', pos=(240, 255), size=(120, 50))
        self.KnnParameters['Algorithm'] = wx.ComboBox(self,value='auto', pos=(250, 275), choices=['auto','ball_tree','kd_tree','brute'])
        
        self.KnnParameters['LeafText'] = wx.StaticBox(self, label='Leaf Size', pos=(370, 55), size=(120, 50))
        self.KnnParameters['Leaf'] = wx.SpinCtrl(self, value='20', pos=(380, 75), size=(50, -1), min=0, max=100)
        
        Show(self.KnnParameters,False)
        
        #ANN
        self.ann = wx.CheckBox(self, label='Artificial Neural Network', pos=(15, 200))
        self.ann.Value=False
        self.ann.Bind(wx.EVT_CHECKBOX,self.Ann)
        
        self.AnnParameters={}
        self.AnnParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 55), size=(120, 50))
        self.AnnParameters['Scaler'] = wx.ComboBox(self,value='MinMax', pos=(250, 75), choices=['MinMax','Standard','Robust','None'])
        
        self.AnnParameters['DimText'] = wx.StaticBox(self, label='Reduce Dim', pos=(240, 105), size=(120, 50))
        self.AnnParameters['Dim'] = wx.SpinCtrl(self, value='17', pos=(250, 125), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['SolverText'] = wx.StaticBox(self, label='Solver', pos=(240, 155), size=(120, 50))
        self.AnnParameters['Solver'] = wx.ComboBox(self,value='lbfgs', pos=(250, 175), choices=['lbfgs','adam'])
        
        self.AnnParameters['HiddenLayerText'] = wx.StaticBox(self, label='Hidden Layer', pos=(240, 205), size=(120, 50))
        self.AnnParameters['HiddenLayer1'] = wx.SpinCtrl(self, value='30', pos=(250, 225), size=(50, -1), min=0, max=100)
        self.AnnParameters['HiddenLayer2'] = wx.SpinCtrl(self, value='20', pos=(305, 225), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['ActivationText'] = wx.StaticBox(self, label='Activation', pos=(240, 255), size=(120, 50))
        self.AnnParameters['Activation'] = wx.ComboBox(self,value='relu', pos=(250, 275), choices=['relu','tanh'])
        
        self.AnnParameters['AlphaText'] = wx.StaticBox(self, label='Logarit of Alpha', pos=(370, 55), size=(120, 50))
        self.AnnParameters['Alpha'] = wx.SpinCtrl(self, value='-2', pos=(380, 75), size=(50, -1), min=-100, max=100)
        
        self.AnnParameters['IterText'] = wx.StaticBox(self, label='Logarit of Max_iter', pos=(370, 105), size=(120, 50))
        self.AnnParameters['Iter'] = wx.SpinCtrl(self, value='4', pos=(380, 125), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['MomentumText'] = wx.StaticBox(self, label='10 x Momentum', pos=(370, 155), size=(120, 50))
        self.AnnParameters['Momentum'] = wx.SpinCtrl(self, value='9', pos=(380, 175), size=(50, -1), min=0, max=10)
        
        self.AnnParameters['NesterovsText'] = wx.StaticBox(self, label='NesterovMomentum', pos=(370, 205), size=(120, 50))
        self.AnnParameters['Nesterovs'] = wx.ComboBox(self,value='Yes', pos=(380, 225), choices=['Yes','No'])
        
        Show(self.AnnParameters,False)
        
        #Random Forest
        self.rf = wx.CheckBox(self, label='Random Forest', pos=(15, 230))
        self.rf.Value=False
        self.rf.Bind(wx.EVT_CHECKBOX,self.Rf)
        
        self.RfParameters={}
        self.RfParameters['EstimatorText'] = wx.StaticBox(self, label='Number of Estimator', pos=(240, 55), size=(120, 50))
        self.RfParameters['Estimator'] = wx.SpinCtrl(self, value='240', pos=(250, 75), size=(50, -1), min=0, max=1000)
        
        self.RfParameters['CriterionText'] = wx.StaticBox(self, label='Criterion', pos=(240, 105), size=(120, 50))
        self.RfParameters['Criterion'] = wx.ComboBox(self,value='absolute_error', pos=(250, 125), choices=['absolute_error','squared_error'])
        
        self.RfParameters['MinSplitText'] = wx.StaticBox(self, label='Min Sample Split', pos=(240, 155), size=(120, 50))
        self.RfParameters['MinSplit'] = wx.SpinCtrl(self, value='2', pos=(250, 175), size=(50, -1), min=0, max=100)
        
        self.RfParameters['DepthText'] = wx.StaticBox(self, label='Max Depth', pos=(240, 205), size=(120, 50))
        self.RfParameters['Depth'] = wx.SpinCtrl(self, value='9', pos=(250, 225), size=(50, -1), min=0, max=100)
        
        self.RfParameters['MinLeafText'] = wx.StaticBox(self, label='Min Sample Leaf', pos=(240, 255), size=(120, 50))
        self.RfParameters['MinLeaf'] = wx.SpinCtrl(self, value='1', pos=(250, 275), size=(50, -1), min=0, max=100)
        
        self.RfParameters['MaxFeatureText'] = wx.StaticBox(self, label='Max Feature', pos=(370, 55), size=(120, 50))
        self.RfParameters['MaxFeature'] = wx.ComboBox(self,value='sqrt', pos=(380, 75), choices=['sqrt','log2','auto'])
        
        self.RfParameters['BootstrapText'] = wx.StaticBox(self, label='Bootstrap', pos=(370, 105), size=(120, 50))
        self.RfParameters['Bootstrap'] = wx.ComboBox(self,value='No', pos=(380, 125), choices=['Yes','No'])
        
        Show(self.RfParameters,False)
        
        #Gradient Boosting
        self.gb = wx.CheckBox(self, label='Gradient Boosting', pos=(15, 260))
        self.gb.Value=False
        self.gb.Bind(wx.EVT_CHECKBOX,self.Gb)
        
        self.GbParameters={}
        self.GbParameters['EstimatorText'] = wx.StaticBox(self, label='Number of Estimator', pos=(240, 55), size=(120, 50))
        self.GbParameters['Estimator'] = wx.SpinCtrl(self, value='120', pos=(250, 75), size=(50, -1), min=0, max=1000)
        
        self.GbParameters['DepthText'] = wx.StaticBox(self, label='Max Depth', pos=(240, 105), size=(120, 50))
        self.GbParameters['Depth'] = wx.SpinCtrl(self, value='7', pos=(250, 125), size=(50, -1), min=0, max=100)
        
        self.GbParameters['LearningText'] = wx.StaticBox(self, label='1000 x Learning Rate', pos=(240, 155), size=(120, 50))
        self.GbParameters['Learning'] = wx.SpinCtrl(self, value='88', pos=(250, 175), size=(50, -1), min=0, max=1000)
        
        self.GbParameters['CriterionText'] = wx.StaticBox(self, label='Criterion', pos=(240, 205), size=(120, 50))
        self.GbParameters['Criterion'] = wx.ComboBox(self,value='friedman_mse', pos=(250, 225), choices=['friedman_mse','mae','mse'])
        
        self.GbParameters['MinSplitText'] = wx.StaticBox(self, label='Min Sample Split', pos=(240, 255), size=(120, 50))
        self.GbParameters['MinSplit'] = wx.SpinCtrl(self, value='4', pos=(250, 275), size=(50, -1), min=0, max=100)
        
        self.GbParameters['MinLeafText'] = wx.StaticBox(self, label='Min Sample Leaf', pos=(370, 55), size=(120, 50))
        self.GbParameters['MinLeaf'] = wx.SpinCtrl(self, value='2', pos=(380, 75), size=(50, -1), min=0, max=100)
        
        self.GbParameters['MaxFeatureText'] = wx.StaticBox(self, label='Max Feature', pos=(370, 105), size=(120, 50))
        self.GbParameters['MaxFeature'] = wx.ComboBox(self,value='sqrt', pos=(380, 125), choices=['sqrt','log2','auto'])
        
        Show(self.GbParameters,False)
        
    def Lr(self,e):
        Show(self.LrParameters,True)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
    
    def Ridge(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,True)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
    def Svr(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,True)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
    
    def Knn(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,True)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
    
    def Ann(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,True)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
    
    def Rf(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,True)
        Show(self.GbParameters,False)
    
    def Gb(self,e):
        Show(self.LrParameters,False)
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,True)

class Feature_Selection(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        
        #Feature Selection
        wx.StaticBox(self, label='Feature Selcetion', pos=(5, 5), size=(480, 50))
        
        self.SelectionParameters={}
        wx.StaticText(self, label='Model:', pos=(10, 30))
        self.SelectionParameters['Model'] = wx.ComboBox(self, value='Feature Backward Selection', pos=(50, 25), choices=['Feature Backward Selection','Feature Forward Selection'])
        
        wx.StaticText(self, label='n_step:', pos=(235, 30))
        self.SelectionParameters['NStep'] = wx.SpinCtrl(self, value='1', pos=(275, 25), size=(50, -1), min=0, max=100)
        
        wx.StaticText(self, label='Metric:', pos=(335, 30))
        self.SelectionParameters['Metric'] = wx.ComboBox(self, value='Pearson R', pos=(375, 25), choices=['MAPE','negative MAE', 'RMSE', 'Pearson R', 'P1 Sigma', 'P2 Sigma'])
        
        #ML Models
        wx.StaticBox(self, label='Machine Learning Model', pos=(5, 55), size=(180, 240))
        
        #Linear Regression
        self.lr = wx.CheckBox(self, label='Linear Regression', pos=(15, 80))
        self.lr.Value=False
        self.lr.Bind(wx.EVT_CHECKBOX,self.Lr)
        
        self.LrParameters={}
        self.LrParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 55), size=(120, 50))
        self.LrParameters['Scaler'] = wx.ComboBox(self,value='None', pos=(250, 75), choices=['MinMax','Standard','Robust','None'])
        
        Show(self.LrParameters,False)
        
        #Ridge
        self.ridge = wx.CheckBox(self, label='Ridge Regression', pos=(15, 110))
        self.ridge.Value=False
        self.ridge.Bind(wx.EVT_CHECKBOX,self.Ridge)
        
        self.RidgeParameters={}
        self.RidgeParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 55), size=(120, 50))
        self.RidgeParameters['Scaler'] = wx.ComboBox(self,value='None', pos=(250, 75), choices=['MinMax','Standard','Robust','None'])
        
        self.RidgeParameters['SolverText'] = wx.StaticBox(self, label='Solver', pos=(240, 105), size=(120, 50))
        self.RidgeParameters['Solver'] = wx.ComboBox(self,value='svd', pos=(250, 125), choices=['svd','cholesky','lsqr','sag'])
        
        self.RidgeParameters['AlphaText'] = wx.StaticBox(self, label='Logarit of Alpha', pos=(240, 155), size=(120, 50))
        self.RidgeParameters['Alpha'] = wx.SpinCtrl(self, value='10', pos=(250, 175), size=(50, -1), min=-100, max=100)
        
        self.RidgeParameters['FitInterceptText'] = wx.StaticBox(self, label='Fit Intercept', pos=(240, 205), size=(120, 50))
        self.RidgeParameters['FitIntercept'] = wx.ComboBox(self,value='No', pos=(250, 225), choices=['Yes','No'])
        
        Show(self.RidgeParameters,False)
        
        #SVR
        self.svr = wx.CheckBox(self, label='Support Vector Regression', pos=(15, 140))
        self.svr.Value=False
        self.svr.Bind(wx.EVT_CHECKBOX,self.Svr)
        
        self.SvrParameters={}
        self.SvrParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 55), size=(120, 50))
        self.SvrParameters['Scaler'] = wx.ComboBox(self,value='MinMax', pos=(250, 75), choices=['MinMax','Standard','Robust','None'])
        
        self.SvrParameters['KernelText'] = wx.StaticBox(self, label='Kernel', pos=(240, 105), size=(120, 50))
        self.SvrParameters['Kernel'] = wx.ComboBox(self,value='poly', pos=(250, 125), choices=['linear','poly','rbf','sigmoid','precomputed'])
        
        
        self.SvrParameters['GammaText'] = wx.StaticBox(self, label='Gamma', pos=(240, 155), size=(120, 50))
        self.SvrParameters['Gamma'] = wx.ComboBox(self,value='auto', pos=(250, 175), choices=['scale','auto'])
        
        self.SvrParameters['CText'] = wx.StaticBox(self, label='C', pos=(240, 205), size=(120, 50))
        self.SvrParameters['C'] = wx.SpinCtrl(self, value='4', pos=(250, 225), size=(50, -1), min=1, max=10)
        
        self.SvrParameters['EpsilonText'] = wx.StaticBox(self, label='100 x Epsilon', pos=(240, 255), size=(120, 50))
        self.SvrParameters['Epsilon'] = wx.SpinCtrl(self, value='5', pos=(250, 275), size=(50, -1), min=0, max=100)
        
        Show(self.SvrParameters,False)
        
        #KNN
        self.knn = wx.CheckBox(self, label='K Nearest Neighbors', pos=(15, 170))
        self.knn.Value=False
        self.knn.Bind(wx.EVT_CHECKBOX,self.Knn)
        
        self.KnnParameters={}
        self.KnnParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 55), size=(120, 50))
        self.KnnParameters['Scaler'] = wx.ComboBox(self,value='MinMax', pos=(250, 75), choices=['MinMax','Standard','Robust','None'])
        
        self.KnnParameters['DimText'] = wx.StaticBox(self, label='Reduce Dim', pos=(240, 105), size=(120, 50))
        self.KnnParameters['Dim'] = wx.SpinCtrl(self, value='6', pos=(250, 125), size=(50, -1), min=0, max=100)
        
        self.KnnParameters['NeighborText'] = wx.StaticBox(self, label='Number of Neighbors', pos=(240, 155), size=(120, 50))
        self.KnnParameters['Neighbor'] = wx.SpinCtrl(self, value='7', pos=(250, 175), size=(50, -1), min=0, max=100)
        
        self.KnnParameters['WeightText'] = wx.StaticBox(self, label='Weight', pos=(240, 205), size=(120, 50))
        self.KnnParameters['Weight'] = wx.ComboBox(self,value='distance', pos=(250, 225), choices=['distance','uniform'])
        
        self.KnnParameters['AlgorithmText'] = wx.StaticBox(self, label='Algorithm', pos=(240, 255), size=(120, 50))
        self.KnnParameters['Algorithm'] = wx.ComboBox(self,value='auto', pos=(250, 275), choices=['auto','ball_tree','kd_tree','brute'])
        
        self.KnnParameters['LeafText'] = wx.StaticBox(self, label='Leaf Size', pos=(370, 55), size=(120, 50))
        self.KnnParameters['Leaf'] = wx.SpinCtrl(self, value='20', pos=(380, 75), size=(50, -1), min=0, max=100)
        
        Show(self.KnnParameters,False)
        
        #ANN
        self.ann = wx.CheckBox(self, label='Artificial Neural Network', pos=(15, 200))
        self.ann.Value=False
        self.ann.Bind(wx.EVT_CHECKBOX,self.Ann)
        
        self.AnnParameters={}
        self.AnnParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 55), size=(120, 50))
        self.AnnParameters['Scaler'] = wx.ComboBox(self,value='MinMax', pos=(250, 75), choices=['MinMax','Standard','Robust','None'])
        
        self.AnnParameters['DimText'] = wx.StaticBox(self, label='Reduce Dim', pos=(240, 105), size=(120, 50))
        self.AnnParameters['Dim'] = wx.SpinCtrl(self, value='17', pos=(250, 125), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['SolverText'] = wx.StaticBox(self, label='Solver', pos=(240, 155), size=(120, 50))
        self.AnnParameters['Solver'] = wx.ComboBox(self,value='lbfgs', pos=(250, 175), choices=['lbfgs','adam'])
        
        self.AnnParameters['HiddenLayerText'] = wx.StaticBox(self, label='Hidden Layer', pos=(240, 205), size=(120, 50))
        self.AnnParameters['HiddenLayer1'] = wx.SpinCtrl(self, value='30', pos=(250, 225), size=(50, -1), min=0, max=100)
        self.AnnParameters['HiddenLayer2'] = wx.SpinCtrl(self, value='20', pos=(305, 225), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['ActivationText'] = wx.StaticBox(self, label='Activation', pos=(240, 255), size=(120, 50))
        self.AnnParameters['Activation'] = wx.ComboBox(self,value='relu', pos=(250, 275), choices=['relu','tanh'])
        
        self.AnnParameters['AlphaText'] = wx.StaticBox(self, label='Logarit of Alpha', pos=(370, 55), size=(120, 50))
        self.AnnParameters['Alpha'] = wx.SpinCtrl(self, value='-2', pos=(380, 75), size=(50, -1), min=-100, max=100)
        
        self.AnnParameters['IterText'] = wx.StaticBox(self, label='Logarit of Max_iter', pos=(370, 105), size=(120, 50))
        self.AnnParameters['Iter'] = wx.SpinCtrl(self, value='4', pos=(380, 125), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['MomentumText'] = wx.StaticBox(self, label='10 x Momentum', pos=(370, 155), size=(120, 50))
        self.AnnParameters['Momentum'] = wx.SpinCtrl(self, value='9', pos=(380, 175), size=(50, -1), min=0, max=10)
        
        self.AnnParameters['NesterovsText'] = wx.StaticBox(self, label='NesterovMomentum', pos=(370, 205), size=(120, 50))
        self.AnnParameters['Nesterovs'] = wx.ComboBox(self,value='Yes', pos=(380, 225), choices=['Yes','No'])
        
        Show(self.AnnParameters,False)
        
        #Random Forest
        self.rf = wx.CheckBox(self, label='Random Forest', pos=(15, 230))
        self.rf.Value=False
        self.rf.Bind(wx.EVT_CHECKBOX,self.Rf)
        
        self.RfParameters={}
        self.RfParameters['EstimatorText'] = wx.StaticBox(self, label='Number of Estimator', pos=(240, 55), size=(120, 50))
        self.RfParameters['Estimator'] = wx.SpinCtrl(self, value='240', pos=(250, 75), size=(50, -1), min=0, max=1000)
        
        self.RfParameters['CriterionText'] = wx.StaticBox(self, label='Criterion', pos=(240, 105), size=(120, 50))
        self.RfParameters['Criterion'] = wx.ComboBox(self,value='absolute_error', pos=(250, 125), choices=['absolute_error','squared_error'])
        
        self.RfParameters['MinSplitText'] = wx.StaticBox(self, label='Min Sample Split', pos=(240, 155), size=(120, 50))
        self.RfParameters['MinSplit'] = wx.SpinCtrl(self, value='2', pos=(250, 175), size=(50, -1), min=0, max=100)
        
        self.RfParameters['DepthText'] = wx.StaticBox(self, label='Max Depth', pos=(240, 205), size=(120, 50))
        self.RfParameters['Depth'] = wx.SpinCtrl(self, value='9', pos=(250, 225), size=(50, -1), min=0, max=100)
        
        self.RfParameters['MinLeafText'] = wx.StaticBox(self, label='Min Sample Leaf', pos=(240, 255), size=(120, 50))
        self.RfParameters['MinLeaf'] = wx.SpinCtrl(self, value='1', pos=(250, 275), size=(50, -1), min=0, max=100)
        
        self.RfParameters['MaxFeatureText'] = wx.StaticBox(self, label='Max Feature', pos=(370, 55), size=(120, 50))
        self.RfParameters['MaxFeature'] = wx.ComboBox(self,value='sqrt', pos=(380, 75), choices=['sqrt','log2','auto'])
        
        self.RfParameters['BootstrapText'] = wx.StaticBox(self, label='Bootstrap', pos=(370, 105), size=(120, 50))
        self.RfParameters['Bootstrap'] = wx.ComboBox(self,value='No', pos=(380, 125), choices=['Yes','No'])
        
        Show(self.RfParameters,False)
        
        #Gradient Boosting
        self.gb = wx.CheckBox(self, label='Gradient Boosting', pos=(15, 260))
        self.gb.Value=False
        self.gb.Bind(wx.EVT_CHECKBOX,self.Gb)
        
        self.GbParameters={}
        self.GbParameters['EstimatorText'] = wx.StaticBox(self, label='Number of Estimator', pos=(240, 55), size=(120, 50))
        self.GbParameters['Estimator'] = wx.SpinCtrl(self, value='120', pos=(250, 75), size=(50, -1), min=0, max=1000)
        
        self.GbParameters['DepthText'] = wx.StaticBox(self, label='Max Depth', pos=(240, 105), size=(120, 50))
        self.GbParameters['Depth'] = wx.SpinCtrl(self, value='7', pos=(250, 125), size=(50, -1), min=0, max=100)
        
        self.GbParameters['LearningText'] = wx.StaticBox(self, label='1000 x Learning Rate', pos=(240, 155), size=(120, 50))
        self.GbParameters['Learning'] = wx.SpinCtrl(self, value='88', pos=(250, 175), size=(50, -1), min=0, max=1000)
        
        self.GbParameters['CriterionText'] = wx.StaticBox(self, label='Criterion', pos=(240, 205), size=(120, 50))
        self.GbParameters['Criterion'] = wx.ComboBox(self,value='friedman_mse', pos=(250, 225), choices=['friedman_mse','mae','mse'])
        
        self.GbParameters['MinSplitText'] = wx.StaticBox(self, label='Min Sample Split', pos=(240, 255), size=(120, 50))
        self.GbParameters['MinSplit'] = wx.SpinCtrl(self, value='4', pos=(250, 275), size=(50, -1), min=0, max=100)
        
        self.GbParameters['MinLeafText'] = wx.StaticBox(self, label='Min Sample Leaf', pos=(370, 55), size=(120, 50))
        self.GbParameters['MinLeaf'] = wx.SpinCtrl(self, value='2', pos=(380, 75), size=(50, -1), min=0, max=100)
        
        self.GbParameters['MaxFeatureText'] = wx.StaticBox(self, label='Max Feature', pos=(370, 105), size=(120, 50))
        self.GbParameters['MaxFeature'] = wx.ComboBox(self,value='sqrt', pos=(380, 125), choices=['sqrt','log2','auto'])
        
        Show(self.GbParameters,False)
        
    def Lr(self,e):
        Show(self.LrParameters,True)        
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
        self.lr.Value=True
        self.ridge.Value=False
        self.svr.Value=False
        self.knn.Value=False
        self.ann.Value=False
        self.rf.Value=False
        self.gb.Value=False
        
    def Ridge(self,e):
        Show(self.LrParameters,False) 
        Show(self.RidgeParameters,True)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
        self.lr.Value=False
        self.ridge.Value=True
        self.svr.Value=False
        self.knn.Value=False
        self.ann.Value=False
        self.rf.Value=False
        self.gb.Value=False
        
    def Svr(self,e):
        Show(self.LrParameters,False) 
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,True)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
        self.lr.Value=False
        self.ridge.Value=False
        self.svr.Value=True
        self.knn.Value=False
        self.ann.Value=False
        self.rf.Value=False
        self.gb.Value=False
        
    def Knn(self,e):
        Show(self.LrParameters,False) 
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,True)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
        self.lr.Value=False
        self.ridge.Value=False
        self.svr.Value=False
        self.knn.Value=True
        self.ann.Value=False
        self.rf.Value=False
        self.gb.Value=False
        
    def Ann(self,e):
        Show(self.LrParameters,False) 
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,True)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
        self.lr.Value=False
        self.ridge.Value=False
        self.svr.Value=False
        self.knn.Value=False
        self.ann.Value=True
        self.rf.Value=False
        self.gb.Value=False
        
    def Rf(self,e):
        Show(self.LrParameters,False) 
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,True)
        Show(self.GbParameters,False)
        
        self.lr.Value=False
        self.ridge.Value=False
        self.svr.Value=False
        self.knn.Value=False
        self.ann.Value=False
        self.rf.Value=True
        self.gb.Value=False
        
    def Gb(self,e):
        Show(self.LrParameters,False) 
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,True)
        
        self.lr.Value=False
        self.ridge.Value=False
        self.svr.Value=False
        self.knn.Value=False
        self.ann.Value=False
        self.rf.Value=False
        self.gb.Value=True

class CV_Feature_Selection(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        
        #Feature Selection
        wx.StaticBox(self, label='Feature Selcetion', pos=(5, 5), size=(480, 50))
        
        self.SelectionParameters={}
        wx.StaticText(self, label='Model:', pos=(10, 30))
        self.SelectionParameters['Model'] = wx.ComboBox(self, value='FBS', pos=(50, 25), choices=['FBS','FFS'])
        
        wx.StaticText(self, label='n_step:', pos=(110, 30))
        self.SelectionParameters['NStep'] = wx.SpinCtrl(self, value='1', pos=(150, 25), size=(50, -1), min=0, max=100)
        
        wx.StaticText(self, label='Metric:', pos=(210, 30))
        metric=['explained_variance','max_error','neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error','neg_mean_squared_log_error','neg_median_absolute_error','r2','neg_mean_poisson_deviance','neg_mean_gamma_deviance','neg_mean_absolute_percentage_error']
        self.SelectionParameters['Metric'] = wx.ComboBox(self, value='r2', pos=(250, 25), choices=metric)
        
        #Cross Validation
        wx.StaticBox(self, label='Cross Validation', pos=(5, 55), size=(480, 50))
        
        self.RepeatedParameters={}
        wx.StaticText(self, label='n_splits:', pos=(15, 80))
        self.RepeatedParameters['NSplit'] = wx.SpinCtrl(self, value='5', pos=(60, 75), size=(50, -1), min=0, max=100)
        
        wx.StaticText(self, label='n_repeats:', pos=(175, 80))
        self.RepeatedParameters['NRepeat'] = wx.SpinCtrl(self, value='6', pos=(230, 75), size=(50, -1), min=0, max=100)
        
        wx.StaticText(self, label='random_state:', pos=(345, 80))
        self.RepeatedParameters['RandomState'] = wx.SpinCtrl(self, value='22', pos=(425, 75), size=(50, -1), min=0, max=100)
        
        #ML Models
        wx.StaticBox(self, label='Machine Learning Model', pos=(5, 105), size=(180, 200))
        
        #Linear Regression
        self.lr = wx.CheckBox(self, label='Linear Regression', pos=(15, 130))
        self.lr.Value=False
        self.lr.Bind(wx.EVT_CHECKBOX,self.Lr)
        
        self.LrParameters={}
        self.LrParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 105), size=(120, 50))
        self.LrParameters['Scaler'] = wx.ComboBox(self,value='None', pos=(250, 125), choices=['MinMax','Standard','Robust','None'])
        
        Show(self.LrParameters,False)
        
        #Ridge
        self.ridge = wx.CheckBox(self, label='Ridge Regression', pos=(15, 155))
        self.ridge.Value=False
        self.ridge.Bind(wx.EVT_CHECKBOX,self.Ridge)
        
        self.RidgeParameters={}
        self.RidgeParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 105), size=(120, 50))
        self.RidgeParameters['Scaler'] = wx.ComboBox(self,value='None', pos=(250, 125), choices=['MinMax','Standard','Robust','None'])
        
        self.RidgeParameters['SolverText'] = wx.StaticBox(self, label='Solver', pos=(240, 155), size=(120, 50))
        self.RidgeParameters['Solver'] = wx.ComboBox(self,value='svd', pos=(250, 175), choices=['svd','cholesky','lsqr','sag'])
        
        self.RidgeParameters['AlphaText'] = wx.StaticBox(self, label='Logarit of Alpha', pos=(240, 205), size=(120, 50))
        self.RidgeParameters['Alpha'] = wx.SpinCtrl(self, value='10', pos=(250, 225), size=(50, -1), min=-100, max=100)
        
        self.RidgeParameters['FitInterceptText'] = wx.StaticBox(self, label='Fit Intercept', pos=(240, 255), size=(120, 50))
        self.RidgeParameters['FitIntercept'] = wx.ComboBox(self,value='No', pos=(250, 275), choices=['Yes','No'])
        
        Show(self.RidgeParameters,False)
        
        #SVR
        self.svr = wx.CheckBox(self, label='Support Vector Regression', pos=(15, 180))
        self.svr.Value=False
        self.svr.Bind(wx.EVT_CHECKBOX,self.Svr)
        
        self.SvrParameters={}
        self.SvrParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 105), size=(120, 50))
        self.SvrParameters['Scaler'] = wx.ComboBox(self,value='MinMax', pos=(250, 125), choices=['MinMax','Standard','Robust','None'])
        
        self.SvrParameters['KernelText'] = wx.StaticBox(self, label='Kernel', pos=(240, 155), size=(120, 50))
        self.SvrParameters['Kernel'] = wx.ComboBox(self,value='poly', pos=(250, 175), choices=['linear','poly','rbf','sigmoid','precomputed'])
        
        
        self.SvrParameters['GammaText'] = wx.StaticBox(self, label='Gamma', pos=(240, 205), size=(120, 50))
        self.SvrParameters['Gamma'] = wx.ComboBox(self,value='auto', pos=(250, 225), choices=['scale','auto'])
        
        self.SvrParameters['CText'] = wx.StaticBox(self, label='C', pos=(240, 255), size=(120, 50))
        self.SvrParameters['C'] = wx.SpinCtrl(self, value='4', pos=(250, 275), size=(50, -1), min=1, max=10)
        
        self.SvrParameters['EpsilonText'] = wx.StaticBox(self, label='100 x Epsilon', pos=(370, 105), size=(120, 50))
        self.SvrParameters['Epsilon'] = wx.SpinCtrl(self, value='5', pos=(380, 125), size=(50, -1), min=0, max=100)
        
        Show(self.SvrParameters,False)
        
        #KNN
        self.knn = wx.CheckBox(self, label='K Nearest Neighbors', pos=(15, 205))
        self.knn.Bind(wx.EVT_CHECKBOX,self.Knn)
        
        self.KnnParameters={}
        self.KnnParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 105), size=(120, 50))
        self.KnnParameters['Scaler'] = wx.ComboBox(self,value='MinMax', pos=(250, 125), choices=['MinMax','Standard','Robust', 'None'])
        
        self.KnnParameters['DimText'] = wx.StaticBox(self, label='Reduce Dim', pos=(240, 155), size=(120, 50))
        self.KnnParameters['Dim'] = wx.SpinCtrl(self, value='1', pos=(250, 175), size=(50, -1), min=0, max=100)
        
        self.KnnParameters['NeighborText'] = wx.StaticBox(self, label='Number of Neighbors', pos=(240, 205), size=(120, 50))
        self.KnnParameters['Neighbor'] = wx.SpinCtrl(self, value='7', pos=(250, 225), size=(50, -1), min=0, max=100)
        
        self.KnnParameters['WeightText'] = wx.StaticBox(self, label='Weight', pos=(240, 255), size=(120, 50))
        self.KnnParameters['Weight'] = wx.ComboBox(self,value='distance', pos=(250, 275), choices=['distance','uniform'])
        
        self.KnnParameters['AlgorithmText'] = wx.StaticBox(self, label='Algorithm', pos=(370, 105), size=(120, 50))
        self.KnnParameters['Algorithm'] = wx.ComboBox(self,value='auto', pos=(380, 125), choices=['auto','ball_tree','kd_tree','brute'])
        
        self.KnnParameters['LeafText'] = wx.StaticBox(self, label='Leaf Size', pos=(370, 155), size=(120, 50))
        self.KnnParameters['Leaf'] = wx.SpinCtrl(self, value='20', pos=(380, 175), size=(50, -1), min=0, max=100)
        
        Show(self.KnnParameters,False)
        
        #ANN
        self.ann = wx.CheckBox(self, label='Artificial Neural Network', pos=(15, 230))
        self.ann.Bind(wx.EVT_CHECKBOX,self.Ann)
        
        self.AnnParameters={}
        self.AnnParameters['ScalerText'] = wx.StaticBox(self, label='Scaler', pos=(240, 105), size=(120, 50))
        self.AnnParameters['Scaler'] = wx.ComboBox(self,value='MinMax', pos=(250, 125), choices=['MinMax','Standard','Robust','None'])
        
        self.AnnParameters['DimText'] = wx.StaticBox(self, label='Reduce Dim', pos=(240, 155), size=(120, 50))
        self.AnnParameters['Dim'] = wx.SpinCtrl(self, value='1', pos=(250, 175), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['AlphaText'] = wx.StaticBox(self, label='Logarit of Alpha', pos=(240, 205), size=(120, 50))
        self.AnnParameters['Alpha'] = wx.SpinCtrl(self, value='-2', pos=(250, 225), size=(50, -1), min=-100, max=100)
        
        self.AnnParameters['HiddenLayerText'] = wx.StaticBox(self, label='Hidden Layer', pos=(240, 255), size=(120, 50))
        self.AnnParameters['HiddenLayer1'] = wx.SpinCtrl(self, value='30', pos=(250, 275), size=(50, -1), min=0, max=100)
        self.AnnParameters['HiddenLayer2'] = wx.SpinCtrl(self, value='20', pos=(305, 275), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['ActivationText'] = wx.StaticBox(self, label='Activation', pos=(370, 105), size=(120, 42))
        self.AnnParameters['Activation'] = wx.ComboBox(self,value='relu', pos=(380, 120), choices=['relu','tanh'])
        
        self.AnnParameters['SolverText'] = wx.StaticBox(self, label='Solver', pos=(370, 146), size=(120, 42))
        self.AnnParameters['Solver'] = wx.ComboBox(self,value='lbfgs', pos=(380, 161), choices=['lbfgs','adam'])
        
        self.AnnParameters['IterText'] = wx.StaticBox(self, label='Logarit of Max_iter', pos=(370, 187), size=(120, 48))
        self.AnnParameters['Iter'] = wx.SpinCtrl(self, value='4', pos=(380, 207), size=(50, -1), min=0, max=100)
        
        self.AnnParameters['MomentumText'] = wx.StaticBox(self, label='10 x Momentum', pos=(370, 234), size=(120, 48))
        self.AnnParameters['Momentum'] = wx.SpinCtrl(self, value='9', pos=(380, 254), size=(50, -1), min=0, max=10)
        
        self.AnnParameters['NesterovsText'] = wx.StaticBox(self, label='NesterovMomentum', pos=(370, 281), size=(120, 42))
        self.AnnParameters['Nesterovs'] = wx.ComboBox(self,value='Yes', pos=(380, 296), choices=['Yes','No'])
        
        Show(self.AnnParameters,False)
        
        #Random Forest
        self.rf = wx.CheckBox(self, label='Random Forest', pos=(15, 255))
        self.rf.Bind(wx.EVT_CHECKBOX,self.Rf)
        
        self.RfParameters={}
        self.RfParameters['EstimatorText'] = wx.StaticBox(self, label='Number of Estimator', pos=(240, 105), size=(120, 50))
        self.RfParameters['Estimator'] = wx.SpinCtrl(self, value='240', pos=(250, 125), size=(50, -1), min=0, max=1000)
        
        self.RfParameters['CriterionText'] = wx.StaticBox(self, label='Criterion', pos=(240, 155), size=(120, 50))
        self.RfParameters['Criterion'] = wx.ComboBox(self,value='absolute_error', pos=(250, 175), choices=['absolute_error','squared_error'])
        
        self.RfParameters['MinSplitText'] = wx.StaticBox(self, label='Min Sample Split', pos=(240, 205), size=(120, 50))
        self.RfParameters['MinSplit'] = wx.SpinCtrl(self, value='2', pos=(250, 225), size=(50, -1), min=0, max=100)
        
        self.RfParameters['DepthText'] = wx.StaticBox(self, label='Max Depth', pos=(240, 255), size=(120, 50))
        self.RfParameters['Depth'] = wx.SpinCtrl(self, value='9', pos=(250, 275), size=(50, -1), min=0, max=100)
        
        self.RfParameters['MinLeafText'] = wx.StaticBox(self, label='Min Sample Leaf', pos=(370, 105), size=(120, 50))
        self.RfParameters['MinLeaf'] = wx.SpinCtrl(self, value='1', pos=(380, 125), size=(50, -1), min=0, max=100)
        
        self.RfParameters['MaxFeatureText'] = wx.StaticBox(self, label='Max Feature', pos=(370, 155), size=(120, 50))
        self.RfParameters['MaxFeature'] = wx.ComboBox(self,value='sqrt', pos=(380, 175), choices=['sqrt','log2','auto'])
        
        self.RfParameters['BootstrapText'] = wx.StaticBox(self, label='Bootstrap', pos=(370, 205), size=(120, 50))
        self.RfParameters['Bootstrap'] = wx.ComboBox(self,value='No', pos=(380, 225), choices=['Yes','No'])
        
        Show(self.RfParameters,False)
        
        #Gradient Boosting
        self.gb = wx.CheckBox(self, label='Gradient Boosting', pos=(15, 280))
        self.gb.Bind(wx.EVT_CHECKBOX,self.Gb)
        
        self.GbParameters={}
        self.GbParameters['EstimatorText'] = wx.StaticBox(self, label='Number of Estimator', pos=(240, 105), size=(120, 50))
        self.GbParameters['Estimator'] = wx.SpinCtrl(self, value='120', pos=(250, 125), size=(50, -1), min=0, max=1000)
        
        self.GbParameters['DepthText'] = wx.StaticBox(self, label='Max Depth', pos=(240, 155), size=(120, 50))
        self.GbParameters['Depth'] = wx.SpinCtrl(self, value='7', pos=(250, 175), size=(50, -1), min=0, max=100)
        
        self.GbParameters['LearningText'] = wx.StaticBox(self, label='1000 x Learning Rate', pos=(240, 205), size=(120, 50))
        self.GbParameters['Learning'] = wx.SpinCtrl(self, value='88', pos=(250, 225), size=(50, -1), min=0, max=1000)
        
        self.GbParameters['CriterionText'] = wx.StaticBox(self, label='Criterion', pos=(240, 255), size=(120, 50))
        self.GbParameters['Criterion'] = wx.ComboBox(self,value='friedman_mse', pos=(250, 275), choices=['friedman_mse','mae','mse'])
        
        self.GbParameters['MinSplitText'] = wx.StaticBox(self, label='Min Sample Split', pos=(370, 105), size=(120, 50))
        self.GbParameters['MinSplit'] = wx.SpinCtrl(self, value='4', pos=(380, 125), size=(50, -1), min=0, max=100)
        
        self.GbParameters['MinLeafText'] = wx.StaticBox(self, label='Min Sample Leaf', pos=(370, 155), size=(120, 50))
        self.GbParameters['MinLeaf'] = wx.SpinCtrl(self, value='2', pos=(380, 175), size=(50, -1), min=0, max=100)
        
        self.GbParameters['MaxFeatureText'] = wx.StaticBox(self, label='Max Feature', pos=(370, 205), size=(120, 50))
        self.GbParameters['MaxFeature'] = wx.ComboBox(self,value='sqrt', pos=(380, 225), choices=['sqrt','log2','auto'])
        
        Show(self.GbParameters,False)
        
    def Lr(self,e):
        Show(self.LrParameters,True)        
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
        self.lr.Value=True
        self.ridge.Value=False
        self.svr.Value=False
        self.knn.Value=False
        self.ann.Value=False
        self.rf.Value=False
        self.gb.Value=False
        
    def Ridge(self,e):
        Show(self.LrParameters,False) 
        Show(self.RidgeParameters,True)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
        self.lr.Value=False
        self.ridge.Value=True
        self.svr.Value=False
        self.knn.Value=False
        self.ann.Value=False
        self.rf.Value=False
        self.gb.Value=False
        
    def Svr(self,e):
        Show(self.LrParameters,False) 
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,True)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
        self.lr.Value=False
        self.ridge.Value=False
        self.svr.Value=True
        self.knn.Value=False
        self.ann.Value=False
        self.rf.Value=False
        self.gb.Value=False
        
    def Knn(self,e):
        Show(self.LrParameters,False) 
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,True)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
        self.lr.Value=False
        self.ridge.Value=False
        self.svr.Value=False
        self.knn.Value=True
        self.ann.Value=False
        self.rf.Value=False
        self.gb.Value=False
        
    def Ann(self,e):
        Show(self.LrParameters,False) 
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,True)
        Show(self.RfParameters,False)
        Show(self.GbParameters,False)
        
        self.lr.Value=False
        self.ridge.Value=False
        self.svr.Value=False
        self.knn.Value=False
        self.ann.Value=True
        self.rf.Value=False
        self.gb.Value=False
        
    def Rf(self,e):
        Show(self.LrParameters,False) 
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,True)
        Show(self.GbParameters,False)
        
        self.lr.Value=False
        self.ridge.Value=False
        self.svr.Value=False
        self.knn.Value=False
        self.ann.Value=False
        self.rf.Value=True
        self.gb.Value=False
        
    def Gb(self,e):
        Show(self.LrParameters,False) 
        Show(self.RidgeParameters,False)
        Show(self.SvrParameters,False)
        Show(self.KnnParameters,False)
        Show(self.AnnParameters,False)
        Show(self.RfParameters,False)
        Show(self.GbParameters,True)
        
        self.lr.Value=False
        self.ridge.Value=False
        self.svr.Value=False
        self.knn.Value=False
        self.ann.Value=False
        self.rf.Value=False
        self.gb.Value=True
        
class Analyzing(wx.Panel):    
    def __init__(self, parent, title=None):
        wx.Panel.__init__(self, parent=parent)
        
        self.notebook = wx.Notebook(self,size=(505,350), pos=(5,5))
        self.tabOne = Grid_Search(self.notebook)
        self.notebook.AddPage(self.tabOne, "Grid Search")
        
        self.tabTwo = Train_Evaluation(self.notebook)
        self.notebook.AddPage(self.tabTwo, "Train Evaluation")
        
        self.tabThree = Cross_Validation(self.notebook)
        self.notebook.AddPage(self.tabThree, "Cross Validation")
        
        self.tabFour = Feature_Selection(self.notebook)
        self.notebook.AddPage(self.tabFour, "Feature Selection")
        
        self.tabFive = CV_Feature_Selection(self.notebook)
        self.notebook.AddPage(self.tabFive, "CV_Feature Selection")
        
    
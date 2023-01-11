import wx
import os
import sys
    
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import Frontend.Loading as Loading
import Frontend.Analysis as Analysis
import Backend.Data_Loading as Data_Loading
import Backend.Grid_Search as Grid_Search
import Backend.Train_Evaluation as Train_Evaluation
import Backend.Cross_Validation as Cross_Validation
import Backend.Feature_Selection as Feature_Selection
import Backend.CV_Feature_Selection as CV_Feature_Selection

class WizardPanel(wx.Panel):
    #----------------------------------------------------------------------
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.pages = []
        self.page_num = 0
        
        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.panelSizer = wx.BoxSizer(wx.VERTICAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
                
        # add prev/next buttons
        self.prevBtn = wx.Button(self, label="Previous",pos=(340, 360), size=(80, 25))
        self.prevBtn.Bind(wx.EVT_BUTTON, self.onPrev)
        self.prevBtn.Show(False)
        
        self.nextBtn = wx.Button(self, label="Next",pos=(425, 360), size=(80, 25))
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        
        # finish layout
        self.mainSizer.Add(self.panelSizer, 1, wx.EXPAND)
        self.mainSizer.Add(btnSizer, 0, wx.ALIGN_RIGHT)
        self.SetSizer(self.mainSizer)
        
        
    #----------------------------------------------------------------------
    def LoadingPage(self, title=None):
        self.Loading = Loading.Loading(self, title)
        self.panelSizer.Add(self.Loading, 2, wx.EXPAND)
        self.pages.append(self.Loading)
        if len(self.pages) > 1:
            # hide all panels after the first one
            self.Loading.Hide()
            self.Layout()
            
    #----------------------------------------------------------------------
    def AnalysisPage(self, title=None):
        self.Analyzing = Analysis.Analyzing(self, title)
        self.panelSizer.Add(self.Analyzing, 2, wx.EXPAND)
        self.pages.append(self.Analyzing)
        if len(self.pages) > 1:
            # hide all panels after the first one
            self.Analyzing.Hide()
            self.Layout()
            
    #----------------------------------------------------------------------
    def onNext(self, event):
        self.prevBtn.Show()
        
        pageCount = len(self.pages)
        if pageCount-1 != self.page_num:
            self.pages[self.page_num].Hide()
            self.page_num += 1
            self.pages[self.page_num].Show()
            self.panelSizer.Layout()
            
        if self.nextBtn.GetLabel() == "Finish":
            if self.Loading.PathBox.Items!=[]:
                self.train, self.test, self.trainX, self.trainy, self.testX, self.testy, self.traintest, self.traintestX, self.traintesty, self.cutoff1, self.cutoff2= Data_Loading.DataLoading(self.Loading.paths[0],int(self.Loading.SplitRatio.GetValue()),self.Loading.YBox.Items,self.Loading.XBox.Items,self.Loading.Subfeatures)
            else:
                self.train, self.test, self.trainX, self.trainy, self.testX, self.testy, self.traintest, self.traintestX, self.traintesty, self.cutoff1, self.cutoff2= Data_Loading.DataLoading2(self.Loading.trainpaths[0],self.Loading.testpaths[0],self.Loading.YBox.Items,self.Loading.XBox.Items,self.Loading.Subfeatures)

            if self.Analyzing.notebook.GetSelection() == 0:
                Grid_Search.GridSearch(self)
            elif self.Analyzing.notebook.GetSelection() == 1:
                Train_Evaluation.train_evaluate(self.Analyzing.tabTwo, self.train, self.test, self.trainX, self.trainy, self.testX, self.testy, self.traintest, self.traintestX, self.traintesty, self.cutoff1, self.cutoff2, self.Loading.YBox.Items, self.Loading.XBox.Items, self.Loading.Subfeatures, self.Loading.InputName)
            elif self.Analyzing.notebook.GetSelection() == 2:
                Cross_Validation.cross_validation(self.Analyzing.tabThree, self.train, self.test, self.trainX, self.trainy, self.testX, self.testy, self.traintest, self.traintestX, self.traintesty, self.cutoff1, self.cutoff2, self.Loading.YBox.Items, self.Loading.XBox.Items, self.Loading.Subfeatures, self.Loading.InputName)
            elif self.Analyzing.notebook.GetSelection() == 3:
                if self.Analyzing.tabFour.SelectionParameters['Model'].GetValue()=='Feature Forward Selection':
                    Feature_Selection.Feature_Forward_Selection(self.Analyzing.tabFour, self.trainX, self.trainy, self.testX, self.testy, self.traintest, self.cutoff1, self.cutoff2, self.Loading.YBox.Items, self.Loading.XBox.Items, self.Loading.Subfeatures, self.Loading.InputName)                
                else:
                    Feature_Selection.Feature_Backward_Selection(self.Analyzing.tabFour, self.trainX, self.trainy, self.testX, self.testy, self.traintest, self.cutoff1, self.cutoff2, self.Loading.YBox.Items, self.Loading.XBox.Items, self.Loading.Subfeatures, self.Loading.InputName)                                  
            elif self.Analyzing.notebook.GetSelection() == 4:
                if self.Analyzing.tabFive.SelectionParameters['Model'].GetValue()=='FFS':
                    CV_Feature_Selection.CV_Feature_Forward_Selection(self.Analyzing.tabFive, self.traintestX, self.traintesty, self.traintest, self.cutoff1, self.cutoff2, self.Loading.YBox.Items, self.Loading.XBox.Items, self.Loading.Subfeatures, self.Loading.InputName)                
                else:
                    CV_Feature_Selection.CV_Feature_Backward_Selection(self.Analyzing.tabFive, self.traintestX, self.traintesty, self.traintest, self.cutoff1, self.cutoff2, self.Loading.YBox.Items, self.Loading.XBox.Items, self.Loading.Subfeatures, self.Loading.InputName)                                  
                   
            self.GetParent().Close()
            
        if pageCount == self.page_num+1:
            # change label
            self.nextBtn.SetLabel("Finish")
    
    #----------------------------------------------------------------------
    def onPrev(self, event):
        pageCount = len(self.pages)
        if self.page_num-1 != -1:
            self.pages[self.page_num].Hide()
            self.page_num -= 1
            self.pages[self.page_num].Show()
            self.panelSizer.Layout()
        
        if pageCount != self.page_num+1:
            # change label
            self.nextBtn.SetLabel("Next")
        
        if self.page_num == 0:
            self.prevBtn.Show(False)
        
    
########################################################################
class MainFrame(wx.Frame):
    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, title="Machine Learning Regression", size=(530,435),style = wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        
        self.panel = WizardPanel(self)
        self.panel.LoadingPage("Page 1")
        self.panel.AnalysisPage("Page 2")
        
        self.Show()
    
if __name__ == "__main__":
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()
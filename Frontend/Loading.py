#C:\Users\USER\Desktop\LA-LB-hung\data\23LB12LA_HOMO.csv
import wx
import pandas as pd
import os

wildcard ="All files (*.*)|*.*"

class Loading(wx.Panel):
    
    def __init__(self, parent, title=None):
        wx.Panel.__init__(self, parent=parent)  
        self.currentDirectory = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'Data')
        
        wx.StaticBox(self, label='Full Dataset Input', pos=(5, 5), size=(245, 90))
        OpenButton = wx.Button(self, label='Open File', pos=(10, 25), size=(80, 30))
        OpenButton.Bind(wx.EVT_BUTTON, self.onOpenFile)
        self.PathBox = wx.ListBox(self, pos=(95,25), size=(150,30))
        wx.StaticText(self,label='Split Ratio:  1 /',pos=(25,63))
        self.SplitRatio = wx.SpinCtrl(self, value='5', pos=(105, 60), size=(50, -1), min=2, max=10)
        wx.StaticText(self,label='(Test / Full Data)',pos=(160,63))
        
        wx.StaticBox(self, label='Train + Test Dataset Input', pos=(255, 5), size=(245, 90))
        TrainButton = wx.Button(self, label='Train Data', pos=(260, 25), size=(80, 30))
        TrainButton.Bind(wx.EVT_BUTTON, self.onTrainFile)
        self.TrainBox = wx.ListBox(self, pos=(345,25), size=(150,30))
        TestButton = wx.Button(self, label='Test Data', pos=(260, 60), size=(80, 30))
        TestButton.Bind(wx.EVT_BUTTON, self.onTestFile)
        self.TestBox = wx.ListBox(self, pos=(345,60), size=(150,30))
        
        wx.StaticBox(self, label='Feature Box', pos=(5, 95), size=(175, 260))
        self.FeatureBox = wx.ListBox(self, pos=(10,115),size=(165,160))
        MoveYButton = wx.Button(self, label='Move to Y', pos=(10, 280), size=(80, 30))
        MoveYButton.Bind(wx.EVT_BUTTON, self.moveY)
        MoveXButton = wx.Button(self, label='Move to X', pos=(95, 280), size=(80, 30))
        MoveXButton.Bind(wx.EVT_BUTTON, self.moveX)
        MoveAllYButton = wx.Button(self, label='Move All to Y', pos=(10, 315), size=(80, 30))
        MoveAllYButton.Bind(wx.EVT_BUTTON, self.moveallY)
        MoveAllXButton = wx.Button(self, label='Move All to X', pos=(95, 315), size=(80, 30))
        MoveAllXButton.Bind(wx.EVT_BUTTON, self.moveallX)
        
        wx.StaticBox(self, label='Y Box', pos=(190, 95), size=(150, 260))
        self.YBox = wx.ListBox(self, pos=(195,115),size=(140,160))
        YRemoveButton = wx.Button(self, label='Remove', pos=(195, 280), size=(80, 30))
        YRemoveButton.Bind(wx.EVT_BUTTON, self.Yremove)
        YRemoveAllButton = wx.Button(self, label='Remove All', pos=(195, 315), size=(80, 30))
        YRemoveAllButton.Bind(wx.EVT_BUTTON, self.Yremoveall)
        
        wx.StaticBox(self, label='X Box', pos=(350, 95), size=(150, 260))
        self.XBox = wx.ListBox(self, pos=(355,115),size=(140,160))
        XRemoveButton = wx.Button(self, label='Remove', pos=(355, 280), size=(80, 30))
        XRemoveButton.Bind(wx.EVT_BUTTON, self.Xremove)
        XRemoveAllButton = wx.Button(self, label='Remove All', pos=(355, 315), size=(80, 30))
        XRemoveAllButton.Bind(wx.EVT_BUTTON, self.Xremoveall)
    
    def onOpenFile(self, e):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory, 
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        if dlg.ShowModal() == wx.ID_OK:
            self.paths = dlg.GetPaths()
            self.InputName=os.path.basename(self.paths[0])
            self.PathBox.Append(self.InputName)
            
            contents=pd.read_csv(self.paths[0])
            features=contents.columns
            self.Subfeatures=[]
            for i in features:
                if i[0]!='#':
                    self.FeatureBox.Append(i)
                else:
                    self.Subfeatures.append(i)
        dlg.Destroy()
    
    def onTrainFile(self, e):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory, 
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        if dlg.ShowModal() == wx.ID_OK:
            self.trainpaths = dlg.GetPaths()
            self.InputName=os.path.basename(self.trainpaths[0])
            self.TrainBox.Append(self.InputName)
            
            contents=pd.read_csv(self.trainpaths[0])
            features=contents.columns
            self.Subfeatures=[]
            for i in features:
                if i[0]!='#':
                    self.FeatureBox.Append(i)
                else:
                    self.Subfeatures.append(i)
        dlg.Destroy()
    
    def onTestFile(self, e):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory, 
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        if dlg.ShowModal() == wx.ID_OK:
            self.testpaths = dlg.GetPaths()
            self.InputName=os.path.basename(self.testpaths[0])
            self.TestBox.Append(self.InputName)
        dlg.Destroy()
    
    def moveY(self, e):
        sel = self.FeatureBox.GetSelection()
        self.YBox.Append(self.FeatureBox.GetString(sel))
        if sel != -1:
            self.FeatureBox.Delete(sel)
    
    def moveX(self, e):
        sel = self.FeatureBox.GetSelection()
        self.XBox.Append(self.FeatureBox.GetString(sel))
        if sel != -1:
            self.FeatureBox.Delete(sel)
    
    def moveallY(self,e):
        for i in self.FeatureBox.Items:
            self.YBox.Append(i)
        self.FeatureBox.Clear()
        
    def moveallX(self,e):
        for i in self.FeatureBox.Items:
            self.XBox.Append(i)
        self.FeatureBox.Clear()
    
    def Yremove(self, e):
        sel = self.YBox.GetSelection()
        self.FeatureBox.Append(self.YBox.GetString(sel))
        if sel != -1:
            self.YBox.Delete(sel)
            
    def Yremoveall(self, e):
        for i in self.YBox.Items:
            self.FeatureBox.Append(i)
        self.YBox.Clear()
        
    def Xremove(self, e):
        sel = self.XBox.GetSelection()
        self.FeatureBox.Append(self.XBox.GetString(sel))
        if sel != -1:
            self.XBox.Delete(sel)
            
    def Xremoveall(self, e):
        for i in self.XBox.Items:
            self.FeatureBox.Append(i)
        self.XBox.Clear()
 
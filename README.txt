------------------------------------------REQUIREMENT-------------------------------------------------------------------------------

Verified operating system: Windows

Pyhton version: python3

Library requirement: wxPython, scikit-learn, pandas, numpy, os, sys

------------------------------------------INSTRUCTION-------------------------------------------------------------------------------

1. Put your dataset (your dataset needs to be in the csv format) in 'Data' folder before running.

2. Open the 'Main.py' file with a Python application and run the code.

3. After the interface is completely loaded, select one of following approaches to input your dataset:
	
   3.1. Full Dataset Input:
	Click on 'Open File' Button in the 'Full Dataset Input' section to choose your dataset.
	The split ratio can be modified by changing the number in the corresponding box.

   3.2. Train + Test Dataset Input:
	Respectively click on 'Train Data' and 'Test Data' buttons to select your train and test sets.
	NOTE: the features and their arrangement should be identical in train and test sets.

4. When the dataset input process is completed, features of your dataset will appear in the 'Feature Box' (features with '#' as the initial character will not appear).
   Use the buttons below boxes to select your Y feature and X features (this code only works with one Y feature).
   Then, click on 'Next' button to move forward.

5. There are 5 analyzing options in this code as 5 tabs on the head of the interface.
   The code is only able to execute 1 analyzing option at one time. Select your option on the according tab.

6. Select the model and modify its hyperparameters of your option to be more suitable with your dataset.

7. Click on 'Finish' button to execute the analysis. The interface will close after the analysis is done.
   Results of the analysis will be saved in the 'Result' folder.
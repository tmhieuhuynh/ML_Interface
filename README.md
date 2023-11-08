# ML_Interface: A useful interface for Scikit-learn ML models
## Usage notes
Verified operating system: Windows

Pyhton version: python3

Dependencies: wxPython, scikit-learn, pandas, numpy, os, sys

## Installation
### Dependent Library Installation
- wxPython
```bash
pip install -U wxPython
```
- scikit-learn
```bash
pip install -U scikit-learn
```
- pandas
```bash
pip install pandas
```
- numpy
```bash
pip install numpy
```
### ML_Interface Installation
```bash
git clone https://github.com/Serendipity12345/ML_Interface
```
## Tutorials
1. Put your datasets (your datasets need to be in the csv format) in the 'Data' folder before running.

2. Run the 'Main.py' file with a Python application.

3. After the interface is completely loaded, select one of the following approaches to input your dataset:
	
   3.1. Full Dataset Input:
	Click on the 'Open File' Button in the 'Full Dataset Input' section to choose your dataset.
	The split ratio can be modified by changing the number in the corresponding box.

   3.2. Train + Test Dataset Input:
	Respectively click on the 'Train Data' and 'Test Data' buttons to select your train and test sets.
	NOTE: the features and their arrangement should be identical in train and test sets.

4. When the dataset input process is completed, features of your dataset will appear in the 'Feature Box' (features with '#' as the initial character will not appear).
   Use the buttons below boxes to select your Y feature and X features (this code only works with one Y feature).
   Then, click on the 'Next' button to move forward.

5. There are 5 analyzing options in this code as 5 tabs on the head of the interface.
   The code is only able to execute 1 analyzing option at one time. Select your option on the according tab.

6. Select the model and modify its hyperparameters of your option to be more suitable with your dataset.

7. Click on the 'Finish' button to execute the analysis. The interface will close after the analysis is done.
   The results of the analysis will be saved in the 'Result' folder.

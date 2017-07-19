# sciblox
An all in one Python3 Data Science Package. Easy visualisation, data mining, data preparation and machine learning.

Please check the Jupyter Notebook for instructions on how to use it.
Open the HTML file if you want to see extra styling.

Some features explained include:

1. MICE, BPCA missing data imputation with Random Forests, XGBoost and Linear Regression support
2. Automatic Data Plotting
3. Word extraction and frequency plots
4. Sequential text processing
5. CARET like processes including ZeroVarCheck, FreqRatios etc.
6. Discretization and Continuisation
7. Easy data structure changes like Hcat, Vcat, reversing etc.

Features in construction:

1. Easy CARET like Machine Learning modules
2. Automatic Best Graphs Plotting
3. Advanced text extraction methods
4. Automatic Machine Learning methods

***TO INSTALL***:
If you are using ANACONDA Python, place sciblox.py in C:\ProgramData\Anaconda3\Lib.
Then, for easier calling:
```python
from sciblox import *
%matplotlib inline
```
If you are using other methods, just copy paste sciblox.py into whatever Python3 main directory.
Then call it same as top.

***PREREQS***:
1. Numpy
2. Pandas
3. Matplotlib
4. Seaborn
5. Sklearn
6. (If you use KNN or SVD imputation) - FancyImpute AND Theano
7. Scipy
8. XGBoost

Some screenshots:

![Plotting](/Plot.jpg?raw=true "Auto Plotting")

![Imputing](/impute.jpg?raw=true "MICE,BPCA Missing Data Imputation")

![Analysing](/Analyse.jpg?raw=true "Clear descriptive statistics")

![Data_Mining](/datamining.jpg?raw=true "Sequential Data Mining")

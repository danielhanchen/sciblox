# sciblox
An all in one Python3 Data Science Package. Easy visualisation, data mining, data preparation and machine learning.

Please check the Jupyter Notebook for instructions on how to use it.
You can also check sciblox out on https://danielhanchen.github.io/

https://pypi.python.org/pypi/sciblox

Install:
```sh
[sudo] pip install sciblox
```
NOTE: If you intend to use remove linearly dependent rows or KNN,SVD impute:
```sh
[sudo] pip install fancyimpute sympy theano
```
If fancyimpute fails: Please install C++ or MingW compiler


WHAT'S NEW?
1. FASTER (x10) BPCA fill
2. Better analyser
3. NEW modules - Machine Learning

Some features explained include:

1. MICE, BPCA missing data imputation with Random Forests, XGBoost and Linear Regression support
2. Automatic Data Plotting
3. Word extraction and frequency plots
4. Sequential text processing
5. CARET like processes including ZeroVarCheck, FreqRatios etc.
6. Discretization and Continuisation
7. Easy data structure changes like Hcat, Vcat, reversing etc.
8. Easy CARET like Machine Learning modules
9. Automatic Best Graphs Plotting

IN CONSTRUCTION:
1. Advanced text extraction methods
2. Automatic Machine Learning methods

For easier calling:
```python
from sciblox import *
%matplotlib notebook
```
If you are using other methods, just copy paste sciblox.py into whatever Python3 main directory.
Then call it same as top.

Some screenshots:

![Analysing](/img/Analyse.jpg?raw=true "Auto analysing and 3d plots")

![Preprocessing](/img/Preprocess.jpg?raw=true "CARET like Preprocess")

![Analytics](/img/Analytics.jpg?raw=true "CARET like checking")

![Plotting](/img/Plot.jpg?raw=true "Cool easy plots")

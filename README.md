# CALIFORNIA-HOUSE-PRICE-PREDICTION
-----
The aim of this project is to predict the prices of the houses in California based on a dataset from sklearn using Machine Learning.

# Libraries used:-
-----
- **Numpy**

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.001.png)

**Importing Numpy Library**

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.002.png)

**About Numpy**

Numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

- **Pandas**

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.003.png)

**Importing Pandas Library**

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.004.png)

**About Pandas**

Pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real-world data analysis in Python.

- **Sklearn**

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.005.png)

**Importing Sklearn functions**

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.006.png)

**About Sklearn**

Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistent interface in Python. This library, which is largely written in Python, is built upon NumPy, SciPy and Matplotlib.

**Different functions imported from Sklearn**

- **sklearn.datasets-** The sklearn.datasets package embeds some small toy datasets. This package also features helpers to fetch large datasets commonly used by the machine learning community to benchmark algorithms on data that comes from the 'real world'.
- **train\_test\_split-** Using [train_test_split()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) from the data science library [scikit-learn](https://scikit-learn.org/stable/index.html), you can split your dataset into subsets that minimize the potential for bias in your evaluation and validation process.
- **metrics-** This module implements several loss, score, and utility functions to measure classification performance. Some metrics might require probability estimates of the positive class, confidence values, or binary decisions values.

- **Matplolib.pyplot**

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.007.png)

**Importing Matplotlib.pyplot**

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.008.png)

**About Matplotlib.pyplot**

Matplotlib.pyplot is a collection of functions that make matplotlib work like MATLAB. Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels, etc.

- **Seaborn**

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.009.png)

**Importing Seaborn** 

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.010.png)

**About Seaborn**

Seaborn is a library for making statistical graphics in Python. It builds on top of matplotlib and integrates closely with pandas data structures.It helps you explore and understand your data. Its plotting functions operate on dataframes and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots.

# Algorithm Used:-
-----
- **XG Boost Regressor**

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.011.png)

**Importing XG Boost Regressor**

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.012.png)

**About XG Boost Regressor**

XGBoost is a powerful approach for building supervised regression models. XGBoost minimizes a regularized (L1 and L2) objective function that combines a convex loss function (based on the difference between the predicted and target outputs) and a penalty term for model complexity (in other words, the regression tree functions).






# Dataset Used:-
-----
- **fetch\_california\_housing()**

This dataset consists of 20,640 samples and 9 features.

**Fetching dataset from sklearn.datasets**

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.013.png)

# Data Analysis:-
-----
- Heat Map based on correlation between various features

White boxes- represents Negative Correlation

Dark Blue boxes- represents Positive Correlation

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.014.png)

# Model Analysis:-
-----
- Based on training data

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.015.png)

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.016.png)

- Based on test data

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.017.png)

![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.018.png)

# A home-grown success!
-----
![](Aspose.Words.29c59542-abd5-49dd-9c3a-ff87baef20f9.019.png)

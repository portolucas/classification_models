<h3>There are Classification Models Algorithms used in Data Mining and Machinne Learning, implement in Python with Sklearn, Pandas, Numpy and Matplotlib</h3>

This is a experimental implmentation. You can use in your search, but <strong>, exist best implementations</strong> around internet.

But you need learn now, let's go.

For search the perfect number of clusters, the Elbow method was used. Read more <a href="https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/">here</a>.

First, install <a href="https://www.python.org/downloads/">Python</a>.

Second, install <a href="https://pandas.pydata.org/">Pandas</a>, <a href="https://numpy.org/">Numpy</a>, <a href="https://matplotlib.org/">Matplotlib</a> and <a href="http://scikit-learn.org/stable/install.html">Sklearn</a>.

Now, create a folder and drop the dataset cancer.csv. This dataset is about breast cancer and are discreetly as you can see how at third step. You can see a original dataset <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/">here</a>. Our objective is find relations with different data about breast cancer and test the algorithm acurracy.

We used diferent algorithm for classification. It may be important to understand predictable which model to use to save time and resource. Interesting metric is estimating the accuracy of models using samples for practice and test.

Third, read the data with Pandas. The parameter sep=";" is one option. The more comum is comma. To know if everthing is right, call method .head().

Fourth, creat trainning and test sets with data we want to compare X with y. The data scale is different, so we need make them equivalent.

Fifith, we can instance the algorithms. Each one model have a different write form. You can learn about each one:

- <a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html">Gaussian Naive Bayes</a>
- <a href="http://scikit-learn.org/stable/modules/tree.html">Decision Tree</a>
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">Random Forest</a>
- <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">Logistic Regression</a>

Now, we can plot the graphs. Used <a href="https://matplotlib.org/3.1.0/tutorials/introductory/pyplot.html">Matplotlib</a>.

Twitter: @feedlucas



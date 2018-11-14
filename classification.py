import pandas as pd
import numpy as np
cancer = pd.read_table('cancer.csv', sep=';')   #Read data and split with ;
cancer.head()   #Show head

#Distribuition
# import pylab as pl
# import matplotlib.pyplot as plt
# cancer.drop('age', axis=1).hist(bins=30, figsize=(9,9))
# pl.suptitle('Histogram for each numeric input variable')
# plt.savefig('cancer_age_hist')
# plt.show()

# Create training and test sets
from sklearn.model_selection import train_test_split
X = cancer[['age', 'tumor_size', 'inv_nodes', 'dreg_malign']]
y = cancer['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Apply scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ** -- Build Models of Classifier -- **

# k-NN Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print('Acuraccy of K-NN classifier on training set: {:.2f}'
      .format(knn.score(X_train, y_train)))
print('Acurracy of K-NN classifier on test set: {:.2f}'
      .format(knn.score(X_test, y_test)))

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Acuracy of GNB classifier on training set: {:.2f}'
      .format(gnb.score(X_train, y_train)))
print('Acuracy of GNB classifier on test set: {:.2f}'
      .format(gnb.score(X_test, y_test)))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print('Acuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Acuracy of Decision Tree classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
print('Acuracy of Random Forest classifier on training set: {:.2f}'
      .format(rf.score(X_test, y_test)))
print('Acuracy of Random Forest classifier on test set: {:.2f}'
      .format(rf.score(X_test, y_test)))


# Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
print('Acuracy of Logistic regression classifier on training set: {:.2f}'
      .format(logreg.score(X_train, y_train)))
print('Acuracy of Logistic regression classifier on set test: {:.2f}'
      .format(logreg.score(X_test, y_test)))


# Find of perfect K (number of clusters)

k_range = range(1, 20)
scores = []
for k in k_range:
      knn = KNeighborsClassifier(n_neighbors = k)
      knn.fit(X_train, y_train)
      scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])

# Plot the Decision Boundary of the k-NN Classifier

from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from sklearn import neighbors
import matplotlib.pyplot as plt

X = cancer[['age', 'tumor_size', 'inv_nodes', 'dreg_malign']]
y = cancer['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_mat = X[['age', 'tumor_size']].values
y_mat = y.values

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#AFAFAF'])
cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#AFAFAF'])

clf = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
clf.fit(X_mat, y_mat)

mesh_step_size = .01  # step size in the mesh
plot_symbol_size = 50

x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                     np.arange(y_min, y_max, mesh_step_size))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor = 'black')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

patch0 = mpatches.Patch(color='#FF0000', label='apple')
patch1 = mpatches.Patch(color='#00FF00', label='mandarin')
patch2 = mpatches.Patch(color='#0000FF', label='orange')
patch3 = mpatches.Patch(color='#AFAFAF', label='lemon')
plt.legend(handles=[patch0, patch1, patch2, patch3])

plt.xlabel('age (cm)')
plt.ylabel('tumor_size (cm)')
plt.title("4-Class classification (k = %i, weights = '%s')"
          % (n_neighbors, weights))
print(plt.show())
print(plot_cancer_knn(X_train, y_train, 5, 'uniform'))































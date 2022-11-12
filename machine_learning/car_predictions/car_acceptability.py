# Load Libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from seaborn import countplot
import pickle

# Load Dataset
url = "https://raw.githubusercontent.com/dancanomondi/machine_learning/main/machine_learning/car_predictions/car.data"
names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
dataset = read_csv(url, names=names)

# Label Encoding: Label Encoder
label_encoder = LabelEncoder()

for i in dataset.columns:
    dataset[i] = label_encoder.fit_transform(dataset[i])

# Statistical Summary
# shape
print(dataset.shape)

# head
print(dataset.head(20))

# describe
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# Data Visualization

# histograms
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# plotting class variable
dataset['class'].value_counts().plot(kind='bar')
pyplot.show()

# distribution of independent variables
dataset['safety'].value_counts().plot(kind='bar')
pyplot.show()

# Creating a Validation Dataset
# split-out validation dataset
array = dataset.values
X = array[:, 0:6]
y = array[:, 6]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)

# Test Harness
# Spot check Algorithms
models = []
models.append(('LR', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# Make Predictions Using SVM (most precise)
# make predictions on the validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# saving the model
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

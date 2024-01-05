import pickle

# Load Iris Dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split to Train and Test Datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Transform Features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Build and Train Model
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)

# Evaluation of the Model
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# Save the model to a file
with open('iris_nn_model.pkl', 'wb') as f:
    pickle.dump(mlp, f)
    print("Pickling completed")

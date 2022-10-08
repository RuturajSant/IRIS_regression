import pandas as pd
import numpy as np

# from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# Plotting Packages
# ML Packages

from sklearn import model_selection
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/iris.csv")

# Split-out validation dataset
array = df.values
X = array[:,0:4]
Y = array[:,4]

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()

logit.fit(X_train,Y_train)

# logit.predict(X_validation)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)

svm = SVC()
svm.fit(X_train, Y_train) 



from flask import Flask,render_template,url_for,request
from flask_material import Material

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv("data/iris.csv")
    return render_template("preview.html",df_view = df)

@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		petal_length = request.form['petal_length']
		sepal_length = request.form['sepal_length']
		petal_width = request.form['petal_width']
		sepal_width = request.form['sepal_width']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [sepal_length,sepal_width,petal_length,petal_width]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

        # Reloading the Model
		if model_choice == 'logitmodel':
			# logit_model = pickle.load('data/logit_model_iris.pkl')
            
			result_prediction = logit.predict(ex1)
		elif model_choice == 'knnmodel':
			# knn_model = pickle.load('data/knn_model_iris.pkl')
			result_prediction = knn.predict(ex1)
		elif model_choice == 'dtree':
			result_prediction = dtree.predict(ex1)
		elif model_choice == 'svmmodel':
			result_prediction = svm.predict(ex1)

	return render_template('index.html', petal_width=petal_width,
		sepal_width=sepal_width,
		sepal_length=sepal_length,
		petal_length=petal_length,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)


if __name__ == '__main__':
	app.run(debug=True)
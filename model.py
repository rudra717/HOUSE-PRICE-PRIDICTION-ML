import numpy as np
import pandas as pd

data7=pd.read_csv('after_prepocessing.csv')
data7=data7.drop('Unnamed: 0',axis='columns')

X = data7.drop('price', axis='columns')

y = data7['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1, max_iter=1)
model.fit(X_train, y_train)

ypred = model.predict(X_test)

from flask import Flask,request, render_template
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict_price(data=X):
    location=request.form['Location']
    sqft=request.form['sqft']
    bath=request.form['bath']
    bhk=request.form['bhk']
    loc_index = np.where(data.columns==location)[0][0]
    x = np.zeros(len(data.columns))
    x[0] = int(bath)
    x[1] = int(bhk)
    x[2] = int(sqft)
    if loc_index >= 0:
        x[loc_index] = 1
    prediction=model.predict([x])[0]
    return render_template("index.html", pred='Predicted price is {0:.2f}'.format(prediction*100000))

if __name__ == '__main__':
    app.run()

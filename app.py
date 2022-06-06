from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from flask import Flask
from flask import request
class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)
bazadanych = load_iris()
df = pd.DataFrame(data=np.c_[bazadanych["data"], bazadanych["target"]], columns=bazadanych["feature_names"] + ["target"])
df = df.drop(["sepal width (cm)", "petal width (cm)"], axis=1)
df = df[df["target"] != 2]
df.loc[df["target"] == 0, "target"] = -1
print(df)
nn = Perceptron()
nn.fit(df.iloc[:, :2].values, df.target.values)
app = Flask(__name__)
@app.route('/api/predict/', methods=['GET'])
def home():
    sl = request.args.get("sl", 50)
    pl = request.args.get("pl", 6)
    res = nn.predict([float(sl), float(pl)]).tolist()
    res = {-1: 'setosa', 1: 'versicolor'}[res]
    return f"{res}"
app.run(port='5013')




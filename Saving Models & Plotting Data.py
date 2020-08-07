import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Since our data is seperated by semicolons we need to do sep=";"
data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]        # attributes

predict = "G3"      # lable

# trimmed our data set down we need to separate it into 4 arrays
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# # TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
# best = 0
# for _ in range(30):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
#
#     # defining the model which we will be using.
#     linear = linear_model.LinearRegression()
#
#     # train and score our model using the arrays we created
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)      # acc stands for accuracy
#
#     # print out the accuracy.
#     print(acc)
#
#     # If the current model has a better score than one we've already trained then save it
#     if acc > best:
#         best = acc
#         with open ("studentmodel.pickel", "wb") as f:
#             pickle.dump(linear, f)

pickle_in = open("studentmodel.pickel", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# Gets a list of all predictions
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# Drawing and plotting model
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
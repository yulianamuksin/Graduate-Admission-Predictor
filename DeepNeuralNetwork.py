from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from keras.optimizers import SGD, RMSprop

# 1. load data
def get_data():
    # load data from csv
    dataset = loadtxt('Admission_Predict_Ver1.1.csv', delimiter=',', skiprows=1)
    # 7 input features (1-7)
    x = dataset[ : , 1:8]
    # 1 output feature (8)
    y = dataset[ : , 8]
    # preprocess data
    x = StandardScaler().fit_transform(x)
    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = get_data()

# 2. get keras model
def get_model(x_train, y_train):   
    # define keras model 
    model = Sequential()
    model.add(Dense(units=64, input_dim=7, activation='relu')) # 1st hidden layer (12 nodes, expects 7 variables, use sigmoid as activation function)
    model.add(Dense(units=64, activation='relu')) # second hidden layer (12 nodes)
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1)) # output layer (one node)

    sgd=SGD(lr=0.1)
    # compile the keras model
    model.compile(loss='mse', optimizer=sgd, metrics=['mae'])

    # fit keras model
    model.fit(x_train, y_train, epochs=500, batch_size=200, verbose=0)

    return model

model = get_model(x_train, y_train)

# predict
y_pred = model.predict(x_test)

# evaluate keras model
mse_value, mae_value = model.evaluate(x_test,y_test, verbose=0)
print(mse_value)

r2_score_value = r2_score(y_test, y_pred)
print(r2_score_value)





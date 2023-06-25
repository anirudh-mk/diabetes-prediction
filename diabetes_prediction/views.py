from django.shortcuts import render

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


def home(request):
    return render(request, 'home.html')


def predict(request):
    pregnancies = float(request.GET['pregnancies'])
    glucose = float(request.GET['glucose'])
    blood_pressure = float(request.GET['bloodPressure'])
    skin_thickness = float(request.GET['skinThickness'])
    insulin = float(request.GET['insulin'])
    bmi = float(request.GET['bmi'])
    dpf = float(request.GET['dpf'])
    name = request.GET['name']
    age = float(request.GET['age'])

    data = pd.read_csv(r'C:\Users\pc\Desktop\Ajmal\diabetes.csv')

    data.head()

    data.describe()

    data['Outcome'].value_counts()

    data.groupby('Outcome').mean()

    x = data.drop('Outcome', axis=1)
    y = data['Outcome']
    # print(x)

    scaler = StandardScaler()

    scaler.fit(x)

    sta_data = scaler.transform(x)

    # print(sta_data)

    x = sta_data
    y = data['Outcome']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
    # print(x.shape, x_train.shape, x_test.shape)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(x_train, y_train)

    x_train_prediction = classifier.predict(x_train)
    training_data_accuracy = accuracy_score(x_train_prediction, y_train)
    # print('accuracy score of training data:', training_data_accuracy)

    x_test_prediction = classifier.predict(x_test)
    test_data_accuracy = accuracy_score(x_test_prediction, y_test)

    # print('accuracy score of training data:', test_data_accuracy)

    # input_data = (1, 85, 66, 29, 0, 26.6, 0.351, 31)
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    std_data = scaler.transform(input_data_reshaped)
    # print(std_data)

    prediction = classifier.predict(std_data)
    # print(prediction)

    if prediction[0] == 0:
        result = f'{name}, you are diabetes free'
    else:
        result = f'{name}, you are diagnose with diabetes'

    return render(request, 'home.html', {"result": result})

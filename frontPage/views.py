from django.shortcuts import render
from django.http import HttpResponse

from sklearn.externals import joblib
import numpy as np
import pandas as pd
reloadmodel = joblib.load('./model_pickle')
df=pd.read_csv("Testing.csv")
X = df.iloc[:,:-1]
symptoms_dict = {}
for index, symptom in enumerate(X):
    symptoms_dict[symptom] = index

input_vector = np.zeros(132)

# Create your views here.

def index(request):
    context = {'a':'hello'}
    return render(request, "index.html", context)

def predictd(request):
    if request.method =="POST":
        print(request.POST.dict())
        temp = []
        temp.append(request.POST.get('Symptom1'))
        temp.append(request.POST.get('Symptom2'))
        temp.append(request.POST.get('Symptom3'))
        temp.append(request.POST.get('Symptom4'))
        temp.append(request.POST.get('Symptom5'))
        for i in temp:
            if i in symptoms_dict:
                input_vector[symptoms_dict[i]] = 1
        prediction = reloadmodel.predict([input_vector])
        context = {'prediction': prediction}

        print(prediction)
    return render(request, "index.html", context)


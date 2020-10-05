import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
import math
import helper as hp

	

def minmaxnorm(matrix):
    for column in matrix: 
        minimum = (matrix[column]).min()
        maximum = (matrix[column]).max()
        matrix[column] = pd.DataFrame(((matrix[column])-minimum)/(maximum-minimum))
    return matrix
def zscorenorm(matrix):
    for column in matrix:
        if column!="zvalue":
            std = (matrix[column]).std()
            mean = (matrix[column]).mean()
            matrix[column] = pd.DataFrame(((matrix[column])-mean)/(std))
    return matrix

def gettype1data(birds):
	data = birds[len(birds)-1]
	del birds[len(birds)-1]
	y_long_test=[]
	y_lat_test=[]
	X_test=[]
	y_long_test.append(data["Y1"].as_matrix())
	y_lat_test.append(data["Y2"].as_matrix())	
	y_long_test = data["Y1"].as_matrix()
	y_lat_test = data["Y2"].as_matrix()
	del data["Y1"]
	del data["Y2"]
	X_test.append(data.as_matrix())
	X_test = data.as_matrix()
	y_long_train=[]
	y_lat_train=[]
	X_train=[]
	for bird in birds:
		data=bird
		y_long_train.append(data["Y1"].as_matrix())
		y_lat_train.append(data["Y2"].as_matrix()) 
		del data["Y1"]
		del data["Y2"]
		X_train.append(data.as_matrix())
	
	return X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test

def gettype2data(birds):
	X_train=[]
	y_long_train=[]
	y_lat_train=[]
	X_test=[]
	y_long_test=[]
	y_lat_test=[]
	for bird in birds:
		tn=len(bird)
		n=np.floor(tn*0.8).astype(int)
		
		y1=bird["Y1"]
		y2=bird["Y2"]
		del bird["Y1"]
		del bird["Y2"]
		bird=bird.as_matrix()
		y1=y1.as_matrix()
		y2=y2.as_matrix()
		X_train.append(bird[:n,:])
		X_test.append(bird[n:tn,:])
		y_long_train.append(y1[:n])
		y_lat_train.append(y2[:n])
		y_long_test.append(y1[n:tn])
		y_lat_test.append(y2[n:tn])

	return X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test

def distance(lon1,lat1,lon2,lat2):
	R = 6371000
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	delphi = phi2-phi1
	dellambda = math.radians(lon2)-math.radians(lon1)
	a=math.sin(delphi/2.0)*math.sin(delphi/2.0)+math.cos(phi1)*math.cos(phi2)*math.sin(dellambda/2.0)*math.sin(dellambda/2.0)
	c=2.0*math.atan2(math.sqrt(a),math.sqrt(1-a))
	d = R * c
	return d	

path="features/regression/"
clusters=[0,1,2]
plot=False
data=pd.DataFrame()
birds=[]
birds2=[]
clusterno=2
for cluster in clusters:
        path2=path+"/"+str(cluster)
        if(cluster==clusterno):
	        for file in os.listdir(path2):
        	        if file.endswith(".csv"):
							temp=pd.read_csv(path2+"/"+file)
			birds.append(temp)
			birds2.append(temp.copy())

models = ["LinearRegression()"]
print("Tipo1 de resultados")
additional="tip1"

X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test = gettype1data(birds)

print("Entrenamiento")
trainresults = hp.traintype2(X_train,y_long_train,y_lat_train,models[0])
print(trainresults)

print("Validaciones)
valresults = hp.crossvalidatetype1(X_train,y_long_train,y_lat_train,models[0])
print(valresults)

print("Pruebas")
testresults = hp.evaluatetype1(X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test,models[0],"Comparacion de la ruta del tipo de aves "+str(clusterno+1),plot)
print(testresults)


print("Tipo2 de resultados")

X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test = gettype2data(birds2)

additional="tipo2"

print("Entrenamiento")
trainresults = hp.traintype2(X_train,y_long_train,y_lat_train,models[0])
print(trainresults)

print("Validación")
validresults = hp.crossvalidatetype2(X_train,y_long_train,y_lat_train,models[0],"Validación cruzada entre los 2 tipos de grupos de aves "+str(clusterno+1))
print(validresults)

print("Prueba")
testresults = hp.evaluatetype2(X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test,models[0],"Comparación de la ruta del segundo tipo de aves"+str(clusterno+1),plot)
print(testresults)


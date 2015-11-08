import warnings
import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab

# load houses training data directly into numpy array. skip id column and name row
houses = np.loadtxt("housesRegr.csv", delimiter=";", usecols=(1,2,3,4), skiprows = 1)
alpha=0.0000001
costs=[]

#import digits pixel values into digits
digits = np.loadtxt("digits123.csv", delimiter=",")

# basic hypothesis function, takes training data and theta vector, returns predicted values by doing dot product
def linhypo(TrainingData, theta):
	return np.dot(TrainingData, theta)


# get the hypotheses get error by subtracting actual targets from hypotheses get the gradient
# and update theta's with the update rule, T
def linupdate(TrainingData, y, theta, alpha):

	hypotheses = linhypo(TrainingData, theta)
	errors = np.subtract(hypotheses, y)
	m = len(errors)
	alphaM = np.float64(alpha)/np.float64(m)
	cost =  np.transpose(TrainingData).dot(errors)
	gradient = alphaM * cost
	return np.subtract(theta, gradient) 


#########################################################################
# Takes the trainingdata with an inserted column of ones linear cost
# does the cost function itterative and returns the cost as a float
def gradientcost(TrainingData, y, theta, alpha):
	hypotheses = linhypo(TrainingData, theta)
	errors = np.subtract(hypotheses, np.transpose(y))
	m = len(errors)
	J = (np.float64(alpha) /  np.float64(m)) * np.float64(sum(errors))
	return J
###########################################################################


def linclosed(TrainingData, y):
	XtX = np.multiply(np.transpose(TrainingData), TrainingData)
	theta = np.multiply(np.inv(XtX), np.multiply(np.transpose(TrainingData), y))
	return theta

# Returns the updated theta vector. Inputs are training data, targets, learningrate and number of 
# iterations.
def lingrad(TrainingData, y, alpha, iterations):
	# theta is a vector of features + 1 length. we already inserted a row of X0 when creating
	# the training data. So we can just create an array of ones with the length of a feature row
	theta = np.ones(len(TrainingData[0,:]))

	# iterate for given amount of iterations
	for i in range(iterations):
		thetaNew = linupdate(TrainingData, y, theta, alpha)
		theta = thetaNew
	return theta

# arguments are the training data matrix and a vector with thetas returns vector with log hypothesis
def loghypo(TrainingData, theta):
	# get the linear hypotheses
	hypotheses = linhypo(TrainingData, theta)
	# sigmoid function on array
	one = np.float64(1)
	loghypos = one / (one + np.exp(-hypotheses))
	return loghypos 

def logupdate(TrainingData, y, theta, alpha):
	# get hypothesis from logistic 
	hypotheses = loghypo(TrainingData, theta)
	error = np.subtract(hypotheses, y)
	return theta - (alpha * np.dot(np.transpose(TrainingData), error))
	
# Takes training data matrix and a result vector y along with a theta vector outputs predicted cost
# create a vector of hypotheses
# final step in cost function to mean the sum of predictions
def logcost(TrainingData, y, theta):

	hypotheses = loghypo(TrainingData, theta)	
	costs = np.multiply(y, np.log(hypotheses)) - (np.subtract(1, y) * np.log(np.float(1)-hypotheses))	
	m = len(costs)

	J =	(-np.float64(1)/np.float64(m)) * sum(costs)
	return J

# Gradient descent for logistic regression
# takes training data matrix, and result vector along with theta vector and learningrate and iterations
# returns theta vector
# run number of iterations with new theta vector each time
def loggrad(TrainingData, y, alpha, iterations):
	theta = np.ones(len(TrainingData[0,:]))
	for i in range(iterations):
		thetaNew = logupdate(TrainingData, y, theta, alpha)
		theta = thetaNew
	return theta 	

def multiple(data, iterations, alpha):
	# for houses split data into features and prices and insert column of X0 (ones)
	# last column are prices put in y divided by 1000 for better readability
	y=data[:, 3]/1000

	# first three colums are feutures, add column of ones and put in new array
	# np.insert(<array> , 0, 1, axis=1) 0: insert as first col or row, 1: isert all ones 
	# and axis=1 will insert it as a column.. axis 0 would insert a row.
	TrainingData = np.insert(data[:,[0,1,2]], 0, 1, axis=1)
	
	# call linear gradient descennt with trainingdata prices theata
	theta = lingrad(TrainingData, y, alpha, iterations)

	return theta, TrainingData, y

def multiplesquaredinput(data, iterations, alpha):
	# for houses split data into features and prices and insert column of X0 (ones)
	# last column are prices put in y divided by 1000 for better better readability
	y=data[:, 3]/1000

	# first three colums are feutures, add column of ones and put in new array
	# np.insert(<array> , 0, 1, axis=1) 0: insert as first col or row, 1: isert all ones 
	# and axis=1 will insert it as a column.. axis 0 would insert a row.
	TrainingDataRoot = np.insert(data[:,[0,1,2]], 0, 1, axis=1)
	TrainingData = np.power(TrainingDataRoot, 2)
	
	# call linear gradient descennt with trainingdata prices theata
	theta = lingrad(TrainingData, y, alpha, iterations)

	return theta, TrainingData, y

def logreg(iterations, data):
	n = len(data[0, :])
  dataNoY = np.delete(data, -1, axis=1)
	y = data[:,-1]
	classes = set(y)

	TrainingData = np.insert(dataNoY, 0, 1, axis=1)
	# one versus all classification array
	classY = []
	for myClass in classes:
		for i in y:
			if y[i] is not myClass:
				classY[i] = 0
			else :
				classY[i] = 1
		theta = loggrad(TrainingData, classY, alpha, iterations)
		print(classY)
	print(TrainingData)




### houses vector linear regression
thetas, TrainingData, y = multiple(houses, 100, alpha)

#logthetas, logData, logY = logreg(iterations)
logreg(100, digits)

#plot for  size
plt.scatter(TrainingData[:,3], y)
XRange = np.float64(10000)
X1 = range(XRange)
Y1 = thetas[0] + thetas[3]*X1

plt.xlabel('Size of the ground in square feet')
plt.ylabel('Cost of the house in australian dollars (x1000)')
# set axis to resonable sizes
ymax=max(y) # for maximum value
plt.axis([0,XRange,0,ymax+1000])
plt.plot(X1, Y1, color='r')

# plot for number of bedrooms
#plt.scatter(TrainingData[:,2], y)
#X1 = range(20)
#X2 = thetas[0] + thetas[2]*X1
#plt.plot(X1, X2)

# plot for number of bathrooms
#plt.scatter(TrainingData[:,1], y, marker='*', color='y')
#X100 = range(100)
#X200 = thetas[0] + thetas[1]*X100
#plt.plot(X100, X200)


plt.show()
#,,,
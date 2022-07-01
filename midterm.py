from black import Line
import numpy as np
from matplotlib import pylab
from sklearn.linear_model import LinearRegression

# helper function to read data
def read_csv(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(','), 
    data = [ line.strip().split(',') for line in lines[1:] ]
    data = np.array(data).astype('float')
    return data

# read data and store in respective vars
filename_one = "data/dataset-one.csv"
filename_two = "data/dataset-two.csv"
data_one = read_csv(filename_one)
data_two = read_csv(filename_two)

# divide data_two into age groups
age_groups = [np.empty((1,3)) for i in range(6)]
for line in data_two:  
    index = int((line[0] / 10) - 2)
    age_groups[index] = np.append(arr=age_groups[index], values=line, axis=0)
twenties,thirties,fourties,fifties,sixties,seventies = age_groups

# initialize the logistic regression class
linreg = LinearRegression()

# fit the regressor to the full dataset and predict
linreg.fit(data_one[:,:-1], data_one[:,1])
predictions = linreg.predict(data_one[:,:-1])


# plot full dataset prediction
pylab.clf()
pylab.plot(data_one[:,0], data_one[:,1], ls='', marker='.', label="data")
pylab.plot(data_one[:,0], predictions, label="prediction")
pylab.title("Linear Regressor Trained to Full Dataset")
pylab.legend(loc='best')
pylab.show()

# fit the regressor to the twenties
import pdb; pdb.set_trace()
linreg.fit(twenties[:], twenties[:][2])
predictions = linreg.predict(twenties[:][0])


# plot twenties prediction
pylab.clf()
pylab.plot(twenties[:][0], twenties[:][2], ls='', marker='.', label="data")
pylab.plot(twenties[:][0], predictions, label="prediction")
pylab.title("Linear Regressor Trained to Full Dataset")
pylab.legend(loc='best')
pylab.show()
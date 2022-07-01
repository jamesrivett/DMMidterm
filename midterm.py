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
age_groups = [np.zeros((3,)*1) for i in range(6)]
for line in data_two:  
    index = int((line[0] / 10) - 2)
    age_groups[index] = np.vstack((age_groups[index], np.asarray(line)))
age_groups = np.asarray(age_groups)
twenties,thirties,fourties,fifties,sixties,seventies = age_groups[:,1:]



# initialize the logistic regression classdata_one
linreg = LinearRegression()

# fit the regressor to the full dataset and predict
linreg.fit(data_one[:,:-1], data_one[:,1])
predictions = linreg.predict(data_one[:,:-1])


# plot full dataset prediction
pylab.clf()
pylab.plot(data_one[:,0], data_one[:,1], ls='', marker='.', label="data")
pylab.plot(data_one[:,0], predictions, label="prediction")
pylab.title("Linear Regressor Trained to Full Dataset | Slope: %s" % linreg.coef_)
pylab.legend(loc='best')
pylab.show()

# select group to compare
group = seventies

# fit the regressor to the twenties and predict
linreg.fit(group[:,1:2], group[:,2])
predictions = linreg.predict(group[:,1:2])


# plot group prediction
pylab.clf()
pylab.plot(group[:,1], group[:,2], ls='', marker='.', label="data")
pylab.plot(group[:,1], predictions, label="prediction")
pylab.title("Linear Regressor Trained to Selected Group | Slope: %s" % linreg.coef_)
pylab.legend(loc='best')
pylab.show()
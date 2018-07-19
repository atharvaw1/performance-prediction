import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def result_to_html(x):

    '''Function to convert the obtained result numpy matrix into a html format required'''

    array = [['English','Kannada','Hindi','Maths','Science','Social Science','Phy and Health','Art Education'],['fa1','fa2','fa3','fa4','sa1','sa2']]
    col_index = pd.MultiIndex.from_product(array,names=['Subject',''])
    row_index = list(map(str,range(1,len(x)+1)))
    row_index  = ['8a'+x for x in row_index]
    df = pd.DataFrame(x,index=row_index,columns=col_index)
    df = df.stack()
    cols = df.columns.tolist()
    cols = cols[1:]+cols[:1]
    df = df[cols]
    df = df.round(2)
    #print(df)
    df.to_html('result.html',col_space=5)



def excel_to_np(pathname=r'SAMPLE Performance DATA SCS/class 6-A(2016-2017).xlsx',sheet='Sheet1'):

    '''Fucntion to extract and convert the given data from excel to numpy array'''
    #Read the excel file using pandas for easier data manipulation
    df = pd.read_excel(pathname,sheet_name=sheet,header=[0,1])
    #Drop the columns and rows with no data
    df.dropna(axis=0, inplace=True,how='all')
    df.dropna(axis=1, inplace=True,how='all')
    df.index = pd.Series(df.index).fillna(method='ffill')
    #Drop the unwanted columns
    del df["Student Name and Admission No"]
    del df['NO. OF DAYS ATTENDED']
    del df["NUMBER OF WORKING DAYS"]
    del df["SEMESTER PERIOD"]
    del df["% "]
    del df['Total']
    df.drop('GRADES',level=1,axis=1,inplace=True)
    df.columns=df.columns.droplevel(1)
    #Drop the Total and GrandTotal to eliminate data redundancy
    df=df[(df.EVALUATION !='Total') & (df.EVALUATION!= 'GrandTotal')]
    #Pivot the data to extend into suitable format
    df = df.pivot(index=df.index,columns='EVALUATION')
    df.replace(to_replace='--',value=0,inplace=True)
    #convert the dataframe into numpy array
    x = np.array(df.values.tolist())
    return x

def merge(*x):
    m = 10000000
    for i in x:
        if m > len(i):
            m=len(i)

    for i in x:
        i = i[:m,:]
    x = np.concatenate((x),axis=0)
    #print(x.shape)
    return x

def split(x1,x2,y,p=1,shuffle=False):
    '''Function to split the data into train and test sets along with shuffle data'''
    n=min(len(x1),len(x2))
    f= len(x1[0])
    x1=x1[:n,:]
    x2=x2[:n,:]
    y = y[:n,:]
    xy = np.concatenate((x1,x2,y),axis=1)
    if shuffle:
        np.random.shuffle(xy)

    return(xy[:int(n*p),:2*f] , xy[int(n*p):,:2*f] , xy[:int(n*p),2*f:] , xy[int(n*p):,2*f:])


def normalize(x):
    '''Function to normalize the data between values 0 ad 1 to speed up learning'''
    #Normalization is done by subtracting mean and dividing by std. deviation for each value
    mu = np.mean(x,axis=1)
    sigma = np.std(x,axis=1)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += mu[i]
            x[i,j] /= sigma[i]

    return(x)

def train_data_GD(x_train,y_train,learning_rate,epoch,l):
    '''Function to train parameters using Gradient Descent'''


    no_of_samples=x_train.shape[0]
    no_of_features=y_train.shape[1]

    #Declaring tensorflow variables
    X1 = tf.placeholder(dtype=tf.float64)
    X2 = tf.placeholder(dtype=tf.float64)
    Y = tf.placeholder(dtype=tf.float64)
    W1 = tf.Variable(np.zeros(x_train.shape[1]//2),dtype=tf.float64)
    W2 = tf.Variable(np.zeros(x_train.shape[1]//2),dtype=tf.float64)
    b = tf.Variable(np.zeros(x_train.shape[1]//2),dtype=tf.float64)

    #Regularization of W1 and W2.
    # b is not regularized
    r1 = l * tf.reduce_sum(tf.square(W1))
    r2 = l * tf.reduce_sum(tf.square(W2))
    #Prediction
    prediction = tf.add(tf.add(tf.multiply(X1,W1),tf.multiply(X2,W2)),b)
    #Computing the cost
    cost = (tf.reduce_sum(tf.square(prediction-Y))/(2*no_of_samples)) + r1 + r2
    #Gradient Descent for Optimization
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    #Initializer from tf variables
    init = tf.global_variables_initializer()
    #Cost list for ploting
    cost_list=[]

    #Start Session
    with tf.Session() as sess:
        sess.run(init)
        print("Training Data")
        for i in range(epoch):
            sess.run(optimizer,feed_dict={X1:x_train[:,:no_of_features],X2:x_train[:,no_of_features:],Y:y_train})
            cost_list.append(sess.run(cost,feed_dict={X1:x_train[:,:no_of_features],X2:x_train[:,no_of_features:],Y:y_train}))

        print('Training Completed')
        #print(sess.run(cost,feed_dict={X1:x_train[:,:no_of_features],X2:x_train[:,no_of_features:],Y:y_train}))

        #Ploting the cost vs no_of_iterations
        plt.plot(range(epoch),cost_list)
        plt.xlabel('No of Iterations')
        plt.ylabel('Cost')
        plt.show()

        return (sess.run(W1),sess.run(W2),sess.run(b))

def predict(x_test,W1,W2,b,y_test = None,test=False):
    no_of_samples = x_test.shape[0]
    no_of_features =x_test.shape[1]//2
    X1 = tf.Variable(x_test[:,:no_of_features])
    X2 = tf.Variable(x_test[:,no_of_features:])
    W1 = tf.Variable(W1)
    W2 = tf.Variable(W2)
    b = tf.Variable(b)
    prediction = tf.add(tf.add(tf.multiply(X1,W1),tf.multiply(X2,W2)),b)
    if test:
        Y = tf.Variable(y_test)
        cost = tf.reduce_sum(tf.square(prediction-Y))/(2*no_of_samples)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        if test:
            return(sess.run(cost))
        return(sess.run(prediction))


def optimize_hyperparameters(x1,x2,y,lr=False,r=False):

    '''Function to optimize the hyperparameters'''

    #Split the data into train and test sets
    x_train,x_test,y_train,y_test = split(x1,x2,y,p=0.8,shuffle=True)

    #Normalize the training data
    x_train = normalize(x_train)
    y_train = normalize(y_train)

    #Initialize the learning rates and regularization parameters
    learning_rates = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,1]
    reg_params  = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,1,1.5,2,2.5,5,10]
    cost_list = []

    #Iterate through the learning rates and append the cost of test set to a list
    if lr==True:
        parameter=learning_rates
        for i in learning_rates:
            W1,W2,b = train_data_GD(x_train,y_train,i,1000,1)
            cost = predict(x_test,W1,W2,b,y_test,test=True)
            cost_list.append(cost)

    #Iterate through the regularization parameters and append the cost of test set to a list
    if r==True:
        parameter=reg_params
        for i in reg_params:
            W1,W2,b = train_data_GD(x_train,y_train,0.001,1000,i)
            cost = predict(x_test,W1,W2,b,y_test,test=True)
            cost_list.append(cost)

    #Plot the learning rate vs cost of test set
    plt.scatter(parameter,cost_list)
    plt.xlabel('Learning Rate')
    plt.ylabel('Cost')
    plt.show()



x11 = excel_to_np(r'Data/6-A 2015-16.xlsx')
x12 = excel_to_np(r'Data/6-B 2015-16.xlsx')
x13 = excel_to_np(r'Data/6-C 2015-16.xlsx')
x1 = merge(x11,x12,x13)

x21 = excel_to_np(r'Data/7-A 2016-17.xlsx')
x22 = excel_to_np(r'Data/7-B 2016-17.xlsx')
x23 = excel_to_np(r'Data/7-C 2016-17.xlsx')
x2 = merge(x21,x22,x23)

y1 = excel_to_np(r'Data/8-A 2017-18.xlsx')
y2 = excel_to_np(r'Data/8-B 2017-18.xlsx')
y3 = excel_to_np(r'Data/8-C 2017-18.xlsx')
y = merge(y1,y2,y3)

optimize_hyperparameters(x1,x2,y,r=True)
'''
x_train,x_test,y_train,y_test = split(x1,x2,y,p=0.8,shuffle=True)
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
x_train = normalize(x_train)
y_train = normalize(y_train)

W1,W2,b = train_data_GD(x_train,y_train,0.001,1000,1)

result = predict(x_train,W1,W2,b,y_test=y_train,test=False)
result_to_html(result)
'''

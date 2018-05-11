
# coding: utf-8

# ### Classifier Example
# The task is to predict whether the GDP per capita for a country is more than the average GDP, based on the following features:<br>
# 
# -  Population density (per suqare km)<br>
# -  Population growth rate (%)<br>
# -  Urban population (%)<br>
# -  Life expectancy at birth (years)<br>
# -  Fertility rate (births per woman)<br>
# -  Infant mortality (deaths per 1000 births)<br>
# -  Enrolment in tertiary education (%)<br>
# -  Unemployment (%)<br>
# -  Estimated control of corruption (score)<br>
# -  Estimated government effectiveness (score)<br>
# -  Internet users (per 100 people)<br>
# 
# 120 examples are provided for training and 40 for testing. Each row represents one country, the first column is the label, followed by the features. The feature values have been normalised, by subtracting the mean and dividing by the standard deviation. The label is 1 if the GDP is more than average, and 0 otherwise.
# 
# 
# Based on----> https://github.com/marekrei/<br>
# Based on----> https://github.com/marekrei/<br>
# Based on----> https://github.com/marekrei/<br>
# 
# 

# In[1]:


import theano as T 
import numpy as np
import sys 
import collections
floatX = T.config.floatX


# In[2]:


class Classifier(object):
    def __init__ (self,inputSize):
        random_seed=42 #Ensure reproducibility
        hiddenSize=5
        lambdaL2=0.001
        
        rng=np.random.RandomState(random_seed)
        
        
        #Variables for the squeleton
        myInput=T.tensor.fvector('myInput')
        target=T.tensor.fscalar('target')
        learningRate=T.tensor.fscalar('learningRate')
        
        W_inputs_hidden_vals = np.asarray(rng.normal(loc=0.0,scale=0.1,size=(inputSize,hiddenSize)),dtype=floatX)
        W_inputs_hidden = T.shared(W_inputs_hidden_vals,'W_inputs_hidden')
        
        
        hidden=T.tensor.dot(myInput,W_inputs_hidden)
        hidden=T.tensor.nnet.sigmoid(hidden)
        
        W_hidden_output_vals=np.asarray(rng.normal(loc=0.0,scale=0.1,size=(hiddenSize,1)),dtype=floatX)
        W_hidden_output = T.shared(W_hidden_output_vals,'W_hidden_output')
        
        output=T.tensor.dot(hidden,W_hidden_output)
        output=T.tensor.nnet.sigmoid(output)
        
        cost=T.tensor.sqr(output-target).sum()
        cost += lambdaL2 * (T.tensor.sqr(W_hidden_output).sum() + T.tensor.sqr(W_inputs_hidden).sum())
        
        
        params = [W_inputs_hidden,W_hidden_output]
        gradients = T.tensor.grad(cost,params)
        #W_updated=W-(0.01*gradients[0])
        #updates=[(W,W_updated)]
        #These two lines above are the same as the one below. Updates is a list of tuples.
        updates = [(param,param-(learningRate * grad)) for param,grad in zip(params,gradients)]
        
        
        #####################################################################################################
        ##########################################                                           ################
        ##########################################   After the skeleton define the functions ################
        ##########################################                                           ################
        #####################################################################################################
        
        self.train = T.function([myInput,target,learningRate],[cost,output],updates=updates, allow_input_downcast=True)
        self.test = T.function([myInput,target],[cost,output],allow_input_downcast=True)
        
        


# In[3]:


def read_dataset(path):
    #Each line of the dataset is an example, which is labeled (fisrt col) and 
    #is followed by the different features
    
    dataset=[]
    with open(path,"r")as f:
        for line in f:
            #for each line, we get the label and features
            line_parts = line.strip().split()
            label = float(line_parts[0])
            vector= np.array([float(line_parts[i]) for i in range(1,len(line_parts))])
            dataset.append((label,vector))#array of tuples(label, array of features)
    return dataset


# In[ ]:


if __name__ == "__main__":
    #If we are going to call this from the console this way:
    #python classifier.py data/countries-classify-gdp-normalised.train.txt data/countries-classify-gdp-normalised.test.txt
    #path_train = sys.argv[1]
    #path_test = sys.argv[2]
    
    learningRate=0.1
    epochs = 10 #10 runs over the whole dataset
    
    data_train = read_dataset("/home/xavipor/Documentos/Cosas Deep Learning/TheanoPractice/data/train.txt")
    data_test = read_dataset("/home/xavipor/Documentos/Cosas Deep Learning/TheanoPractice/data/test.txt")
    
    numberFeatures = len(data_train[0][1])
    myClassifier = Classifier(inputSize=numberFeatures)
    
    
    for epoch in range(epochs):
        costSum=0.0
        correct=0
        for label,vector in data_train:
            cost,predicted_value = myClassifier.train(vector,label,learningRate)
            costSum += cost
            if (label == 1.0 and predicted_value >= 0.5) or (label == 0.0 and predicted_value < 0.5):
                 correct += 1
        print("Epoch: " + str(epoch) + ", Training_cost: " + str(costSum) + ", Training_accuracy: " + str(float(correct) / len(data_train)))
        
    #Testing
    
    cost_sum = 0.0 
    correct = 0
    for label,vector in data_test:
        cost, predicted_value = myClassifier.test(vector, label)
        costSum += cost
        if (label == 1.0 and predicted_value >= 0.5) or (label == 0.0 and predicted_value < 0.5):
            correct += 1
    print ( "Test_cost: " + str(costSum) + ", Test_accuracy: " + str(float(correct) / len(data_test)))


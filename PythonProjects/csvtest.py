import tflearn
import numpy as np

from tflearn.data_utils import load_csv
data, labels = load_csv('data2.csv',target_column=3, columns_to_ignore=[0], categorical_labels=True, n_classes=2)

print("---")
print(labels[0])
print(labels)
print("---")

net = tflearn.input_data(shape=[None, 5]) #An input layer, with variable input size of examples with 6 features (the [None, 6])
net = tflearn.fully_connected(net, 32) #Two hidden layers with 32 nodes
net = tflearn.fully_connected(net, 32) #net tells the computer to add it to the line above
net = tflearn.fully_connected(net, 2, activation='softmax') #An output later of 2 nodes, and a "softmax" activation (more on activations later)
net = tflearn.regression(net) #find the pattern


# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=500, batch_size=10, show_metric=True)



print("the first row of data",data[0])

print('Likely is Rich Frank to be Happy or sad:',model.predict([[42,145674,1,1,1]]))
print('Likely is Poor Bill to be Happy or sad:',model.predict([[30,3,0,0,0]]))

# right target column
# no strings
# net = tflearn.input_data(shape=[None, 5]) #right number of inputs (matches the number of columns you're actually looking at
# net = tflearn.fully_connected(net, 2, activation='softmax') right number of output nodes
# categorical_labels=True, n_classes=2) right number of n_classes
# model.fit(data, labels, n_epoch=500, batch_size=10, show_metric=True) batch sizes being smaller then the amount of data being fed in
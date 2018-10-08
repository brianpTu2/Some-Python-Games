import tflearn

# Using Neural Networks

from tflearn.datasets import titanic
titanic.download_dataset("csv/titanic_dataset.csv")

from tflearn.data_utils import load_csv
data, labels = load_csv("csv/titanic_dataset.csv", target_column=0, categorical_labels=True, n_classes=2, columns_to_ignore=[2, 7])
# Columns 3 and 8 were skipped, thus changing the indexes of the columns

# This for loop changes female to 1 and male to 0
for i in data:
    if i[1] == 'female':
        i[1] = 1
    else:
        i[1] = 0


print("The first row of data", data[0])
print("Age of the first person: ", data[0][2])

# Challenge: try to find the avg ticket price
avgFare = 0
allFare = 0
l = 0
for i in data:
    fare = data[l][5]
    allFare += float(fare)
    l += 1
    avgFare = allFare / l
print('average fare price =', avgFare)



net = tflearn.input_data(shape=[None, 6]) #An input layer, with variable input size of examples with 6 features (the [None, 6])
net = tflearn.fully_connected(net, 32) #Two hidden layers with 32 nodes
net = tflearn.fully_connected(net, 32) #net tells the computer to add it to the line above
net = tflearn.fully_connected(net, 2, activation='softmax') #An output later of 2 nodes, and a "softmax" activation (more on activations later)
net = tflearn.regression(net) #find the pattern

# Define model
model = tflearn.DNN(net)
# Start training
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

# (class, sex, age, nsiblings, nparents, ticketprice)

print('Jack', model.predict([[3, 0, 19, 0, 0, 5.0]])[0][0], "chance of death")
print('Brian', model.predict([[3, 0, 70, 0, 0, 2.0]])[0][0])

#  survived, died
#  40, 60
import tflearn
import csv

# from tflearn.data_utils import load_csv
# data, labels = load_csv("csv/FIFA_2018_Statistics2.csv", target_column=1, categorical_labels=True, n_classes=2,
#                         columns_to_ignore=[0])

with open('csv/FIFA_2018_Statistics2.csv') as file:
    reader = csv.reader(file)
    data = list(reader)

for i in data:
    if i[0]:
        if i[0] == 'W':
            i[0] = 0
        elif i[0] == 'L':
            i[0] = 1
        else:
            i[0] = 2
    if i[11]:
        if i[11] == 'no':
            i[11] = 0
        else:
            i[11] = 1


net = tflearn.input_data(shape=[None, 3]) # An input layer, with variable input size of examples with 6 features (the [None, 6])
net = tflearn.fully_connected(net, 32) # Two hidden layers with 32 nodes
net = tflearn.fully_connected(net, 32) # net tells the computer to add it to the line above
net = tflearn.fully_connected(net, 3, activation='softmax') # An output later of 2 nodes, and a "softmax" activation (more on activations later)
net = tflearn.regression(net) # find the pattern

# Define model
model = tflearn.DNN(net)
# Start training
model.fit(data, data, n_epoch=10, batch_size=11, show_metric=True)

# (Goal Scored,Ball Possession %,Attempts,On-Target,Off-Target,Free Kicks,Pass Accuracy %,Fouls Committed,Yellow Card,Red,PSO)
print('My team', model.predict([[2, 50, 15, 6, 3, 12, 80, 10, 1, 0, 0]])[0][0])



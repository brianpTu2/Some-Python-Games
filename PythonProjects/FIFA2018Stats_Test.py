import tflearn

# Import csv file
from tflearn.data_utils import load_csv
data, labels = load_csv('csv/FIFA_2018_Statistics2.csv', target_column=0, categorical_labels=True, n_classes=2)

# Changing yes and no to 1 and 0, respectively
for i in data:
    if i[10]:
        if i[10] == 'no':
            i[10] = 0
        else:
            i[10] = 1

# Creating team statistics input
name = input("Name of your team: ")
goals_scored = input("Goals you want to score: ")
possession = input("Your possession percentage: ")
attempts = input("Shot attempts: ")
on_target = input("How many shots on target: ")
off_target = input("How many shots off target: ")
free_kick = input("Number of free kicks: ")
pass_acc = input("Your passing accuracy: ")
fouls = input("Fouls committed: ")
yellow_card = input("Number of yellow cards: ")
red_card = input("Number of red cards: ")
PSO = input("End in penalty shootout? (type 1 for yes / type 0 for no): ")

# Neural Networks
net = tflearn.input_data(shape=[None, 11]) # An input layer, with variable input size of examples with 6 features (the [None, 6])
net = tflearn.fully_connected(net, 32) # Two hidden layers with 32 nodes
net = tflearn.fully_connected(net, 32) # net tells the computer to add it to the line above
net = tflearn.fully_connected(net, 2, activation='softmax') # An output later of 2 nodes, and a "softmax" activation (more on activations later)
net = tflearn.regression(net) # find the pattern

# Define model
model = tflearn.DNN(net)
# Start training
model.fit(data, labels, n_epoch=100, batch_size=16, show_metric=True)

# Print chance of winning
print(name, model.predict([[goals_scored, possession, attempts, on_target, off_target, free_kick, pass_acc, fouls, yellow_card, red_card, PSO]])[0][1])



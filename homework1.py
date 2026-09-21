# Mark Evers
# 9/20/2026
# CSCI 4931 - Deep Learning
# Homework 1

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# We will be implementing a hard-coded 2-2-1 neural network
# layer 0 has 2 neurons + 1 bias
# layer 1 has 2 neurons + 1 bias
# layer 2 has 1 neuron



#############################################################
###################### HYPERPARAMETERS ######################
#############################################################
learning_rate = 0.1
n_epochs = 1000000
convergence_threshold = 1e-5

# set seed to get same random weights every time
np.random.seed(1337)


#####################################################
##################### VARIABLES #####################
#####################################################
# our inputs
samples = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]

# our expected outputs
targets = [
    0,
    1,
    1,
    1
]

# these are the neuron outputs (aka the inputs to the next layer)
# these will be two dimensional arrays where
# - the first dimension is the layer index
# - the second dimension is the neuron index
x = [
    [0, 0, 1],      # layer 0: [neuron 0, neuron 1, bias]
    [0, 0, 1],      # layer 1: [neuron 0, neuron 1, bias]
    [0]    # layer 2: [neuron 0]
]

# the weights will be three dimensional arrays where
# - the first dimension is the layer index
# - the second dimension is the neuron index
# - the third dimension is the weight index
# note: the weights are preceeding the neuron
weights = [
    [],  # layer 0 has no weights because it is the input layer
    [
        (np.random.rand(3) * 2 - 1).tolist(),  # layer 1, neuron 0
        (np.random.rand(3) * 2 - 1).tolist()  # layer 1, neuron 1
    ],  
    [
        (np.random.rand(3) * 2 - 1).tolist()  # layer 2, neuron 0
    ]
]

# `z` will be the input to the activation function
# two dimensional like x
z = [
    [],      # layer 0 has no input to an activation function
    [0, 0],  # layer 1, neuron 0 and neuron 1
    [0]      # layer 2, neuron 0
]





#######################################################
###################### FUNCTIONS ######################
#######################################################
# our activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y_pred):
    return y_pred * (1 - y_pred)

# our loss function and its derivative
# NOTE: this is for stochastic gradient descent
def sample_loss(y_true, y_pred):
    return 1/2 * (y_true - y_pred) ** 2

def loss_derivative(y_true, y_pred):
    # Derivative with respect to y_pred: 2 * (y_pred - y_true) / N
    return y_pred - y_true

def mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)




#######################################
############ FORWARD PASS #############
#######################################
def forward_pass(sample_i):
    global samples, targets, x, weights, z

    ### LAYER 0 ###
    # the outputs of the input layer, plus the bias
    x[0][0] = samples[sample_i][0]
    x[0][1] = samples[sample_i][1]
    x[0][2] = 1
    

    ### LAYER 1 ###
    # calculate the inputs to the activation function for layer 1
    z[1][0] = (x[0][0] * weights[1][0][0]) + (x[0][1] * weights[1][0][1]) + (x[0][2] * weights[1][0][2])
    z[1][1] = (x[0][0] * weights[1][1][0]) + (x[0][1] * weights[1][1][1]) + (x[0][2] * weights[1][1][2])
    
    # calculate the outputs of the activation function for layer 1, plus the bias
    x[1][0] = sigmoid(z[1][0])
    x[1][1] = sigmoid(z[1][1])
    x[1][2] = 1  # the bias

    ### LAYER 2 ###
    # calculate the inputs to the activation function for layer 2
    z[2][0] = (x[1][0] * weights[2][0][0]) + (x[1][1] * weights[2][0][1]) + (x[1][2] * weights[2][0][2])

    # calculate the outputs of the activation function for layer 2
    x[2][0] = sigmoid(z[2][0])

    # we now have the predicted output!
    return x[2][0]





#######################################
########## BACK PROPAGATION ###########
#######################################
def back_propagation(y_pred, sample_i):
    global samples, targets, x, weights, z

    #####################################
    ### CALCULATE LOSS AND dL/dy_pred ###
    loss = sample_loss(targets[sample_i], y_pred)
    loss_gradient = loss_derivative(targets[sample_i], y_pred)


    #######################
    ### CALCULATE dσ/dz ###
    # store the gradients in a two dimensional array similar to x and z
    activation_gradient = [
        [],      # layer 0 has no activation function
        [0, 0],  # layer 1 has 2 neurons
        [0]      # layer 2 has 1 neuron
    ]

    # LAYER 2 
    activation_gradient[2][0] = sigmoid_derivative(y_pred)
    # LAYER 1
    activation_gradient[1][0] = sigmoid_derivative(x[1][0])
    activation_gradient[1][1] = sigmoid_derivative(x[1][1])
    # LAYER 0
    # no activation function here
    

    #############################
    ### CALCULATE dz/dw ###
    # three dimensional like the weights
    z_gradient = [
        [],  # layer 0 has no z because it is the input layer
        [
            np.zeros(3).tolist(),  # layer 1, neuron 0
            np.zeros(3).tolist()  # layer 1, neuron 1
        ],  
        [
            np.zeros(3).tolist()  # layer 2, neuron 0
        ]
    ]

    # LAYER 2
    z_gradient[2][0][0] = x[1][0]
    z_gradient[2][0][1] = x[1][1]
    z_gradient[2][0][2] = x[1][2]
    # LAYER 1
    z_gradient[1][0][0] = x[0][0]
    z_gradient[1][0][1] = x[0][1]
    z_gradient[1][0][2] = x[0][2]
    z_gradient[1][1][0] = x[0][0]
    z_gradient[1][1][1] = x[0][1]
    z_gradient[1][1][2] = x[0][2]
    # LAYER 0
    # no z values here because there are no weights leading into the input layer
    

    #######################
    ### CALCULATE dl/dw ###
    # same shape as the weights
    total_gradient = [
        [],  # layer 0 has no z because it is the input layer
        [
            np.zeros(3).tolist(),  # layer 1, neuron 0
            np.zeros(3).tolist()   # layer 1, neuron 1
        ],  
        [
            np.zeros(3).tolist()  # layer 2, neuron 0
        ]
    ]

    # let's make a variable, delta, to store the partial derivative at various points
    delta = [
        [],      # layer 0 has no delta because it is the input layer
        [0, 0],  # layer 1 has 2 neurons
        [0]      # layer 2 has 1 neuron
    ]

    # now do the chain rule: dl/dw = dl/dz * dz/dw
    # LAYER 2
    delta[2][0] = loss_gradient * activation_gradient[2][0]
    total_gradient[2][0][0] = delta[2][0] * z_gradient[2][0][0]
    total_gradient[2][0][1] = delta[2][0] * z_gradient[2][0][1]
    total_gradient[2][0][2] = delta[2][0] * z_gradient[2][0][2]
    # LAYER 1
    delta[1][0] = delta[2][0] * weights[2][0][0] * activation_gradient[1][0]
    total_gradient[1][0][0] = delta[1][0] * z_gradient[1][0][0]
    total_gradient[1][0][1] = delta[1][0] * z_gradient[1][0][1]
    total_gradient[1][0][2] = delta[1][0] * z_gradient[1][0][2]
    delta[1][1] = delta[2][0] * weights[2][0][1] * activation_gradient[1][1]
    total_gradient[1][1][0] = delta[1][1] * z_gradient[1][1][0]
    total_gradient[1][1][1] = delta[1][1] * z_gradient[1][1][1]
    total_gradient[1][1][2] = delta[1][1] * z_gradient[1][1][2]
    # LAYER 0
    # no weights to update here because there are no weights leading into the input layer


    ##########################
    ### UPDATE THE WEIGHTS ###
    global learning_rate

    for layer in range(1, len(weights)):
        for neuron in range(len(weights[layer])):
            for weight in range(len(weights[layer][neuron])):
                weights[layer][neuron][weight] -= learning_rate * total_gradient[layer][neuron][weight]
    

    ################
    ### COMPLETE ###
    # what to return?  loss i guess...
    return loss





##########################################
########### TRAINING FUNCTIONS ###########
##########################################
# do one epoch (loop through all the samples once)
def do_epoch():
    # we're going to save the loss from each sample
    sample_predictions = []

    # for each sample, do a forward pass and then a back propagation
    for i in range(len(samples)):
        y_pred = forward_pass(i)
        sample_predictions.append(y_pred)
        loss = back_propagation(y_pred, i)

    # return the mean square error of all the samples
    return mse(targets, sample_predictions)


# calls do_epoch n times and returns a list of the losses
def train(n_epochs):
    global convergence_threshold

    # save the loss at the end of each epoch
    loss_by_epoch = []

    # loop for the number of epochs
    with tqdm(range(n_epochs), desc="Training model", unit="epoch") as pbar:
        for i in pbar:
            total_loss = do_epoch()
            loss_by_epoch.append(total_loss)
            pbar.set_postfix(loss=total_loss)

            # if it has converged, print the results and exit
            if total_loss <= convergence_threshold:
                print("It has converged!")
                break

    # return the loss by epoch for graphing
    return loss_by_epoch


# graph the MSE loss over epochs
def plot_loss(loss_by_epoch):
    global n_epochs, learning_rate

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_by_epoch) + 1), loss_by_epoch, label="Training Loss (MSE)", color="blue")
    plt.title(f"MSE Loss vs. Epochs (n_epochs = {n_epochs}, learning_rate = {learning_rate})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


################################################################
############################# MAIN #############################
################################################################
def main():
    print("**************************************************************")
    print("********** 2-2-1 Neural Network for Logical OR Gate **********")
    print("**************************************************************")
    print(f"Starting training with {n_epochs} epochs and a learning rate of {learning_rate}...")

    # run the training function
    loss_by_epoch = train(n_epochs)

    # print the final loss
    print(f"\nFinal Loss (MSE): {loss_by_epoch[-1]:.10f}")

    # graph the loss over epochs
    plot_loss(loss_by_epoch)



if __name__ == "__main__":
    main()

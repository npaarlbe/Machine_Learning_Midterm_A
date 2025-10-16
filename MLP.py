"""From:  https://www.kaggle.com/code/prashant111/mnist-deep-neural-network-with-keras/notebook"""
#98.33 is what he got
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting library
#Tensorflow modules
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model
#MNIST Data
from keras.datasets import mnist
import random
import tensorflow as tf

seed_value = 42  # you can change this number
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# load dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# count the number of unique train labels
unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))
# count the number of unique test labels
unique, counts = np.unique(y_test, return_counts=True)
print("\nTest labels: ", dict(zip(unique, counts)),'\n')

# sample 25 mnist digits from train dataset
indexes = np.random.RandomState(seed_value).randint(0, x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]

# plot the 25 mnist digits
plt.figure(figsize=(5,5))
for i in range(len(indexes)):
    plt.subplot(5, 5, i + 1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
# plt.show()
# plt.savefig("mnist-samples.png")
# plt.close('all')

num_labels = len(np.unique(y_train))
print("Number of labels:",num_labels)

# convert to one-hot vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# image dimensions (assumed square)
image_size = x_train.shape[1]
input_size = image_size * image_size
#print("Image Size:",input_size,'\n')

# resize and normalize
x_train = np.reshape(x_train, [-1, input_size]) #28 x 28 into 784
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

# network parameters
networkParameters = [
    # Original configurations
    [128, 256, .45,.45, 20],
    [128, 256, .2,.2, 20],  # batch_size, hidden_units, dropout original
    [64, 512, .45, .45, 20],  # I added a second dropout to see if it helps
    # [128, 512, .3, .4], 
    [32, 512, .3, .4, 35], 
    [64, 512, .3, .4, 35],
    [32, 512, .3, .4, 40], 
    [32, 512, .3, .4, 50],
    [128, 512, .4, .4, 500], 




    # [16, 512, .3, .4], 
    # [8, 512, .3, .4], 
    # [128, 512, .45, .45], # I got 98.54 with this and relu
    # [128, 512, .2, .2],
      # I got 98.87 with this and relu
    # [128, 512, .55],
    # 128, 512, seems good and higher than others
    # Vary dropout (keep others constant)
    
]
highest = 0
bestParams = []
for params in networkParameters:
    batch_size = params[0]
    hidden_units = params[1]
    dropout1 = params[2]
    dropout2 = params[3]
    print("Batch Size:",batch_size," Hidden Units:",hidden_units," Dropout:",dropout1, dropout2, "Epochs: ", params[4])
# can change batch size started at 128
# batch_size = 128

# #can change hidden units started at 256
# hidden_units = 256
# #can change dropout started at 0.45
# dropout = 0.45

# model is a 3-layer MLP with ReLU and dropout after each layer

#I can change the hidden layers activation and things 
# activation types are relu, sigmoid, softmax, tanh, softplus, softsign, selu, elu, exponential
    # activations = ['relu', 'sigmoid', 'tanh', 'elu']
    activations = ['relu']
    # relu is best so far and it was the oringinally used too
    for activation in activations:
        print("Using activation:", activation)
        model = Sequential()

        model.add(Dense(512, input_dim=input_size, activation=activation))
        print("I added more layers to the model and batch normalization to help with overfitting")
        # Batch normalization helps to stabilize and accelerate training  I added it
        model.add(BatchNormalization())
        model.add(Dropout(dropout1))
        # for _ in range(3):  # Adding 4 more hidden layers    
        #Adding more layers made by me
        model.add(Dense(256, activation=activation)) # I changed the first one to 512 because that is what I was getting the best and the just halfed it every layer
        model.add(BatchNormalization())
        model.add(Dropout(dropout2))

        # model.add(Dense(512, activation=activation))
        # model.add(BatchNormalization())
        # model.add(Dropout(dropout2))



        model.add(Dense(num_labels,activation='softmax'))
        model.summary()

    #plot_model(model, to_file='mlp-mnist.png', show_shapes=True)
    #  Stochastic Gradient Descent (SGD), Adaptive Moments (Adam) and Root Mean Squared Propagation (RMSprop).
        model.compile(loss='categorical_crossentropy', 
                    optimizer='adam',
                    metrics=['accuracy']) # I added a learning rate

        # train the model
        # can change epochs to mess with it started with 20 
        # more epochs usually helps but takes longer
        epoch = params[4] # I changed the number of epochs 
        history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, verbose=0, validation_split=0.1)
        # I added validation split to see if it helps with overfitting

        train_loss, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
        loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        print("\nTrain accuracy: %.2f%%" % (100.0 * train_acc))
        print("Test accuracy: %.2f%%" % (100.0 * acc))
        if acc > highest:
            highest = acc
            bestParams = params
            activationType = activation
print("\nHighest Test accuracy: %.2f%%" % (100.0 * highest))
print("Parameters: Batch Size:",bestParams[0]," Hidden Units:",bestParams[1]," Dropout:",bestParams[2], "Dropout 2:",bestParams[3]," Activation:",activationType, " Epochs:",epoch, "Validation Split: 0.1")
print("a  How many training cases are being utilized?")
print(f"   Answer: {len(x_train)} training cases")
print("")
print("b  How many test cases are being utilized?")
print(f"   Answer: {len(x_test)} test cases")
print("")
print("c  What is the size of the images (how many pixels?)")
print(f"   Answer: {input_size} pixels (28x28 images flattened)")
print("")
print("d  How many output labels?")
print(f"   Answer: {num_labels} output labels (digits 0-9)")
print("")
print("e  Explain the numbers found in the Param # column in the model summary table")
print("   Answer: ")
print("")
print("f  What are the 'params' found in this table?")
print("   Answer: ")
print("")
print("g  What is the purpose of the dropout layers? What can be used instead?")
print("   Answer: Dropout randomly disables neurons during training to prevent overfitting and improve generalization.")
print("   Alternatives: L1/L2 regularization, Batch Normalization, Early Stopping, Data Augmentation")

print("Reflections:")
print("I got up to 98.87% accuracy on the test set with batch size 128, hidden units 512, dropout 0.45, and relu activation. With 3 hidden layers and batch normalization.  Epoch was 1000 on this one, and took a while to run, and I feel like it was over trained.")
print(" I got 98.86% accuracy with Batch Size: 128  Hidden Units: 450  Dropout: 0.2, 3 hidden layers and 400 epochs")
print(" Highest Test accuracy: 98.54% Best Parameters: Batch Size: 64  Hidden Units: 512  Dropout: 0.3, and .4 Activation: relu Epochs: 20. \n I was able to not change the epochs and improve it by around .3, by just adding 1 more layer, changing network parameters, adding validation split and adding batch normalization. \n I also tried different dropouts and found .3 and .4 worked best. \n I also tried different batch sizes and 64 worked best. \n I also tried different hidden units and 512 worked best. With 50 epochs the same parameters got up to 98.70 accuracy.")

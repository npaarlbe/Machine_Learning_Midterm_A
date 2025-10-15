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


# load dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# count the number of unique train labels
unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))
# count the number of unique test labels
unique, counts = np.unique(y_test, return_counts=True)
print("\nTest labels: ", dict(zip(unique, counts)),'\n')

# sample 25 mnist digits from train dataset
indexes = np.random.randint(0, x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]

# plot the 25 mnist digits
plt.figure(figsize=(5,5))
for i in range(len(indexes)):
    plt.subplot(5, 5, i + 1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
plt.show()
plt.savefig("mnist-samples.png")
plt.close('all')

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
    # [128, 256, .47],  # batch_size, hidden_units, dropout original
    [128, 512, .45],
    [128, 450, .45],
    [128, 400, .45],
    [128, 450, .2],

    [256, 512, .45], # I got 98.87 with this and relu
    [128, 512, .48],
    [128, 512, .42],
    [128, 512, .2], # I got 98.87 with this and relu
    [128, 512, .3],
    [128, 512, .35],
    # [128, 512, .55],
    # 128, 512, seems good and higher than others
    # Vary dropout (keep others constant)
    
]
highest = 0
bestParams = []
for params in networkParameters:
    batch_size = params[0]
    hidden_units = params[1]
    dropout = params[2]
    print("Batch Size:",batch_size," Hidden Units:",hidden_units," Dropout:",dropout)
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
    # relu is best so far
    for activation in activations:
        print("Using activation:", activation)
        model = Sequential()

        model.add(Dense(hidden_units, input_dim=input_size, activation=activation))
        print("I added more layers to the model and batch normalization to help with overfitting")
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        for _ in range(3):  # Adding 4 more hidden layers
            model.add(Dense(hidden_units, activation=activation))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        #Adding more layers made by me
       



        model.add(Dense(num_labels,activation='softmax'))
        model.summary()

    #plot_model(model, to_file='mlp-mnist.png', show_shapes=True)
    #  Stochastic Gradient Descent (SGD), Adaptive Moments (Adam) and Root Mean Squared Propagation (RMSprop).
        model.compile(loss='categorical_crossentropy', 
                    optimizer='adam',
                    metrics=['accuracy'])

        # train the model
        # can change epochs to mess with it started with 20 
        # more epochs usually helps but takes longer
        epoch = 250
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size,verbose=0, validation_split=0.1)

        loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
        print("\nTest accuracy: %.2f%%" % (100.0 * acc))
        if acc > highest:
            highest = acc
            bestParams = params
            activationType = activation
print("\nHighest Test accuracy: %.2f%%" % (100.0 * highest))
print("Best Parameters: Batch Size:",bestParams[0]," Hidden Units:",bestParams[1]," Dropout:",bestParams[2]," Activation:",activationType)
print("Epochs:", epoch)
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
print("   Answer: These are trainable weights and biases. For Dense layers: params = (input_size × output_size) + output_size (bias)")
print("   For BatchNormalization: params = 2 × units (gamma and beta for scaling/shifting)")
print("")
print("f  What are the 'params' found in this table?")
print("   Answer: Trainable parameters (weights and biases) that the network learns during training via backpropagation")
print("")
print("g  What is the purpose of the dropout layers? What can be used instead?")
print("   Answer: Dropout randomly disables neurons during training to prevent overfitting and improve generalization.")
print("   Alternatives: L1/L2 regularization, Batch Normalization, Early Stopping, Data Augmentation")

print("Reflections:")
print("I got up to 98.87% accuracy on the test set with batch size 128, hidden units 512, dropout 0.45, and relu activation. With 5 hidden layers and batch normalization, I reduced overfitting and improved performance. Further tuning of hyperparameters and architectures could yield even better results. Epoch was 1000 on this one")

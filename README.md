## Synopsis

The deep_learning repo stores deep learning projects including a project for recognizing text sentiment in IMDB
movie reviews found in the IMDB folder as well as a project for predicting student acceptance rates in the aind-2dl folder 


## Code Example

Code for text sentiment analysis includes the following: 
```
# Imports
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(42)

# TODO: Build the model architecture
model = Sequential()
model.add(Dense(8, activation='sigmoid', input_shape=(1000,)))
model.add(Dropout(.2))
model.add(Dense(4, activation='sigmoid'))
model.add(Dropout(.1))
model.add(Dense(2, activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)

# TODO: Compile the model using a loss function and an optimizer.
model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
```

Code for the student acceptance project utilizes Keras functions to build a model as illustrated below: 
```
# Imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Building the model
# Note that filling out the empty rank as "0", gave us an extra column, for "Rank 0" students.
# Thus, our input dimension is 7 instead of 6.
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(7,)))
model.add(Dropout(.3))
model.add(Dense(8, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(2, activation='softmax'))

# Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
model.summary()
```




## Motivation

These projects are a part of the Udacity Machine Learning Engineer Nanodegree curriculum. 

## Installation

Installation instructions are provided in the individual projects README files. 

## Contributors

Udacity is the primary contributer

## License

License not included

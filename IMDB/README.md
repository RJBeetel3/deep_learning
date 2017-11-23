# aind2-dl

## Synopsis

This project demonstrates how deep learning can be used to analyze text sentiment in movie reviews found on IMDB. 

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

## Motivation

This project was performed as a a part of the Udacity Machine Learning Nanodegree program. 


### Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/aind2-dl.git
		cd aind2-dl
	```

2. Obtain the necessary Python packages, and switch Keras backend to Tensorflow.  
	
	For __Mac/OSX__:
	```
		conda env create -f requirements/aind-dl-mac.yml
		source activate aind-dl
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```

	For __Windows__:
	```
		conda env create -f requirements/aind-dl-windows.yml
		activate aind-dl
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
	```

	For __Linux__:
	```
		conda env create -f requirements/aind-dl-linux.yml
		source activate aind-dl
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
	


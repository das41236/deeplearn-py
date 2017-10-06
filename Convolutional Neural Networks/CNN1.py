# Convolutional Neural Network

# Building the CNN

#importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution   input shape arguments different for tensorflow/theano
classifier.add(Convolution2D(32,(3,3), input_shape=(64,64,3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Add another convolution layer with pooling (don't need input_shape since it was in step 1)
#add more conv layers and double feature detectors (in this case 32) for even better results
classifier.add(Convolution2D(32,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# step 3 - Flattening
classifier.add(Flatten())

# step 4 - Create ANN (full connection) 
#Add another layer for better results
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN (cross entropy loss function for more than 2 outputs)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images (Check keras.io/preprocessing/image/ for documentation)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')


# get coffee( in steps_per_epoch divide image # by batch size in prev paragraph (in this case 32))
classifier.fit_generator(training_set,
                            steps_per_epoch=8000/32,
                            epochs=25,
                            validation_data=test_set,
                             validation_steps=2000/32)





















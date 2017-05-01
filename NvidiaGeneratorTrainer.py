import csv
import cv2
import numpy as np
import os
import sklearn

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        #print(line)
# print(lines[0])	
del(lines[0])
# print(lines[0])

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: 
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []

            for batch_sample in batch_samples:
                center_source_path = batch_sample[0]
                left_source_path = batch_sample[1]
                right_source_path = batch_sample[2]

                steering_center = float(batch_sample[3])
                correction = 0.2 #can tune this parameter

                center_filename = center_source_path.split('/')[-1]
                left_filename = left_source_path.split('/')[-1]
                right_filename = right_source_path.split('/')[-1]
                
                center_path = 'data/IMG/' + center_filename
                left_path = 'data/IMG/' + left_filename
                right_path = 'data/IMG/' + right_filename

                img_center = cv2.imread(center_path)
                img_left = cv2.imread(left_path)
                img_right = cv2.imread(right_path)

                left_measurement = steering_center + correction
                right_measurement = steering_center - correction

                images.append(img_center)
                measurements.append(steering_center)
                images.append(img_left)
                measurements.append(left_measurement)
                images.append(img_right)
                measurements.append(right_measurement)
    
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle({'input_1': X_train},{'output': y_train})
    

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#    print('filename:' , filename)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5 , input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('Nvidiamodel.h5')


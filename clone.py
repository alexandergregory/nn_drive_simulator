import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        #print(line)
print(lines[0])	
del(lines[0])
print(lines[0])

images = []
measurements = []
for line in lines:
    center_source_path = line[0]
    left_source_path = line[1]
    right_source_path = line[2]

    steering_center = float(line[3])

    correction = 0.2 #can tune this parameter

    center_filename = center_source_path.split('/')[-1]
    left_filename = left_source_path.split('/')[-1]
    right_filename = right_source_path.split('/')[-1]
#    print('filename:' , filename)
    center_path = 'data/IMG/' + center_filename
    left_path = 'data/IMG/' + left_filename
    right_path = 'data/IMG/' + right_filename

    img_center = cv2.imread(center_path)
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)


    centre_measurement = float(line[3])
    left_measurement = centre_measurement + correction
    right_measurement = centre_measurement - correction

    images.append(img_center)
    measurements.append(centre_measurement)
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

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5 , input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=5)

model.save('model.h5')


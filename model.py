import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
#from keras.layers.Convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

def sample_names(path):
	lines=[]

	with open(path+'driving_log.csv') as csvfile:
		reader= csv.reader(csvfile)
		next(reader, None)
		for line in reader:
			lines.append(line)
	return lines


def lenet():
	model = Sequential()
	model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Convolution2D(6,5,5,activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(6,5,5,activation='relu'))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	
	return model

def nvidia_model():
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((50,20), (0,0))))
	model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
	model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
	model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1))
	model.summary()
	return model

def change_brightness(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_bright
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def trans_image(image,angle,r):
	rows,cols,channels = image.shape

	tx = r*np.random.uniform()-r/2
	angle = angle + tx/r*2*.2
	ty = 10*np.random.uniform()-10/2
	A = np.float32([[1,0,tx],[0,1,ty]])
	image = cv2.warpAffine(image,A,(cols,rows))
    
	return image,angle

def flip_image(image, angle):
	image= cv2.flip(image,1)
	angle= angle*-1.0

	return image,angle


def preprocess_image(image_name,angle,train):
	image = cv2.imread(image_name)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	if train:#change image if train data
		opc= np.random.randint(4)

		if opc==1:
			image= change_brightness(image)
		elif opc==2:
			image,angle= trans_image(image,angle,150)
		elif opc==3:
			image,angle= flip_image(image,angle)

	return image,angle




def generator(path, samples, batch_size=32, train=True):
	num_samples = len(samples)
	correction=[0.25, 0, -0.25]

	while 1: # Loop forever so the generator never terminates
		#shuffle(samples)
		images = []
		angles = []

		for i in range(batch_size):
			ind_line= np.random.randint(len(samples))
			line= samples[ind_line]

			ind_camera= np.random.randint(3) #select left,center,right camera image randomnly
			image_name= path + 'IMG/'+ line[ind_camera].split('/')[-1]
			angle = float(line[3]) + correction[ind_camera]

			image,angle= preprocess_image(image_name,angle,train)

			images.append(image)
			angles.append(angle)

		# trim image to only see section with road
		X_train = np.array(images)
		y_train = np.array(angles)
		#yield sklearn.utils.shuffle(X_train, y_train)
		yield X_train, y_train


path= './data/'
print('Load data...')
samples= sample_names(path)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


train_generator = generator(path,train_samples, batch_size=32)
validation_generator = generator(path,validation_samples, batch_size=32, train=False)

model= nvidia_model()

print('train...')
model.compile(loss='mse', optimizer='adam')

history = model.fit_generator(train_generator, samples_per_epoch= 20000, validation_data=validation_generator, \
nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model8.h5')

print('finish...')

print(history.history.keys())
print('Loss')
print(history.history['loss'])
print('Validation Loss')
print(history.history['val_loss'])


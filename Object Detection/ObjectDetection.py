from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
import geocoder
import requests

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# Convolution - input image, applying feature detectors => feature map
# 3D Array because colored images
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# Feature Map - Take Max -> Pooled Feature Map, reduced size, reduce complexity
# without losing performance, don't lose spatial structure
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding second convolution layer
# don't need to include input_shape since we're done it
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# Pooled Feature Maps apply flattening maps to a huge vector 
# for a future ANN that is fully-conntected
# Why don't we lose spatial structure by flattening?
# We don't because the high numbers from convolution feature from the feature detector
# Max Pooling keeps them these high numbers, and flattening keeps these high numbers
# Why didn't we take all the pixels and flatten into a huge vector?
# Only pixels of itself, but not how they're spatially structured around it
# But if we apply convolution and pooling, since feature map corresponds to each feature 
# of an image, specific image unique pixels, we keep the spatial structure of the picture.
classifier.add(Flatten())


# Step 4 - Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile - SGD, Loss Function, Performance Metric
# Logarithmic loss - binary cross entropy, more than two outcomes, categorical cross entropy
# Metrics is the accuracy metric
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# part 2 - Fitting the CNN to the images 
# Keras preprocessing images to prevent overfitting, image augmentation, 
# great accuracy on training poor results on test sets
# Need lots of images to find correlations, patterns in pixels
# Find patterns in pixels, 10000 images, 8000 training, not much exactly or use a trick
# Image augmentation will create batches and each batch will create random transformation
# leading to more diverse images and more training
# Image augmentation allows us to enrich our dataset to prevent overfitting

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

# classifier.fit_generator(training_set,
#                         samples_per_epoch=8000,
#                         nb_epoch=1,
#                         validation_data=test_set,
#                         nb_val_samples=2000)


# model_json = classifier.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# classifier.save_weights("classifier.h5")
# print("Saved model to disk")


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("classifier.h5")
print("Loaded model from disk")
# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
from agora_community_sdk import AgoraRTC
# from imageai.Detection import ObjectDetection
import os

client = AgoraRTC.create_watcher("48f74ecd63554e2b844ca47b20c84116", "chromedriver.exe")
client.join_channel("viren")
[
]
users = client.get_users() # Gets references to everyone participating in the call
print(len(users))
user1 = users[0] # Can reference users in a list
print("Hello")
print("Arnav")
print("Here")
print(user1)

# binary_image = user1.frame # Gets the latest frame from the stream as a PIL image

# with open("test.jpg") as f:
#    f.write(str(binary_image)) # Can write to file
# binary_image.save("in.png") #Replace test.png with your file name
# execution_path = os.getcwd() #Returns current working directory of the project

# detector = ObjectDetection()   #Calls the object detection function from the library ImageAI
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5")) #make sure that you have downloaded resnet50_coco_best_v2.0.1.h5 to your main folder
# detector.loadModel()

# #detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "test.png"), output_image_path=os.path.join(execution_path , "test_output.png"))
# detections, extracted_images = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "in.png"), output_image_path=os.path.join(execution_path , "out.png"), extract_detected_objects=True)
#     #The above line not only just labels the objects in an image but also it extracts those images and saves it in a new directory

# for eachObject in detections:
#     print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

#Arnav Editing Starts

from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

firecount=0
sentt=0
uncount=0
while True:
	binary_image = user1.frame

	binary_image.save("in.png") #Replace test.png with your file name
	imgg = cv2.imread("in.png")
	print("Image Here ")
	print(binary_image)
	frm = cv2.imread("in.png")
	blur = cv2.GaussianBlur(frm, (21, 21), 0)
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
 
	lower = [18, 50, 50]
	upper = [35, 255, 255]
	lower = np.array(lower, dtype="uint8")
	upper = np.array(upper, dtype="uint8")
	mask = cv2.inRange(hsv, lower, upper)
    
 
 
	output = cv2.bitwise_and(frm, hsv, mask=mask)
	no_red = cv2.countNonZero(mask)
	#print("output:", frame)
	if int(no_red) > 1000:
		print ('Fire detected')
		firecount+=1
	else:
		print("Not")
	if firecount>=2:
		firecount=0
		f = open("fire.txt", "w")
		f.write("1")
		f.close()
		if sentt==0:
			sentt=1
			print("Bhai Tune toh Aag Laga di")
			url = "https://www.fast2sms.com/dev/bulk"
			g = geocoder.ip('me')
			print(g.latlng)
			lat=g.latlng[0]
			longi=g.latlng[1]
			msg="Fire detected....Check Your App.."
			querystring = {"authorization":"2ndrfwRFhotlDvcy3P8mbKIWxGsq5j0V1gO4iTAQMLUJzYCe9ZsR0OdaCYXHPIm3kg9Lnufv5r2JTDWU","sender_id":"FSTSMS","message":msg,"language":"english","route":"p","numbers":"9834576425"}

			headers = {
			    'cache-control': "no-cache"
			}

			response = requests.request("GET", url, headers=headers, params=querystring)

			print(response.text)




	rgb = imutils.resize(imgg, width=750)
	test_image = image.load_img('in.png', target_size=(64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = loaded_model.predict(test_image)
	training_set.class_indices
	if result[0][0] == 1:
		prediction = 'No-Weapon'
	else:
		prediction = 'Weapon'
		print("Gun Niche kar")
		f=open("weapon.txt", "w")
	 #    f = open("weapon.txt", "w")
		f.write("1")
		f.close()
		if sentt==0:
			sentt=1
			url = "https://www.fast2sms.com/dev/bulk"
			g = geocoder.ip('me')
			print(g.latlng)
			lat=g.latlng[0]
			longi=g.latlng[1]
			msg="Weapon Detected....Check Your App.."
			querystring = {"authorization":"2ndrfwRFhotlDvcy3P8mbKIWxGsq5j0V1gO4iTAQMLUJzYCe9ZsR0OdaCYXHPIm3kg9Lnufv5r2JTDWU","sender_id":"FSTSMS","message":msg,"language":"english","route":"p","numbers":"9834576425"}

			headers = {
			    'cache-control': "no-cache"
			}

			response = requests.request("GET", url, headers=headers, params=querystring)

			print(response.text)

	print(prediction)
	# r = frame.shape[1] / float(rgb.shape[1])
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"
		if True in matches:
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			name = max(counts, key=counts.get)
		names.append(name)
		print("Name Here: " + str(name))
		if name=="Unknown":
			uncount+=1
		if uncount>=5 and sentt==0:
			sentt=1
			uncount=0
			f = open("intruder.txt", "w")
			f.write("1")
			f.close()
			print("Tu Chor hai saale")
			url = "https://www.fast2sms.com/dev/bulk"
			g = geocoder.ip('me')
			print(g.latlng)
			lat=g.latlng[0]
			longi=g.latlng[1]
			msg="Unknown Person detected....Check Your App.."
			querystring = {"authorization":"2ndrfwRFhotlDvcy3P8mbKIWxGsq5j0V1gO4iTAQMLUJzYCe9ZsR0OdaCYXHPIm3kg9Lnufv5r2JTDWU","sender_id":"FSTSMS","message":msg,"language":"english","route":"p","numbers":"9834576425"}

			headers = {
			    'cache-control': "no-cache"
			}

			response = requests.request("GET", url, headers=headers, params=querystring)

			print(response.text)


#Arnav Editing Ends
client.unwatch() #Ends the stream







from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models import Model, load_model
import cv2
from pickle import load
from time import time
import numpy as np
from PIL import Image, ImageFile



# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# load the photo
	# image = load_img(filename, target_size=(224, 224))
	img = array_to_img(filename)
	img = img.resize((224,224), Image.ANTIALIAS)
	# convert the image pixels to a numpy array
	image = img_to_array(img)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# # load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# # pre-define the max sequence length (from training)
max_length = 34
# # load the model
model = load_model('model_img_cap.h5')







def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (255, 255, 255)
    thickness = cv2.FILLED
    margin = 1

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 2, cv2.LINE_AA)



cap = cv2.VideoCapture(0)  
cap.set(3,640)
cap.set(4,480)

if (cap.isOpened()== False):
    print("Error opening video stream or file")
    

while(cap.isOpened()):
    ret, frame = cap.read()
    capstr = "random test"  # generateCaption(frame)
    framenc = extract_features(frame)#.reshape((1,OUTPUT_DIM))
    capstr = generate_desc(model, tokenizer, framenc, max_length)
    capstr = capstr.replace("startseq", "").replace("endseq", "")
    # draw the label into the frame
    __draw_label(frame, capstr, (50,450), (100,225,50))
    cv2.imshow('Frame',frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break     

cap.release()
cv2.destroyAllWindows()

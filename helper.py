import numpy as np
import tensorflow as tf
from tensorflow import gfile
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import multiprocessing
import sys
import os

def resize(img,shape):
	return tf.image.resize(img,shape)

def load_mnist():
	pass

def preprocess_input(path):
	with Image.open(path) as img:
		img = np.array(img, np.float32)
		img = img/255
	img = create_dataset(np.array([img]),np.array([[0]*10],np.float32),1)
	return img

def create_dataset(images, labels, batch_size):
	dataset = tf.data.Dataset.from_tensor_slices((images, labels))
	dataset = dataset.shuffle(len(images))
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(1)

	return dataset


def load_data(train_path, batch_size=1, test_path=None):
	train_images, train_labels = get_data(train_path)
	if test_path:
		test_images, test_labels = get_data(test_path)
	else:
		train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, shuffle=True, test_size=0.04) 
	
	val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, shuffle=False, test_size=0.5)
	print(val_images[1].shape)
	print(len(val_labels))
	train = create_dataset(train_images, train_labels, batch_size)
	val = create_dataset(val_images, val_labels, len(val_images))
	test = create_dataset(test_images, test_labels, batch_size)
	del train_images, train_labels, test_images, test_labels, val_images, val_labels
	return train, val, test 

def get_data(data_path):
	images, labels = [], []
	# classes = os.listdir(path)
	# cur_dir = os.getcwd()
	# os.chdir(path)
	for cls in os.listdir(data_path):
		path = os.path.join(data_path,cls)

		for img_path in os.listdir(path):
			try:
				with Image.open(os.path.join(path,img_path)) as img:
					images.append(np.array(img))
					labels.append(cls)
			except Exception as e:
				pass
	images = [ image/255 for image in images]
	encoder = LabelBinarizer()
	encoder.fit(labels)

	labels = encoder.transform(labels)

	labels = labels.astype(np.float32)
	# os.chdir(cur_dir)
	print(len(images))
	print(len(labels))
	return np.array(images,np.float32), np.array(labels)


def get_batch():
	pass

def plot():
	pass

def stack_plot():
	pass


if __name__ == '__main__':
	images, labels, _ = load_data("data/notMNIST_small")
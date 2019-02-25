from helper import *
from PIL import Image

import tensorflow as tf
import numpy as np
import os
import time

"""


"""



class CNN(object):

	def __init__(self, learning_rate, dropout, batch_size,
				 epochs, model_dir=".", learning_rate_decay=0.9,
				 train_dir="data/notMNIST_small", training=True):
		
		self.learning_rate = tf.Variable(learning_rate, dtype=tf.float32, trainable=False, name="lr")
		self.batch_size = batch_size
		self.dropout = tf.constant(dropout, dtype=tf.float32)
		self.learning_rate_decay_op = tf.assign(self.learning_rate,self.learning_rate*learning_rate_decay)
		self.epochs = epochs
		self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
		self.train_dir = train_dir
		self.no_of_classes = 10
		self.saver = None
		self.training = training

	def _model_input(self):
		# We are using tf.data variables as input
		train, val, test = load_data(self.train_dir, batch_size=self.batch_size)

		self.iterator = tf.data.Iterator.from_structure(train.output_types, train.output_shapes)
		self.images, self.labels = self.iterator.get_next()
		self.images  = tf.reshape(self.images, [-1,28,28,1])
		self.train_init = self.iterator.make_initializer(train)
		self.val_init = self.iterator.make_initializer(val)
		self.test_init = self.iterator.make_initializer(test)

		# with tf.variable_scope('inputs'):
		# 	images = tf.placeholder(dtype=tf.int32, shape=(None, 784))
		# 	labels = tf.placeholder(dtype=tf.int32, shape=(None, 10))

	def _create_model(self):
		with tf.variable_scope('model'):
			x = tf.layers.conv2d(inputs=self.images, 
								 filters=32, 
								 kernel_size=(3,3),
								 activation=tf.nn.relu,
								 name="conv1")

			x = tf.layers.batch_normalization(x)
			x = tf.layers.conv2d(inputs=x, 
								 filters=32, 
								 kernel_size=(3,3),
								 activation=tf.nn.relu,
								 name="conv2")

			x = tf.layers.batch_normalization(x)
			x = tf.layers.max_pooling2d(x, pool_size=(2,2),
										strides=2, padding="same", 
										name="pool1")

			x = tf.layers.conv2d(inputs=x, 
								 filters=64, 
								 kernel_size=(3,3),
								 activation=tf.nn.relu,
								 name="conv3")

			x = tf.layers.batch_normalization(x)
			x = tf.layers.conv2d(inputs=x, 
								 filters=64, 
								 kernel_size=(3,3),
								 activation=tf.nn.relu,
								 name="conv4")

			x = tf.layers.batch_normalization(x)
			x = tf.layers.max_pooling2d(x, pool_size=(2,2),
										strides=2, padding="same", 
										name="pool2")

			feature_dim = x.shape[1] * x.shape[2] * x.shape[3]

			#flatten
			x = tf.reshape(x, [-1, feature_dim])

			x = tf.layers.dense(inputs=x, 
								units=256,
								activation='relu', name="fc1")

			x = tf.layers.dropout(x, rate=self.dropout,
								  training=self.training, name="dropout")

			self.logits = tf.layers.dense(inputs=x, units=self.no_of_classes, name="logits")

	def _loss(self):
		with tf.variable_scope('accuracy'):
			losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
			self.loss_op = tf.reduce_mean(losses)

	def _summary(self):
		with tf.variable_scope('summary'):
			# tf.summary.image()
			tf.summary.scalar('loss', self.loss_op)
			self.summary_op = tf.summary.merge_all()

	def _accuracy(self):
		with tf.variable_scope('accuracy'):
			self.pred = tf.nn.softmax(self.logits)
			# applying argmax across axis 1
			correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	def build_graph(self):
		start = time.time()
		self._model_input()
		self._create_model()
		self._loss()
		self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op)
		self._accuracy()
		self._summary()
		print("Model Created\n Took {0} seconds".format(time.time() - start))

	def step(self,session):
		"""Run training once"""
		self.training = True
		step = 0
		total_loss = 0
		session.run([self.train_init])
		try:
			while True:
				_, loss, summaries  = session.run([self.train_op, self.loss_op, self.summary_op])
				step += 1

				total_loss += loss
		except tf.errors.OutOfRangeError:
			print(step)

		print(f"Average loss {total_loss/step}")

	def eval(self,session,init):
		""""""
		self.training = False
		session.run([init])
		step = 0
		try:
			while True:
				accuracy  = session.run([self.accuracy])
				step += 1
				print(f"Accuracy on batch {step} : {accuracy[0]*100}")
		except tf.errors.OutOfRangeError:
			pass

	def train(self):
		with tf.Session() as sess:

			self.restore(sess)
			for i in range(self.epochs):
				self.step(sess)
				# print("Eval")
				self.eval(sess, self.val_init)
				self.saver.save(sess,"checkpoint/model2/model.ckpt",global_step=self.global_step)
			self.eval(sess, self.test_init)

	def predict(self, image_path):
		img = preprocess_input(image_path)
		pred_init = self.iterator.make_initializer(img)
		with tf.Session() as sess:
			self.restore(sess)
			sess.run([pred_init])
			pred = sess.run([tf.argmax(self.pred,1)])
			print(pred[0])

	def restore(self,session):
		if not self.saver:
			self.saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state("checkpoint/model2/")

		if ckpt and ckpt.model_checkpoint_path:
			print(f"Model Restored !!! from {ckpt.model_checkpoint_path}")
			self.saver.restore(session, ckpt.model_checkpoint_path)
		else:
			session.run(tf.global_variables_initializer())


if __name__ == '__main__':
	cnn = CNN(learning_rate=0.001,dropout=0.7,batch_size=128,epochs=6)
	cnn.build_graph()
	cnn.train()
	cnn.predict("data/notMNIST_small/B/MDEtMDEtMDAudHRm.png")
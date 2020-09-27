import tensorflow as tf
from Model import Cifar
from DataReader import load_data, train_valid_split
import os

def configure():
	flags = tf.app.flags

	### YOUR CODE HERE
	flags.DEFINE_integer('resnet_version', 1, 'the version of ResNet')
	flags.DEFINE_integer('resnet_size', 18, 'n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
	flags.DEFINE_integer('batch_size', 128, 'training batch size')
	flags.DEFINE_integer('num_classes', 10, 'number of classes')
	flags.DEFINE_integer('save_interval', 10, 'save the checkpoint when epoch MOD save_interval == 0')
	flags.DEFINE_integer('first_num_filters', 16, 'number of classes')
	flags.DEFINE_float('weight_decay', 2e-4, 'weight decay rate')
	flags.DEFINE_string('modeldir', 'model_v1', 'model directory')
	### END CODE HERE
	
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

def main(_):
	sess = tf.Session()
	print('---Prepare data...')

	### YOUR CODE HERE
	data_dir = 
	### END CODE HERE

	x_train, y_train, x_test, y_test = load_data(data_dir)
	x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)

	model = Cifar(sess, configure())

	### YOUR CODE HERE
	# First step: use the train_new set and the valid set to choose hyperparameters.
	# model.train(x_train_new, y_train_new, 200)
	# model.test_or_validate(x_valid, y_valid, [160, 170, 180, 190, 200])

	# Second step: with hyperparameters determined in the first run, re-train
	# your model on the original train set.
	# model.train(x_train, y_train, ?)

	# Third step: after re-training, test your model on the test set.
	# Report testing accuracy in your hard-copy report.
	# model.test_or_validate(x_test, y_test, ?)
	### END CODE HERE

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '9'
	tf.app.run()

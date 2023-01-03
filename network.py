import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	""" Sigmoid function.
	This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
	"""
	#return 0.5 * (1 + np.tanh(0.5 * x))
	return 1 / (1 + np.exp(-x))
def der_sigmoid(y):
	""" First derivative of Sigmoid function.
	The input to this function should be the value that output from sigmoid function.
	"""
	return y * (1 - y)
def der_square_error(error,n):	
	return 2/n * error
class SimpleNet:
	def __init__(self, hidden_size_1 = 100,hidden_size_2 = 100, num_step=20000, print_interval=10):
		""" A hand-crafted implementation of simple network.

		Args:
			hidden_size:	the number of hidden neurons used in this model.
			num_step (optional):	the total number of training steps.
			print_interval (optional):	the number of steps between each reported number.
		"""
		self.num_step = num_step
		self.print_interval = print_interval
		self.learning_rate = 0.1

		# Model parameters initialization
		# Please initiate your network parameters here.
		self.hidden1_weights =	np.random.normal(loc=0.0, scale=1, size=(hidden_size_1,2))
		self.hidden2_weights =	np.random.normal(loc=0.0, scale=1, size=(hidden_size_2,hidden_size_1))
		self.hidden3_weights =	np.random.normal(loc=0.0, scale=1, size=(1,hidden_size_2))
		
		self.hidden1_output = None
		self.hidden2_output = None
		
	@staticmethod
	def plot_result(data, gt_y, pred_y):
		""" Data visualization with ground truth and predicted data comparison. There are two plots
		for them and each of them use different colors to differentiate the data with different labels.

		Args:
			data:	the input data
			gt_y:	ground truth to the data
			pred_y: predicted results to the data
		"""
		assert data.shape[0] == gt_y.shape[0]
		assert data.shape[0] == pred_y.shape[0]

		plt.figure()

		plt.subplot(1, 2, 1)
		plt.title('Ground Truth', fontsize=18)

		for idx in range(data.shape[0]):
			if gt_y[idx] == 0:
				plt.plot(data[idx][0], data[idx][1], 'ro')
			else:
				plt.plot(data[idx][0], data[idx][1], 'bo')

		plt.subplot(1, 2, 2)
		plt.title('Prediction', fontsize=18)

		for idx in range(data.shape[0]):
			if pred_y[idx] == 0:
				plt.plot(data[idx][0], data[idx][1], 'ro')
			else:
				plt.plot(data[idx][0], data[idx][1], 'bo')

		plt.show()

	def forward(self, inputs):
		""" Implementation of the forward pass.
		It should accepts the inputs and passing them through the network and return results.
		"""
		self.hidden1_output = sigmoid(self.hidden1_weights.dot(inputs.T))
		self.hidden2_output = sigmoid(self.hidden2_weights.dot(self.hidden1_output))
		output = sigmoid(self.hidden3_weights.dot(self.hidden2_output))
		return output
		
	def backward(self,inputs):
		""" Implementation of the backward pass.
		It should utilize the saved loss to compute gradients and update the network all the way to the front.
		"""
		output_layer_DL = self.error * der_sigmoid(self.output) #1x1 x 1x1 = 1x1
		theta_3 = np.dot(output_layer_DL,self.hidden2_output.T)
		hidden_2_DL = np.dot(output_layer_DL,self.hidden3_weights) #1x1 x 1x100 = 1x100
		theta_2 = np.dot(self.hidden1_output,der_sigmoid(self.hidden2_output.T)*hidden_2_DL)
		hidden_1_DL = np.dot(hidden_2_DL,self.hidden2_weights) #1x100 x 100x100 = 1x100
		theta_1 = np.dot(inputs.T,der_sigmoid(self.hidden1_output.T)*hidden_1_DL)

		self.hidden3_weights  = self.hidden3_weights  - self.learning_rate * theta_3
		self.hidden2_weights = self.hidden2_weights - self.learning_rate * theta_2.T
		self.hidden1_weights = self.hidden1_weights - self.learning_rate * theta_1.T

	def train(self, inputs, labels):
		""" The training routine that runs and update the model.

		Args:
			inputs: the training (and testing) data used in the model.
			labels: the ground truth of correspond to input data.
		"""
		
		# make sure that the amount of data and label is match
		assert inputs.shape[0] == labels.shape[0]
		n = inputs.shape[0]

		for epochs in range(self.num_step):
			for idx in range(n):
				# operation in each training step:
				#	1. forward passing
				#	2. compute loss
				#	3. propagate gradient backward to the front
				self.output = self.forward(inputs[idx:idx+1, :])
				self.error = (self.output - labels[idx:idx+1, :])
				self.backward(inputs[idx:idx+1, :])
	   

			if epochs % self.print_interval == 0:
				print('Epochs {}: '.format(epochs))
				self.test(inputs, labels)

		print('Training finished')
		self.test(inputs, labels)

	def test(self, inputs, labels):
		""" The testing routine that run forward pass and report the accuracy.

		Args:
			inputs: the testing data. One or several data samples are both okay.
				The shape is expected to be [BatchSize, 2].
			labels: the ground truth correspond to the inputs.
		"""
		n = inputs.shape[0]
		error = 0.0
		for idx in range(n):
			result = self.forward(inputs[idx:idx+1, :])
			error += abs(result - labels[idx:idx+1, :][0][0])
		error /= n
		print('accuracy: %.2f' % ((1 - error)*100) + '%')
		print('error: %.17f' % (error))
		print('')
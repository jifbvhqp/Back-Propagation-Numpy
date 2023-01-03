import numpy as np
import matplotlib.pyplot as plt
from genData import GenData
from network import SimpleNet,sigmoid,der_sigmoid,der_square_error

if __name__ == '__main__':
	#data, label = GenData.fetch_data('XOR', 100)
	data, label = GenData.fetch_data('Linear',100)
	net = SimpleNet(num_step=100)
	net.train(data, label)
	pred_result = np.round(net.forward(data))
	SimpleNet.plot_result(data, label, pred_result.T)
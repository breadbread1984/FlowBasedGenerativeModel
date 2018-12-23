#!/usr/bin/python3

import numpy as np
import tensorflow as tf;
import tensorflow_probability as tfp;
from LeakyReLU import LeakyReLU;

class Model(tfp.bijectors.Bijector):

	def __init__(self, num_layers = 6, validate_args = False, name = "model"):
		super(Model, self).__init__(forward_min_event_ndims = 1, validate_args = validate_args, name = name);
		layers = [];
		for i in range(num_layers):
			#affine(linear transform)
			L = tf.Variable(np.tril(np.random.randn(2,2)), dtype = tf.float32);
			V = tf.Variable(np.random.randn(2,2), dtype = tf.float32);
			shift = tf.Variable(np.random.randn(2), dtype = tf.float32);
			layers.append(tfp.bijectors.Affine(scale_tril = L, scale_perturb_factor = V, shift = shift));
			#leakyrelu(non-linear transform)
			alpha = tf.Variable(0.1,dtype = tf.float32);
			layers.append(LeakyReLU(alpha = alpha));
		self.flow = tfp.bijectors.Chain(list(reversed(layers)));

	def _forward(self,x):
		return self.flow.forward(x);

	def _inverse(self,y):
		return self.flow.inverse(y);

	def _inverse_log_det_jacobian(self,y):
		return self.flow.inverse_log_det_jacobian(y, event_ndims = 1);

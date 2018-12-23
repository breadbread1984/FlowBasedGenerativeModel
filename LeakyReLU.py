#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;

class LeakyReLU(tfp.bijectors.Bijector):

    def __init__(self, alpha = 0.5, validate_args = False, name = "leaky_relu"):
        super(LeakyReLU, self).__init__(forward_min_event_ndims = 1, validate_args = validate_args, name = name);
        self.alpha = alpha;
        
    def _forward(self, x):
        assert(x.shape[1] == 2);
        #      / x (x >= 0)
        # y = |
        #      \ alpha * x (x < 0)
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x);
        
    def _inverse(self, y):
        assert(y.shape[1] == 2);
        #      / y (y >= 0)
        # x = |
        #      \ y / alpha (y < 0)
        return tf.where(tf.greater_equal(y, 0), y, 1. / self.alpha * y);
        
    def _inverse_log_det_jacobian(self, y):
        assert(y.shape[1] == 2);
        #            / 1 (x >= 0)
        # dx / dy = |
        #            \ 1 / alpha (x < 0)
        I = tf.ones_like(y);
        J_inv = tf.where(tf.greater_equal(y, 0), I, 1.0 / self.alpha * I);
        # sum over diagonal elements
        return tf.reduce_sum(tf.log(tf.abs(J_inv)),axis = -1);

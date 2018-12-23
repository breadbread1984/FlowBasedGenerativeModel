#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
from Model import Model;

batch_size = 200;

def main(unused_argv):
    generator = tf.estimator.Estimator(model_fn = model_fn, model_dir = "generator_model");
    tf.logging.set_verbosity(tf.logging.DEBUG);
    logging_hook = tf.train.LoggingTensorHook(tensors = {"loss":"loss"}, every_n_iter = 100);
    generator.train(input_fn = input_fn, steps = 200000, hooks = [logging_hook]);
    eval_results = generator.evaluate(input_fn = input_fn, steps = 1);
    print(eval_results);

def input_fn():
    A = np.array([[2, .3], [-1., 4]]);
    X = np.random.multivariate_normal(mean = [0.4, 1], cov = A.T.dot(A), size = 20000);
    dataset = tf.data.Dataset.from_tensor_slices(X.astype(np.float32));
    dataset = dataset.repeat();
    dataset = dataset.shuffle(buffer_size = X.shape[0]);
    dataset = dataset.prefetch(3 * batch_size);
    dataset = dataset.batch(batch_size);
    iterator = dataset.make_one_shot_iterator();
    features = iterator.get_next();
    return features;

def model_fn(features, labels, mode):
    base_dist = tfp.distributions.MultivariateNormalDiag(
        loc = tf.zeros([2],tf.float32)
    );
    transformed_dist = tfp.distributions.TransformedDistribution(
        distribution = base_dist,
        bijector = Model(num_layers = 6),
        name = "transformed_dist"
    );
    # predict mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        samples = transformed_dist.sample(batch_size);
        return tf.estimator.EstimatorSpec(mode = mode, predictions = samples);
    # train mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = -tf.reduce_mean(transformed_dist.log_prob(features));
        loss = tf.identity(loss, name = "loss");
        #learning rate
        lr = tf.train.cosine_decay(1e-2, global_step = tf.train.get_or_create_global_step(), decay_steps = 1000);
        optimizer = tf.train.AdamOptimizer(lr);
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step());
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op);
    # eval mode
    if mode == tf.estimator.ModeKeys.EVAL:
        loss = -tf.reduce_mean(transformed_dist.log_prob(features));
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = {"loss": loss});

    raise Exception('Unknown mode of estimator!');

if __name__ == "__main__":
    tf.app.run();

#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;
from train_model import model_fn;

def main():
    generator = tf.estimator.Estimator(model_fn = model_fn, model_dir = "generator_model");
    prediction = generator.predict(lambda:0);
    for i in range(200):
        print(next(prediction));

if __name__ == "__main__":
    main();
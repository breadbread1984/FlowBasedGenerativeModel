#!/usr/bin/python3

import cv2;
import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
from train_model import model_fn;

def main():
    #sample
    generator = tf.estimator.Estimator(model_fn = model_fn, model_dir = "generator_model");
    prediction = generator.predict(lambda:0);
    samples = [];
    for i in range(2000):
        samples.append(next(prediction));
    #display
    x,y,w,h = cv2.boundingRect(np.array(samples));
    print(x,y,w,h);
    img = np.zeros([240,320,3],dtype = np.uint8);
    for sample in samples:
        pos = (
            int((sample[0] - x) * 320 / w),
            int((sample[1] - y) * 240 / h)
        );
        cv2.circle(img,pos,1,(255,255,0));
    cv2.namedWindow("distribution");
    cv2.imshow("distribution",img);
    cv2.waitKey();

if __name__ == "__main__":
    main();

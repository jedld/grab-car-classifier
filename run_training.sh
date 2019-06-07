#!/bin/sh

if [ ! -f "cars_train" ]; then
  wget -c http://imagenet.stanford.edu/internal/car196/cars_train.tgz
  tar -xzf cars_train.tgz
fi

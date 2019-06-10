#!/bin/sh

# download training data
if [ ! -d "cars_train" ]; then
  wget -c http://imagenet.stanford.edu/internal/car196/cars_train.tgz
  tar -xzf cars_train.tgz
fi

# download test set
if [ ! -d "cars_test" ]; then
  wget -c http://imagenet.stanford.edu/internal/car196/cars_test.tgz
  tar -xzf cars_test.tgz
fi

echo "resizing and converting photos"

if [ ! -d "work/train" ]; then
  mkdir -p work/train
  for PHOTO in cars_train/*.jpg
    do
        BASE=`basename $PHOTO`
        convert -resize 320x200! $PHOTO work/train/$IMAGES/$BASE
    done
fi

if [ ! -d "work/test" ]; then
  mkdir -p work/test
  for PHOTO in cars_test/*.jpg
    do
        BASE=`basename $PHOTO`
        convert -resize 320x200! $PHOTO work/test/$IMAGES/$BASE
    done
fi

echo "photos preprocessing done"

# download class data
if [ ! -d "devkit" ]; then
  wget -c https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
  tar -xzf car_devkit.tgz
fi

wget -c http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat

# parse matlab file and convert to annotations to yml
ruby unpack_annotations.rb

#!/bin/bash
set -e
source activate occam

for p_bias in 0.1; do
  python datasets/biased_mnist_generator.py \
  --config_file conf/biased_mnist_generator/full.yaml \
  --p_bias ${p_bias} \
  --suffix '_'${p_bias} \
  --generate_test_set 1
done

for p_bias in 0.99 0.95 0.9 0.75 0.5; do
  python datasets/biased_mnist_generator.py \
  --config_file conf/biased_mnist_generator/full.yaml \
  --p_bias ${p_bias} \
  --suffix '_'${p_bias} \
  --generate_test_set 0
done

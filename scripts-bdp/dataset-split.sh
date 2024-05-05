#!/usr/bin/bash

(
  head -n 1 dataset-bert/train2024.csv | tee dataset-bert/train-sample50.csv dataset-bert/test-sample50.csv > /dev/null ;
  shuf < dataset-bert/train2024.csv > dataset-bert/shuf-sample.csv ;
  sed -i "/$(head -n 1 dataset-bert/train2024.csv )/d" dataset-bert/shuf-sample.csv ;
  dataset_size="$(wc -l < dataset-bert/shuf-sample.csv)" ;
  head -n "+$((${dataset_size}*50/100))" dataset-bert/shuf-sample.csv >> dataset-bert/train-sample50.csv ;
  tail -n "-$((${dataset_size}*50/100))" dataset-bert/shuf-sample.csv >> dataset-bert/test-sample50.csv ;
)

#!/usr/bin/bash

(
  dataset_file="train-expanded.csv"
  dataset_dir="dataset-bert/"
  temp_sample="shuf-sample.csv"
  train_sample="train-sample.csv"
  test_sample="test-sample.csv"
  proportion="50/100"
  
  head -n 1 "${dataset_dir}${dataset_file}" | tee "${dataset_dir}${train_sample}" "${dataset_dir}${test_sample}" > /dev/null ;
  shuf < "${dataset_dir}${dataset_file}" > "${dataset_dir}${temp_sample}" ;
  sed -i "/$(head -n 1 "${dataset_dir}${dataset_file}")/d" "${dataset_dir}${temp_sample}" ;
  dataset_size="$(wc -l < "${dataset_dir}${temp_sample}")" ;
  head -n "+$(("${dataset_size}"*"${proportion}"))" "${dataset_dir}${temp_sample}" >> "${dataset_dir}${train_sample}" ;
  tail -n "-$(("${dataset_size}"*"${proportion}"))" "${dataset_dir}${temp_sample}" >> "${dataset_dir}${test_sample}" ;
)

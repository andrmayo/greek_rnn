#!/usr/bin/env bash

cd rnn_code/models/compressed-weights/
cp $(find . -maxdepth 1 -type f -printf '%T@ %f\n' | sort -nr | head -n 1 | cut -d' ' -f2-) ../best/
cd ../best/
gunzip $(find . -maxdepth 1 -type f -printf '%T@ %f\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
cd ../../../

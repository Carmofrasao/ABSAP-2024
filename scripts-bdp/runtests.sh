#!/usr/bin/bash

( python3 -u train-bert-base.py < /dev/null > >(tee train-stdout.txt) 2> >(tee train-stderr.txt >&2) )

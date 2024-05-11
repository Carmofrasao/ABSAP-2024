#!/usr/bin/bash

( python3 -u train-bert-base-1.py < /dev/null > >(tee train-stdout-1.txt) 2> >(tee train-stderr-1.txt >&2) )

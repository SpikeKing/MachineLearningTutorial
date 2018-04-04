#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

import argparse
import sys

parse = argparse.ArgumentParser()
parse.add_argument("--learning_rate", type=float, default=0.01, help="initial learining rate")
parse.add_argument("--max_steps", type=int, default=2000, help="max")
parse.add_argument("--hidden1", type=int, default=100, help="hidden1")
FLAGS, unparsed = parse.parse_known_args(sys.argv[1:])

print FLAGS.learning_rate
print FLAGS.max_steps
print FLAGS.hidden1
print unparsed  # []，表示未指定的参数
print sys.argv[0]  # test_of_argparse.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

for i in range(4):
    # loc = ((i % 2) * 200, (int(i/2) * 200))
    loc = ((int(i/2) * 200), (i % 2) * 200)
    print(loc)
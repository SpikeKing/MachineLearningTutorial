#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

from tensorflow.python.client import device_lib


def get_available_gpus():
    """
    查看GPU的命令：nvidia-smi
    查看被占用的情况：ps aux | grep PID
    :return: GPU个数
    """
    local_device_protos = device_lib.list_local_devices()
    print "all: %s" % [x.name for x in local_device_protos]
    print "gpu: %s" % [x.name for x in local_device_protos if x.device_type == 'GPU']


get_available_gpus()

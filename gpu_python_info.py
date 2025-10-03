#!/usr/bin/env python3
from tensorflow.python.client import device_lib
import tensorflow as tf

sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]

print(device_lib.list_local_devices())
print(cuda_version)

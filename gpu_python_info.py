#!/usr/bin/env python3
from tensorflow.python.client import device_lib
import tensorflow as tf

sys_details = tf.sysconfig.get_build_info()
cuda_version="No cuda"
if "is_cuda_build" in  sys_details and sys_details['is_cuda_build']:
   cuda_version = sys_details["cuda_version"]


print(device_lib.list_local_devices())
print("cuda version" , cuda_version)

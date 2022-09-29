#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：python 
@File ：val_main.py
@Author ：chengbin
@Date ：2022/9/20 15:43 
"""
import os
from glob import glob
from val_support import *
from time import time


def computation_time(config_inner, t1, t2):
    t_cost = t2 - t1
    t_cost = t_cost / 60
    t_cost = round(t_cost, 2)
    print('耗时%0.2f分钟' % t_cost)
    config_inner['runtime'] = str(t_cost) + ' min'


if __name__ == '__main__':  # 主程序
    t_start = time()
    image_path = r'E:\python\deep_learning\wzb\images'
    config = support(image_path)
    # 计算耗时
    t_end = time()
    computation_time(config, t_start, t_end)

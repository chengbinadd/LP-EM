#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：python 
@File ：train_main.py
@Author ：chengbin
@Date ：2022/9/20 12:30 
"""
from config_file import *
from train_support import *
from glob import glob
import os
from sklearn.model_selection import train_test_split
import yaml
from time import time


def computation_time(config_inner, t1, t2):
    t_cost = t2 - t1
    t_cost = t_cost / 60
    t_cost = round(t_cost, 2)
    print('耗时%0.2f分钟' % t_cost)
    config_inner['runtime'] = str(t_cost) + ' min'


if __name__ == '__main__':  # 主程序
    t_start = time()

    # 设置默认参数
    config = vars(parse_args())
    criterion, model, optimizer, scheduler, log = support(config)

    # 加载数据集
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=12)  # 设置训练集与验证集比例

    # 数据处理
    train_loader, val_loader = data_deal(config, train_img_ids, val_img_ids)

    # 训练
    train_process(config, train_loader, val_loader, model, criterion, optimizer, scheduler, log)

    # 计算耗时
    t_end = time()
    computation_time(config, t_start, t_end)

    # 写入训练数据
    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)
    print('END')

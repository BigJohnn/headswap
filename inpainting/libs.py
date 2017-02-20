#!/usr/bin/env python
# encoding: utf-8


# -------------------------------------------------------
# version: v0.1
# author: lirui
# license: Apache Licence
# project: 
# function:
# file: libs.py
# time: 2017/2/8 0008 8:02
# -------------------------------------------------------
import os
import os.path as osp


class PathHelper():
    @classmethod
    def join_path(cls, base_dir):
        ''''''

        def func(sub_dir):
            return osp.join(base_dir, sub_dir)

        return func

    @classmethod
    def ensure_dir_exist(cls, dir):
        if not osp.exists(dir):
            os.mkdir(dir)


class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    pass

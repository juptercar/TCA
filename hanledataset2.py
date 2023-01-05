# -*- coding: utf-8 -*-
"""
这段代码会从data数据的每个类别中抽取一定比例（可以自己定，这里是1/6）的数据重新组成一个测试集（注意不是拷贝是剪切）
"""

import os
import random
import shutil

# source_file:源路径, target_ir:目标路径
def cover_files(source_dir, target_ir):
    for file in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file)
        if os.path.isfile(source_file):
            shutil.copy(source_file, target_ir)

def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.
  Args:
    dir_name: Path string to the folder we want to create.
  """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def moveFile(file_dir, save_dir):
    ensure_dir_exists(save_dir)
    path_dir = os.listdir(file_dir)  # 取图片的原始路径
    filenumber = len(path_dir)
    rate = 0.2  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    print(picknumber)
    sample = random.sample(path_dir, picknumber)  # 随机选取picknumber数量的样本图片
    print (sample)
    for name in sample:
        shutil.move(file_dir + name, save_dir + name)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

if __name__ == '__main__':

    path = 'F:/data/mini-imagenet/'
    dirs = os.listdir(path + 'data/')
    for file in dirs:
        file_dir = path + 'data/' + file + '/'  # 源图片文件夹路径
        print(file_dir)
        save_dir = path + 'test/' + file  # 移动到新的文件夹路径
        print(save_dir)
        mkdir(save_dir)  # 创造文件夹
        save_dir = save_dir + '/'
        moveFile(file_dir, save_dir)
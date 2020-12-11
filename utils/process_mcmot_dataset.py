# encoding=utf-8

import os
import time
import shutil
import re
import cv2
import pickle
import numpy as np
from collections import defaultdict

classes = [
    'car',  # 0
    'bicycle',  # 1
    'person',  # 2
    'cyclist',  # 3
    'tricycle'  # 4
]  # 5类(不包括背景)

cls2id = {
    'car': 0,
    'bicycle': 1,
    'person': 2,
    'cyclist': 3,
    'tricycle': 4
}

id2cls = {
    0: 'car',
    1: 'bicycle',
    2: 'person',
    3: 'cyclist',
    4: 'tricycle'
}

# 视频训练数据图片的宽高是固定的
W, H = 1920, 1080


def gen_labels_for_seq(dark_txt_path, seq_label_dir, classes, one_plus=True):
    """
    """
    global seq_max_id_dict, start_id_dict, fr_cnt

    # ----- 开始一个视频seq的label生成
    # 每遇到一个待处理的视频seq, reset各类max_id为0
    for class_type in classes:
        seq_max_id_dict[class_type] = 0

    # 记录当前seq各个类别的track id集合
    id_set_dict = defaultdict(set)

    # 读取dark label(读取该视频seq的标注文件, 一行代表一帧)
    with open(dark_txt_path, 'r', encoding='utf-8') as r_h:
        # 读视频标注文件的每一行: 每一行即一帧
        for line in r_h.readlines():
            fr_cnt += 1

            line = line.split(',')
            fr_id = int(line[0])
            n_objs = int(line[1])
            # print('\nFrame {:d} in seq {}, total {:d} objects'.format(f_id + 1, seq_name, n_objs))

            # 当前帧所有的检测目标label信息
            fr_label_objs = []

            # 遍历该帧的每一个object
            for cur in range(2, len(line), 6):  # cursor
                class_type = line[cur + 5].strip()
                class_id = cls2id[class_type]  # class type => class id

                # 解析track id
                if one_plus:
                    track_id = int(line[cur]) + 1  # track_id从1开始统计
                else:
                    track_id = int(line[cur])

                # 更新该视频seq各类检测目标(背景一直为0)的max track id
                if track_id > seq_max_id_dict[class_type]:
                    seq_max_id_dict[class_type] = track_id

                # 记录当前seq各个类别的track id集合
                id_set_dict[class_type].add(track_id)

                # 根据起始track id更新在整个数据集中的实际track id
                track_id += start_id_dict[class_type]

                # 读取bbox坐标
                x1, y1 = int(line[cur + 1]), int(line[cur + 2])
                x2, y2 = int(line[cur + 3]), int(line[cur + 4])

                # 根据图像分辨率, 裁剪bbox
                x1 = x1 if x1 >= 0 else 0
                x1 = x1 if x1 < W else W - 1
                y1 = y1 if y1 >= 0 else 0
                y1 = y1 if y1 < H else H - 1
                x2 = x2 if x2 >= 0 else 0
                x2 = x2 if x2 < W else W - 1
                y2 = y2 if y2 >= 0 else 0
                y2 = y2 if y2 < H else H - 1

                # 计算bbox center和bbox width&height
                bbox_center_x = 0.5 * float(x1 + x2)
                bbox_center_y = 0.5 * float(y1 + y2)
                bbox_width = float(x2 - x1 + 1)
                bbox_height = float(y2 - y1 + 1)

                # bbox center和bbox width&height归一化到[0.0, 1.0]
                bbox_center_x /= W
                bbox_center_y /= H
                bbox_width /= W
                bbox_height /= H

                # 打印中间结果, 验证是否解析正确...
                # print(track_id, x1, y1, x2, y2, class_type)

                # 每一帧对应的label中的每一行
                obj_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    class_id,  # class id: 从0开始计算
                    track_id,  # track id: 从1开始计算
                    bbox_center_x,  # center_x
                    bbox_center_y,  # center_y
                    bbox_width,  # bbox_w
                    bbox_height)  # bbox_h
                # print(obj_str, end='')
                fr_label_objs.append(obj_str)

            # ----- 该帧解析结束, 输出该帧的label文件: 每一帧图像对应一个txt格式的label文件
            label_f_path = seq_label_dir + '/{:05d}.txt'.format(fr_id)
            with open(label_f_path, 'w', encoding='utf-8') as w_h:
                for obj in fr_label_objs:
                    w_h.write(obj)
            # print('{} written\n'.format(label_f_path))

    return id_set_dict


"""
将DarkLabel的标注格式: frame# n_obj [id, x1, y1, x2, y2, label]
转化为MCMOT的输入格式:
1. 每张图对应一个txt的label文件
2. 每行代表一个检测目标: cls_id, track_id, center_x, center_y, bbox_w, bbox_h(每个目标6列)
"""


def dark_label2mcmot_label(data_root, one_plus=True, dict_path=None, viz_root=None):
    """
    :param data_root:
    :param one_plus:
    :param dict_path:
    :param viz_root:
    :return:
    """
    if not os.path.isdir(data_root):
        print('[Err]: invalid data root')
        return

    img_root = data_root + '/JPEGImages'
    if not os.path.isdir(img_root):
        print('[Err]: invalid image root')

    # 创建标签文件根目录
    label_root = data_root + '/labels_with_ids'
    if not os.path.isdir(label_root):
        os.makedirs(label_root)
    else:
        shutil.rmtree(label_root)
        os.makedirs(label_root)

    # ---------- 参数初始化
    # 为视频seq的每个检测类别设置[起始]track id
    global start_id_dict
    start_id_dict = defaultdict(int)  # str => int
    for class_type in classes:  # 初始化
        start_id_dict[class_type] = 0

    # 记录每一个视频seq各类最大的track id
    global seq_max_id_dict
    seq_max_id_dict = defaultdict(int)

    global fr_cnt
    fr_cnt = 0

    # ----------- 开始处理
    seq_list = os.listdir(img_root)
    seqs = sorted(seq_list, key=lambda x: int(x.split('_')[-1]))

    # 遍历每一段视频seq
    for seq_name in seqs:
        seq_dir = img_root + '/' + seq_name
        print('\nProcessing seq', seq_dir)

        # 为该视频seq创建label目录
        seq_label_dir = label_root + '/' + seq_name
        if not os.path.isdir(seq_label_dir):
            os.makedirs(seq_label_dir)
        else:
            shutil.rmtree(seq_label_dir)
            os.makedirs(seq_label_dir)

        dark_txt_path = seq_dir + '/' + seq_name + '_gt.txt'
        if not os.path.isfile(dark_txt_path):
            print('[Warning]: invalid dark label file.')
            continue

        # 当前seq生成labels
        id_set_dict = gen_labels_for_seq(dark_txt_path, seq_label_dir, classes, one_plus)

        # 输出该视频seq各个检测类别的max track id(从1开始)
        for k, v in seq_max_id_dict.items():
            print('seq {}'.format(seq_name) + ' ' +
                  k + ' max track id {:d}'.format(v))

            # 输出当前seq各个类别的track id数(独一无二的id个数)
            cls_id_set = id_set_dict[k]
            print('seq {}'.format(seq_name) + ' ' +
                  k + ' track id number {:d}'.format(len(cls_id_set)))

            if len(cls_id_set) != v:
                print(cls_id_set)

        # 处理完成一个视频seq, 基于seq_max_id_dict, 更新各类别start track id
        # for k, v in start_id_dict.items():
        #     start_id_dict[k] += seq_max_id_dict[k]

        # 处理完成一个视频seq, 基于id_set_dict, 更新各类别start track id
        for k, v in start_id_dict.items():
            start_id_dict[k] += len(id_set_dict[k])

    # 输出所有视频seq各个检测类别的track id总数
    print('\n')
    for k, v in start_id_dict.items():
        print(k + ' total ' + str(v) + ' track ids')
    print('Total {} frames.'.format(fr_cnt))

    # 序列化max_id_dict到磁盘
    if not dict_path is None:
        max_id_dict = {cls2id[k]:v for k, v in start_id_dict.items()}
        with open(dict_path, 'wb') as f:
            np.savez(dict_path, max_id_dict=max_id_dict)  # set key 'max_id_dict'

    print('{:s} dumped.'.format(dict_path))


def gen_mcmot_data(img_root, out_f_path):
    """

    :param img_root:
    :return:
    """
    if not os.path.isdir(img_root):
        print('[Err]: ')
        return

    dir_names = [img_root + '/' + x for x in os.listdir(img_root) if os.path.isdir(img_root + '/' + x)]

    with open(out_f_path, 'w', encoding='utf-8') as w_h:
        for dir in tqdm(dir_names):
            for img_name in os.listdir(dir):
                if not img_name.endswith('.jpg'):
                    continue

                img_path = dir + '/' + img_name
                if not os.path.isfile(img_path):
                    print('[Warning]: invalid image file.')
                    continue

                w_h.write(img_path + '\n')


if __name__ == '__main__':
    dark_label2mcmot_label(data_root='/mnt/diskb/even/dataset/MCMOT',
                           one_plus=True,
                           dict_path='/mnt/diskb/even/dataset/MCMOT/max_id_dict.npz',
                           viz_root=None)

    gen_mcmot_data(img_root='/mnt/diskb/even/dataset/MCMOT/JPEGImages',
                   out_f_path='/mnt/diskb/even/YOLOV4/data/train_mcmot.txt')

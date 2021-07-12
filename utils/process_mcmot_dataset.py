# encoding=utf-8

import hashlib
import math
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

cls_names = [
    'car',  # 0
    'bicycle',  # 1
    'person',  # 2
    'cyclist',  # 3
    'tricycle'  # 4
]  # 5类(不包括背景)

# cls_names = [
#     'car',  # 0
#     'bicycle',  # 1
#     'person',  # 2
#     'cyclist',  # 3
#     'tricycle',  # 4
#     'car_plate'  # 5
# ]  # 6类

cls2id = {
    'car': 0,
    'bicycle': 1,
    'person': 2,
    'cyclist': 3,
    'tricycle': 4,
    'car_plate': 5
}

id2cls = {
    0: 'car',
    1: 'bicycle',
    2: 'person',
    3: 'cyclist',
    4: 'tricycle',
    5: 'car_plate'
}

# cls_names = [
#     'car',  # 0
#     # 'car_plate'  # 1
# ]  # 1 or 2类

# cls2id = {
#     'car': 0,
#     # 'car_plate': 1
# }
#
# id2cls = {
#     0: 'car',
#     # 1: 'car_plate'
# }

# 视频训练数据图片的宽高是固定的(并不是固定的)
global W, H
W, H = -1, -1


def gen_lbs_for_a_seq(dark_txt_path, seq_label_dir, cls_names, one_plus=True):
    """
    :param dark_txt_path:
    :param seq_label_dir:
    :param cls_names:
    :param one_plus:
    :return:
    """
    global seq_max_id_dict, start_id_dict, fr_cnt, W, H

    if W < 0 or H < 0:
        print('[Err]: wrong image WH.')
        return None, 0
    print('Image width&height: {}×{}'.format(W, H))

    # ----- 开始一个视频seq的label生成
    # 每遇到一个待处理的视频seq, reset各类max_id为0
    for class_name in cls_names:
        seq_max_id_dict[class_name] = 0

    # 记录当前seq各个类别的track id集合
    id_set_dict = defaultdict(set)

    # 读取dark label(读取该视频seq的标注文件, 一行代表一帧)
    lb_cnt = 0
    with open(dark_txt_path, 'r', encoding='utf-8') as r_h:
        # 读视频标注文件的每一行: 每一行即一帧
        for line_i, line in enumerate(r_h.readlines()):
            fr_cnt += 1

            if len(line.split(',')) < 6:
                continue

            # 判断该帧是否合法的标志
            is_fr_valid = False

            line = line.split(',')
            fr_id = int(line[0])
            if fr_id > line_i:  # to avoid darklabel txt file frame id start from 1
                fr_id -= 1

            n_objs = int(line[1])
            # print('\nFrame {:d} in seq {}, total {:d} objects'.format(f_id + 1, seq_name, n_objs))

            # 当前帧所有的检测目标label信息
            fr_label_objs = []

            # 遍历该帧的每一个object
            for cur in range(2, len(line), 6):  # cursor
                class_name = line[cur + 5].strip()
                if class_name not in cls_names:
                    continue  # 跳过不在指定object class中的目标

                class_id = cls2id[class_name]  # class name => class id

                # 解析track id
                if one_plus:
                    track_id = int(line[cur]) + 1  # track_id从1开始统计
                else:
                    track_id = int(line[cur])

                # 更新该视频seq各类检测目标(背景一直为0)的max track id
                if track_id > seq_max_id_dict[class_name]:
                    seq_max_id_dict[class_name] = track_id

                # 记录当前seq各个类别的track id集合
                id_set_dict[class_name].add(track_id)

                # 根据起始track id更新在整个数据集中的实际track id
                track_id += start_id_dict[class_name]

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

                # 出现坐标错误的情况是应该跳过整个Video
                # 还是仅仅跳过当前帧
                if x1 >= x2 or y1 >= y2:
                    print('{} wrong labeled in line {}.'.format(dark_txt_path, line_i))
                    return None, 0

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

                # Nan 判断
                if math.isnan(class_id) or math.isnan(track_id) \
                        or math.isnan(bbox_center_x) or math.isnan(bbox_center_y) \
                        or math.isnan(bbox_width) or math.isnan(bbox_height):
                    print('Found nan value, invalid frame, skip frame {}.'.format(fr_id))
                    break  # 跳出当前帧的循环
                else:
                    # class id, track id有效性判断
                    if class_id < 0 or class_id >= len(cls_names) or track_id < 0:
                        print('Found illegal value of class id or track id.', class_id, track_id)
                        break

                    if bbox_center_x < 0.0 or bbox_center_x > 1.0 \
                            or bbox_center_y < 0.0 or bbox_center_y > 1.0 \
                            or bbox_width < 0.0 or bbox_width > 1.0 \
                            or bbox_height < 0.0 or bbox_height > 1.0:
                        print('Found illegal value of bbox.',
                              bbox_center_x, bbox_center_y, bbox_width, bbox_height)
                        break

                    is_fr_valid = True

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

            if is_fr_valid:
                # ----- 该帧解析结束, 输出该帧的label文件: 每一帧图像对应一个txt格式的label文件
                label_f_path = seq_label_dir + '/{:05d}.txt'.format(fr_id)
                with open(label_f_path, 'w', encoding='utf-8') as w_h:
                    for obj in fr_label_objs:
                        w_h.write(obj)
                # print('{} written\n'.format(label_f_path))
            else:
                return None, 0

            lb_cnt += 1

    return id_set_dict, lb_cnt


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
    global W, H

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
    for class_type in cls_names:  # 初始化
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

        # 读取正确的image width and image height(W, H)
        img_paths = [seq_dir + '/' + x for x in os.listdir(seq_dir) if x.endswith('.jpg')]
        print('total {} frames for {}.'.format(len(img_paths), seq_name))

        # 读取视频的第0帧, 获取真实的帧宽高
        try:
            img_tmp = cv2.imread(img_paths[0])
            H, W = img_tmp.shape[:2]
        except Exception as e:
            print(e)
            print('[Err]: the first frame load failed!')
            continue

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

        # ---------- 当前seq生成labels
        id_set_dict, lb_cnt = gen_lbs_for_a_seq(dark_txt_path, seq_label_dir, cls_names, one_plus)
        if id_set_dict == None:
            print('Skip video seq {} because of wrong label.'.format(seq_name))
            continue
        # ----------
        print('{} labels generated.'.format(lb_cnt))

        if len(img_paths) != lb_cnt:
            print('[Warning]: difference of frames and labels length: {} frames, {} labels'
                  .format(len(img_paths), lb_cnt))

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

    ## 序列化max_id_dict到磁盘
    if not dict_path is None:
        max_id_dict = {cls2id[k]: v for k, v in start_id_dict.items()}
        with open(dict_path, 'wb') as f:
            np.savez(dict_path, max_id_dict=max_id_dict)  # set key 'max_id_dict'

    print('{:s} dumped.'.format(dict_path))


def check_imgs_and_labels(mcmot_root):
    if not os.path.isdir(mcmot_root):
        print('[Err]: invalid mcmot root.')
        return

    jpg_root = mcmot_root + '/JPEGImages'
    txt_root = mcmot_root + '/labels_with_ids'

    seq_names = [x for x in os.listdir(jpg_root)]
    cnt = 0
    cnt_no_label = 0
    for seq_name in seq_names:
        seq_dir = jpg_root + '/' + seq_name
        txt_dir = txt_root + '/' + seq_name
        if not (os.path.isdir(seq_dir) and os.path.isdir(txt_dir)):
            print('[Warning]: seq {} not complete.'.format(seq_name))
            continue

        for jpg_name in os.listdir(seq_dir):
            if not jpg_name.endswith('.jpg'):
                continue

            jpg_path = seq_dir + '/' + jpg_name
            txt_path = txt_dir + '/' + jpg_name.replace('.jpg', '.txt')

            if os.path.isfile(jpg_path) and os.path.isfile(txt_path):
                cnt += 1
            elif os.path.isfile(jpg_path) and (not os.path.isfile(txt_path)):
                print('Label {} do not exists.'.format(txt_path))
                cnt_no_label += 1

                # 删除相应的帧
                os.remove(jpg_path)
                print('{} removed.'.format(jpg_path))
            elif os.path.isfile(txt_path) and (not os.path.isfile(jpg_path)):
                print('Image {} do not exists.'.format(jpg_path))
                # 删除相应的标注文件
                os.remove(txt_path)
                print('{} removed.'.format(txt_path))

    print('Total {} labels do not exists.\n'.format(cnt_no_label))
    print('Reamain {} images and corresponding labels.'.format(cnt))


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


def FindFileWithSuffix(root, suffix, f_list):
    """
    递归的方式查找特定后缀文件
    """
    for f in os.listdir(root):
        f_path = os.path.join(root, f)
        if os.path.isfile(f_path) and f.endswith(suffix):
            f_list.append(f_path)
        elif os.path.isdir(f_path):
            FindFileWithSuffix(f_path, suffix, f_list)


def GenerateFileList(root, suffix, list_name, mode='name'):
    """
    生成指定后缀的文件名列表txt文件
    """
    if not os.path.isdir(root):
        print('[Err]: invalid root')
        return

    f_list = []
    FindFileWithSuffix(root, suffix, f_list)

    if len(f_list) == 0:
        print('[Warning]: empty file list')
        return

    # f_list.sort(key=lambda x: int(os.path.split(x)[1][:-4]))
    f_list.sort()
    print(f_list)
    with open(root + '/' + list_name, 'w', encoding='utf-8') as f_h:
        for i, f_path in tqdm(enumerate(f_list)):
            f_name = os.path.split(f_path)[1]
            if mode == 'name':
                f_h.write(f_name)
            elif mode == 'path':
                f_h.write(f_path)
            else:
                print('[Err]: un-recognized output mode.')
                return

            if i != len(f_list) - 1:
                f_h.write('\n')


def test_model_md5(model_path):
    """
    :param model_path:
    :return:
    """
    if not os.path.isfile(model_path):
        print('[Err]: invalid model file path.')
        return

    with open(model_path, 'rb') as fp:
        data = fp.read()
        file_md5= hashlib.md5(data).hexdigest()
        print('MD5:\n', file_md5)


if __name__ == '__main__':
    ## ----------
    DATASET = 'MCMOT'  # MCMOT or PLM
    dark_label2mcmot_label(data_root='/mnt/diskb/even/dataset/{:s}'.format(DATASET),
                           one_plus=True,
                           dict_path='/mnt/diskb/even/dataset/{:s}/max_id_dict.npz'.format(DATASET),
                           viz_root=None)

    check_imgs_and_labels(mcmot_root='/mnt/diskb/even/dataset/{:s}'.format(DATASET))

    gen_mcmot_data(img_root='/mnt/diskb/even/dataset/{:s}/JPEGImages'.format(DATASET),
                   out_f_path='/mnt/diskb/even/YOLOV4/data/train_{:s}.txt'.format(DATASET.lower()))
    ## ---------

    # GenerateFileList(root='/mnt/diskb/even/Pic_2/',
    #                  suffix='.jpg',
    #                  list_name='tmp.py.txt',
    #                  mode='path')  # name of path
    # test_model_md5(model_path='../weights/mcmot_tiny_track_last_210508.weights')

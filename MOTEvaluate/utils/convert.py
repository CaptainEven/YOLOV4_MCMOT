# encoding=utf-8

import os
import cv2

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

# 图片数据的宽高
W, H = 1920, 1080


def convert_darklabel_2_mot16(darklabel_txt_path,
                              interval=1,
                              fps=12,
                              out_mot16_path=None):
    """
    将darklabel标注格式frame # n [id, x1, y1, x2, y2, label]
    转换成mot16格式
    """
    if not os.path.isfile(darklabel_txt_path):
        print('[Err]: invalid input file path.')
        return

    if out_mot16_path is None:
        out_fps = fps // int(interval)
        print('[Note]: out_mot16_path not defined, using default.')
        dir_name, file_name = os.path.split(darklabel_txt_path)
        out_mot16_path = dir_name + '/' + \
                         file_name.split('.')[0] + \
                         '_mot16_fps{:d}.txt'.format(out_fps)

    with open(darklabel_txt_path, 'r', encoding='utf-8') as r_h, \
            open(out_mot16_path, 'w', encoding='utf-8') as w_h:
        lines = r_h.readlines()

        # 遍历每一帧
        fr_idx = 0
        for fr_i, line in enumerate(lines):
            if fr_i % interval != 0:
                continue

            line = line.strip().split(',')
            fr_id = int(line[0])
            n_objs = int(line[1])

            # 遍历当前帧的每一个object
            for cur in range(2, len(line), 6):
                class_type = line[cur + 5].strip()
                class_id = cls2id[class_type]  # class type => class id

                # 读取track id
                track_id = int(line[cur]) + 1  # track_id从1开始统计

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

                left, top = x1, y1
                width, height = x2 - x1, y2 - y1

                # 写入该obj的数据
                if interval == 1:
                    write_line_str = str(fr_id + 1) + ',' \
                                     + str(track_id) + ',' \
                                     + str(left) + ',' \
                                     + str(top) + ',' \
                                     + str(width) + ',' \
                                     + str(height) + ',' \
                                     + '1,' + str(class_id) + ',' + '1'
                else:
                    write_line_str = str(fr_idx + 1) + ',' \
                                     + str(track_id) + ',' \
                                     + str(left) + ',' \
                                     + str(top) + ',' \
                                     + str(width) + ',' \
                                     + str(height) + ',' \
                                     + '1,' + str(class_id) + ',' + '1'
                print(write_line_str)
                w_h.write(write_line_str + '\n')

            fr_idx += 1
            print('Frame(start from 0) ', str(fr_id), 'sampled.')


def convert_seqs(seq_root, interval=1):
    """
    """
    if not os.path.isdir(seq_root):
        print('[Err]: invalid seq root.')
        return

    img_root = seq_root + '/images'
    seq_names = [x for x in os.listdir(img_root)]
    # print(seq_names)

    seq_dir_paths = [img_root + '/' + x for x in seq_names]
    # print(seq_dir_paths)

    for seq_dir in seq_dir_paths:
        if not os.path.isdir(seq_dir):
            print('[Err]: invalid seq image dir.')
            continue

        seq_name = os.path.split(seq_dir)[-1]
        darklabel_txt_path = seq_dir + '/' + seq_name + '_gt.txt'

        # ---------- do pasing for a seq
        convert_darklabel_2_mot16(darklabel_txt_path, interval=interval,
                                  out_mot16_path=None)
        # ----------


if __name__ == '__main__':
    # convert_darklabel_2_mot16(darklabel_txt_path='F:/seq_data/images/mcmot_seq_imgs_1/mcmot_seq_imgs_1_gt.txt')
    # convert_seqs(seq_root='F:/seq_data/', interval=2)
    convert_darklabel_2_mot16(darklabel_txt_path='F:/val_seq/val_1_gt.txt',
                              interval=2,
                              out_mot16_path=None)

    print('Done.')

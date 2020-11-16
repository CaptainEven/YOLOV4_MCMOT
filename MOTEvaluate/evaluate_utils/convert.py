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
                              default_fps=12,
                              one_plus=True,
                              out_mot16_path=None):
    """
    将darklabel标注格式frame # n [id, x1, y1, x2, y2, label]
    转换成mot16格式
    """
    if not os.path.isfile(darklabel_txt_path):
        print('[Err]: invalid input file path.')
        return

    if out_mot16_path is None:
        out_fps = default_fps // int(interval)
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
                if one_plus:
                    track_id = int(line[cur]) + 1  # track_id从1开始统计
                else:
                    track_id = int(line[cur])

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
                # print(write_line_str)
                w_h.write(write_line_str + '\n')

            fr_idx += 1
        print('Total {:d} frames sampled'.format(fr_idx))

    print('{:s} written.'.format(out_mot16_path))


def convert_seqs(seq_root, interval=1, default_fps=12, one_plus=True):
    """
    """
    if not os.path.isdir(seq_root):
        print('[Err]: invalid seq root.')
        return

    seq_names = [x for x in os.listdir(seq_root) if x.endswith('.mp4')]
    for seq_name in seq_names:
        darklabel_txt_path = seq_root  + '/' + seq_name[:-4] + '_gt.txt'

        # ---------- do pasing for a seq
        convert_darklabel_2_mot16(darklabel_txt_path,
                                  interval=interval,
                                  default_fps=default_fps,
                                  one_plus=one_plus,
                                  out_mot16_path=None)
        # ----------


if __name__ == '__main__':
    # convert_darklabel_2_mot16(darklabel_txt_path='F:/seq_data/images/mcmot_seq_imgs_1/mcmot_seq_imgs_1_gt.txt')
    convert_seqs(seq_root='/mnt/diskb/even/dataset/MCMOT_Evaluate',
                 interval=1,
                 default_fps=12,
                 one_plus=True)
    # convert_darklabel_2_mot16(darklabel_txt_path='F:/val_seq/val_1_gt.txt',
    #                           interval=1,
    #                           fps=12,
    #                           one_plus=False,
    #                           out_mot16_path=None)

    print('Done.')

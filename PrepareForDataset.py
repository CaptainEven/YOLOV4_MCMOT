# encoding=utf-8

import os
from tqdm import tqdm


def modify_train_txt_f(train_txt_f_path, path_prefix):
    """
    :param train_txt_f_path:
    :param path_prefix:
    :return:
    """
    if not os.path.isfile(train_txt_f_path):
        print('[Err]: invalid file path.')
        return

    with open(train_txt_f_path, 'r', encoding='utf-8') as f_r:
        lines = f_r.readlines()
    lines = [path_prefix + os.path.split(x.strip())[-1] for x in lines]

    with open(train_txt_f_path, 'w', encoding='utf-8') as f_w:
        for line in tqdm(lines):
            f_w.write(line + '\n')


if __name__ == '__main__':
    modify_train_txt_f(train_txt_f_path='./data/coco/val2017.txt',
                       path_prefix='./data/coco/images/val2017/JPEGImages/')
    print('Done')

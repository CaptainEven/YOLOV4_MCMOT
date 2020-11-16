# encoding=utf-8

import os
from easydict import EasyDict
from MOTEvaluate.evaluate_utils.convert import convert_seqs
from MOTEvaluate.evaluate import evaluate_mcmot_seqs
from demo import run_demo, DemoRunner


# @TODO: build evaluation pipeline for test set
def evaluate_test_set(test_root):
    """
    :param test_root:
    :return:
    """
    # ----------
    # Check test root for video and dark label format label file(txt)
    # Convert darklabel label file to mot16 format
    convert_seqs(seq_root=test_root, interval=1)
    # ----------

    # ----------
    # Call mcmot_yolov4(demo.py) to do tracking(generate results.txt)
    demo = DemoRunner()
    ROOT = '/mnt/diskb/even/YOLOV4'
    demo.opt.names = ROOT + '/data/mcmot.names'
    demo.opt.cfg = ROOT + '/cfg/yolov4-tiny-3l_no_group_id_no_upsample.cfg'
    demo.opt.weights = ROOT + '/weights/v4_tiny3l_no_upsample_track_last.pt'
    demo.run()
    # ----------

    # Do evaluation using results and ground truth for each seq
    evaluate_mcmot_seqs(test_root, default_fps=12)


if __name__ == '__main__':
    evaluate_test_set(test_root='/mnt/diskb/even/dataset/MCMOT_Evaluate')
    print('Done.')

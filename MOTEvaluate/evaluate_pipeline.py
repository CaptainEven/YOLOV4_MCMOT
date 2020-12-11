# encoding=utf-8

import os
from MOTEvaluate.evaluate_utils.convert import convert_seqs
from MOTEvaluate.evaluate import evaluate_mcmot_seqs
from demo import DemoRunner


# build evaluation pipeline for test set
def evaluate_test_set(test_root):
    """
    :param test_root:
    :return:
    """
    # ---------- set Project root
    ROOT = '/mnt/diskb/even/YOLOV4'

    # ---------- init demo runner
    demo = DemoRunner()

    # ---------- set object class names
    demo.opt.names = ROOT + '/data/mcmot.names'

    # ----------- set weights and cfg file for different models
    # demo.opt.cfg = ROOT + '/cfg/' + 'yolov4-tiny-3l_no_group_id_no_upsample.cfg'
    # demo.opt.weights = ROOT + '/weights/' + 'v4_tiny3l_no_upsample_track_last.pt'

    # demo.opt.cfg = ROOT + '/cfg/' + 'yolov4_mobilev2_2l.cfg'
    # demo.opt.weights = ROOT + '/weights/' + 'track_last.pt'

    demo.opt.cfg = ROOT + '/cfg/' + 'yolov4-tiny-3l_no_group_id_no_upsample.cfg'
    demo.opt.weights = ROOT + '/weights/' + 'track_last.pt'

    if not os.path.isfile(demo.opt.cfg):
        print('[Err]: invalid cfg file.')
        return
    if not os.path.isfile(demo.opt.weights):
        print('[Err]: invalid weight file.')
        return

    print('Cfg: {:s}.'.format(demo.opt.cfg))
    print('Weights: {:s}.\n'.format(demo.opt.weights))

    # ----------- set test input videos' dir and tracking results dir
    demo.opt.videos = '/mnt/diskb/even/dataset/MCMOT_Evaluate'
    demo.opt.save_img_dir = demo.opt.videos

    # ---------- set standard out fps and interval: set test fps
    demo.opt.outFPS = 12
    demo.opt.interval = 1

    # ---------- labels preparation
    # Check test root for video and dark label format label file(txt)
    # Convert dark-label label file to mot16 format
    convert_seqs(seq_root=test_root,
                 interval=demo.opt.interval,
                 default_fps=demo.opt.outFPS,
                 one_plus=True)
    # ----------

    # ---------- Run tracking
    # Call mcmot-yolov4(demo.py) to do tracking(generate results.txt)
    # set task mode and output results type
    demo.opt.task = 'track'
    demo.opt.output_type = 'txts'

    # run tracking and output results.txt(MOT16)
    demo.run()
    # ----------

    # --------- Run evaluation
    out_fps = demo.opt.outFPS // int(demo.opt.interval)
    evaluate_mcmot_seqs(test_root, default_fps=out_fps)
    # ---------


if __name__ == '__main__':
    evaluate_test_set(test_root='/mnt/diskb/even/dataset/MCMOT_Evaluate')
    print('Done.')

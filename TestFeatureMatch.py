# encoding=utf-8

import os
import argparse


class FeatureMatcher(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--names',
                                 type=str,
                                 default='data/mcmot.names',
                                 help='*.names path')

        # ---------- cfg and weights file
        self.parser.add_argument('--cfg',
                                 type=str,
                                 default='cfg/yolov4-tiny-3l-one-feat.cfg',
                                 help='*.cfg path')

        self.parser.add_argument('--weights',
                                 type=str,
                                 default='weights/v4_tiny3l_one_feat_track_last.weights',
                                 help='weights path')
        # ----------

        # input file/folder, 0 for webcam
        self.parser.add_argument('--videos',
                                 type=str,
                                 default='/mnt/diskb/even/YOLOV4/data/videos',
                                 help='')  # 'data/samples/videos/'

        # task mode
        self.parser.add_argument('--task',
                                 type=str,
                                 default='track',
                                 help='task mode: track or detect')

        self.parser.add_argument('--input-type',
                                 type=str,
                                 default='videos',
                                 help='videos or txt')

        # ---------- Set net input image width and height
        self.parser.add_argument('--img-size', type=int, default=768, help='Image size')
        self.parser.add_argument('--net_w', type=int, default=768, help='inference size (pixels)')
        self.parser.add_argument('--net_h', type=int, default=448, help='inference size (pixels)')

        self.parser.add_argument('--num-classes',
                                 type=int,
                                 default=5,
                                 help='Number of object classes.')

        # ---------- Input image Pre-processing method
        self.parser.add_argument('--img-proc-method',
                                 type=str,
                                 default='resize',
                                 help='Image pre-processing method(letterbox, resize)')

        # ----------

        self.parser.add_argument('--cutoff',
                                 type=int,
                                 default=0,  # 0 or 44
                                 help='cutoff layer index, 0 means all layers loaded.')

        # ----------
        ## read from .npy(max_id_dict.npy file)
        max_id_dict_file_path = '/mnt/diskb/even/dataset/MCMOT/max_id_dict.npz'
        if os.path.isfile(max_id_dict_file_path):
            load_dict = np.load(max_id_dict_file_path, allow_pickle=True)
        max_id_dict = load_dict['max_id_dict'][()]

        # set device
        opt.device = str(FindFreeGPU())
        print('Using gpu: {:s}'.format(opt.device))
        device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
        opt.device = device

        # build model in track mode(do detection and reid feature vector extraction)
        self.model = Darknet(cfg=opt.cfg,
                             img_size=opt.img_size,
                             verbose=False,
                             max_id_dict=max_id_dict,
                             emb_dim=128,
                             mode=opt.task).to(opt.device)
        # print(self.model)

        # Load checkpoint
        if opt.weights.endswith('.pt'):  # pytorch format
            ckpt = torch.load(opt.weights, map_location=device)
            self.model.load_state_dict(ckpt['model'])
            if 'epoch' in ckpt.keys():
                print('Checkpoint of epoch {} loaded.\n'.format(ckpt['epoch']))
        else:  # darknet format
            load_darknet_weights(self.model, opt.weights, int(opt.cutoff))

        # set dataset
        self.dataset = LoadImages(video_path, opt.img_proc_method, net_w=opt.net_w, net_h=opt.net_h)

    def run(self):
        pass

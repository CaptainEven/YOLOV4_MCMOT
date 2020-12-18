# encoding=utf-8

import os
import argparse
from models import *
from demo import FindFreeGPU
from utils.datasets import LoadImages


class FeatureMatcher(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--names',
                                 type=str,
                                 default='data/mcmot.names',
                                 help='*.names path')

        # ----- cfg and weights file
        self.parser.add_argument('--cfg',
                                 type=str,
                                 default='cfg/yolov4-tiny-3l-one-feat.cfg',
                                 help='*.cfg path')

        self.parser.add_argument('--weights',
                                 type=str,
                                 default='weights/v4_tiny3l_one_feat_track_last.weights',
                                 help='weights path')
        # -----

        # input file/folder, 0 for webcam
        self.parser.add_argument('--video',
                                 type=str,
                                 default='/mnt/diskb/even/dataset/MCMOT_Evaluate/val_12.mp4',
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

        # ----- Set net input image width and height
        self.parser.add_argument('--img-size', type=int, default=768, help='Image size')
        self.parser.add_argument('--net_w', type=int, default=768, help='inference size (pixels)')
        self.parser.add_argument('--net_h', type=int, default=448, help='inference size (pixels)')

        self.parser.add_argument('--num-classes',
                                 type=int,
                                 default=5,
                                 help='Number of object classes.')

        # ----- Input image Pre-processing method
        self.parser.add_argument('--img-proc-method',
                                 type=str,
                                 default='resize',
                                 help='Image pre-processing method(letterbox, resize)')

        # -----

        self.parser.add_argument('--cutoff',
                                 type=int,
                                 default=0,  # 0 or 44
                                 help='cutoff layer index, 0 means all layers loaded.')

        # -----
        self.parser.add_argument('--conf', type=float, default=0.2, help='object confidence threshold')
        self.parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
        # ----------

        self.opt = self.parser.parse_args()

        # video GT
        self.darklabel_txt_path = self.opt.video[:-4] + '_gt.txt'

        if not (os.path.isfile(self.opt.video)
                and os.path.isfile(self.darklabel_txt_path)):
            print('[Err]: invalid video path or GT.')
            return

        # ----------
        ## read from .npy(max_id_dict.npy file)
        max_id_dict_file_path = '/mnt/diskb/even/dataset/MCMOT/max_id_dict.npz'
        if os.path.isfile(max_id_dict_file_path):
            load_dict = np.load(max_id_dict_file_path, allow_pickle=True)
        max_id_dict = load_dict['max_id_dict'][()]

        # set device
        self.opt.device = str(FindFreeGPU())
        print('Using gpu: {:s}'.format(self.opt.device))
        device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.opt.device)
        self.opt.device = device

        # build model in track mode(do detection and reid feature vector extraction)
        self.model = Darknet(cfg=self.opt.cfg,
                             img_size=self.opt.img_size,
                             verbose=False,
                             max_id_dict=max_id_dict,
                             emb_dim=128,
                             mode=self.opt.task).to(self.opt.device)
        # print(self.model)

        # Load checkpoint
        if self.opt.weights.endswith('.pt'):  # py-torch format
            ckpt = torch.load(self.opt.weights, map_location=device)
            self.model.load_state_dict(ckpt['model'])
            if 'epoch' in ckpt.keys():
                print('Checkpoint of epoch {} loaded.\n'.format(ckpt['epoch']))
        else:  # dark-net format
            load_darknet_weights(self.model, self.opt.weights, int(self.opt.cutoff))

        # set dataset
        self.dataset = LoadImages(self.opt.video, self.opt.img_proc_method, self.opt.net_w, self.opt.net_h)


    def load_dets(self):  #
        """
        x1, y1, x2, y2, conf, cls_id = det
        :return:
        """

    def load_gt(self):
        """

        :return:
        """

    def get_tp(self):  # Get true positive
        """
        :return:
        """

    def run(self):
        # iterate tracking results of each frame
        for fr_id, (path, img, img0, vid_cap) in enumerate(dataset):
            img = torch.from_numpy(img).to(self.opt.device)
            img = img.float()  # uint8 to fp32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Get image size
            net_h, net_w = img.shape[2:]
            orig_h, orig_w, _ = img0.shape  # H×W×C

            with torch.no_grad():
                pred, pred_orig, reid_feat_out = self.model.forward(img, augment=self.opt.augment)
                pred = pred.float()

                # ----- apply NMS
                pred = non_max_suppression(predictions=pred,
                                           conf_thres=self.opt.conf_thres,
                                           iou_thres=self.opt.iou_thres,
                                           merge=False,
                                           classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                dets = pred[0]  # assume batch_size == 1 here
                if dets is None:
                    print('[Warning]: no objects detected.')
                    return None

                # compute TPs for current frame

                # to storage each detected obj's feature vector
                reid_feat_list = []


if __name__ == '__main__':
    matcher = FeatureMatcher()
    matcher.run()
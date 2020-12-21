# encoding=utf-8

import os
import argparse
from collections import defaultdict
from models import *
from utils.utils import map_resize_back, map_to_orig_coords
from tracking_utils import visualization as vis
from mAPEvaluate.cmp_det_label_sf import box_iou
from demo import FindFreeGPU
from utils.datasets import LoadImages


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
        # -----

        # input file/folder, 0 for webcam
        self.parser.add_argument('--video',
                                 type=str,
                                 default='/mnt/diskb/even/dataset/MCMOT_Evaluate/val_0.mp4',
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
        self.parser.add_argument('--augment', action='store_true', help='augmented inference')
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

        # class name to class id and class id to class name
        names = load_classes(self.opt.names)
        self.id2cls = defaultdict(str)
        self.cls2id = defaultdict(int)
        for cls_id, cls_name in enumerate(names):
            self.id2cls[cls_id] = cls_name
            self.cls2id[cls_name] = cls_id

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

        # Put model to device and set eval mode
        self.model.to(device).eval()

        # set dataset
        self.dataset = LoadImages(self.opt.video, self.opt.img_proc_method, self.opt.net_w, self.opt.net_h)

    def load_gt(self, img_w, img_h, one_plus=True, cls_id=0):
        """
        Convert to x1, y1, x2, y2, tr_id(start from 1), cls_id format
        :param img_w: image width
        :param img_h: image height
        :param cls_id: specified object class id
        :return:
        """
        # each frame contains a list
        objs_gt = []

        with open(self.darklabel_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            # traverse each frame
            fr_idx = 0
            for fr_i, line in enumerate(lines):
                line = line.strip().split(',')
                fr_id = int(line[0])
                n_objs = int(line[1])

                # traverse each object of the frame
                fr_objs = []
                for cur in range(2, len(line), 6):
                    # read object class id
                    class_type = line[cur + 5].strip()
                    class_id = self.cls2id[class_type]  # class type => class id

                    # read track id
                    if one_plus:
                        track_id = int(line[cur]) + 1  # track_id从1开始统计
                    else:
                        track_id = int(line[cur])

                    # read bbox
                    x1, y1 = int(line[cur + 1]), int(line[cur + 2])
                    x2, y2 = int(line[cur + 3]), int(line[cur + 4])

                    # clip bbox
                    x1 = x1 if x1 >= 0 else 0
                    x1 = x1 if x1 < img_w else img_w - 1
                    y1 = y1 if y1 >= 0 else 0
                    y1 = y1 if y1 < img_h else img_h - 1
                    x2 = x2 if x2 >= 0 else 0
                    x2 = x2 if x2 < img_w else img_w - 1
                    y2 = y2 if y2 >= 0 else 0
                    y2 = y2 if y2 < img_h else img_h - 1

                    fr_objs.append([x1, y1, x2, y2, track_id, class_id])

                objs_gt.append(fr_objs)

        return objs_gt

    def get_tp(self, fr_id, dets, cls_id=0):  # Get true positive
        """
        Compute true positives for the current frame and specified object class
        :param fr_id:
        :param dets: x1, y1, x2, y2, score, cls_id
        :param cls_id:
        :return:
        """
        assert len(self.objs_gt) == self.dataset.nframes
        # print('Compute true positives for frame {:d}...'.format(fr_id))

        # get GT objs of current frame for specified object class
        fr_objs_gt = self.objs_gt[fr_id]
        objs_gt = [obj for obj in fr_objs_gt if obj[-1] == cls_id]
        # print(objs_gt)

        # get predicted objs of current frame for specified object class
        objs_pred = [det for det in dets if det[-1] == cls_id]
        # print(objs_pred)

        # compute TPs
        pred_match_flag = [False for n in range(len(objs_pred))]
        correct = 0
        TPs = []
        pred_to_gt_mapping = defaultdict(int)
        for i, obj_gt in enumerate(objs_gt):  # each gt obj
            best_iou = 0
            best_pred_id = -1
            for j, obj_pred in enumerate(objs_pred):  # each pred obj
                box_gt = obj_gt[:4]
                box_pred = obj_pred[:4]
                b_iou = box_iou(box_gt, box_pred)  # compute iou
                if obj_pred[4] > self.opt.conf and b_iou > best_iou:  # meet the conf thresh
                    best_pred_id = j
                    best_iou = b_iou

            # meet the iou thresh and not matched yet
            if best_iou > self.opt.iou and not pred_match_flag[best_pred_id]:
                correct += 1
                pred_match_flag[best_pred_id] = True  # set flag true for matched prediction
                TPs.append(obj_pred)
                pred_to_gt_mapping[best_pred_id] = i

        return TPs

    def run(self, cls_id=0, img_w=1920, img_h=1080, viz_dir=None):
        """
        :param cls_id:
        :param img_w:
        :param img_h:
        :param viz_dir:
        :return:
        """
        # create viz dir
        if viz_dir != None:
            if not os.path.isdir(viz_dir):
                os.makedirs(viz_dir)

        # read net input width and height
        net_h, net_w = self.opt.net_h, self.opt.net_w

        # ---------- load GT for all frames
        self.objs_gt = self.load_gt(img_w, img_h, cls_id=cls_id)

        # ---------- iterate tracking results of each frame
        for fr_id, (path, img, img0, vid_cap) in enumerate(self.dataset):
            img = torch.from_numpy(img).to(self.opt.device)
            img = img.float()  # uint8 to fp32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # get current frames's image size
            img_h, img_w = img0.shape[:2]  # H×W×C

            with torch.no_grad():
                pred, pred_orig, reid_feat_out = self.model.forward(img, augment=self.opt.augment)
                pred = pred.float()

                # ----- get dets: in x1, y1, x2, y2, score, cls_id format
                # apply NMS
                pred = non_max_suppression(predictions=pred,
                                           conf_thres=self.opt.conf,
                                           iou_thres=self.opt.iou,
                                           merge=False,
                                           classes=None,
                                           agnostic=False)
                dets = pred[0]  # assume batch_size == 1 here
                if dets is None:
                    print('[Warning]: no objects detected.')
                    return None

                if self.opt.img_proc_method == 'resize':
                    dets = map_resize_back(dets, net_w, net_h, img_w, img_h)
                elif self.opt.img_proc_method == 'letterbox':
                    dets = map_to_orig_coords(dets, net_w, net_h, img_w, img_h)

                dets = dets.detach().cpu().numpy()

            # --- viz dets
            if viz_dir != None:
                img_plot = vis.plot_detects(img0, dets, len(self.cls2id), fr_id, self.id2cls)
                det_img_path = viz_dir + '/' + str(fr_id) + '_det' + '.jpg'
                cv2.imwrite(det_img_path, img_plot)

            # ----- get GT for current frame
            self.gt_cur = self.objs_gt[fr_id]

            # --- viz GTs
            if viz_dir != None:
                objs_gt = np.array(self.objs_gt[fr_id])
                objs_gt[:, 4] = 1.0
                img_plot = vis.plot_detects(img0, objs_gt, len(self.cls2id), fr_id, self.id2cls)
                det_img_path = viz_dir + '/' + str(fr_id) + '_gt' + '.jpg'
                cv2.imwrite(det_img_path, img_plot)

            # compute TPs for current frame
            TPs = self.get_tp(fr_id, dets, cls_id=cls_id)  # only for car(cls_id == 0)
            print('{:d} true positive cars.'.format(len(TPs)))

            # ---------- matching statistics
            if fr_id > 0:  # start from the second image
                # ----- get gt_pre
                self.gt_pre = self.objs_gt[fr_id - 1]
                self.gt_pre = [obj for obj in self.gt_pre if obj[-1] == cls_id]

                # ----- get intersection of pre and cur GT for specified object class
                # filtering
                tr_ids_cur = [x[4] for x in self.gt_cur]
                tr_ids_pre = [x[4] for x in self.gt_pre]
                tr_ids_common = set(tr_ids_cur) & set(tr_ids_pre)  # GTs intersection
                gt_pre_tmp = [x for x in self.gt_pre if x[4] in tr_ids_common]
                gt_cur_tmp = [x for x in self.gt_cur if x[4] in tr_ids_common]
                self.gt_pre = gt_pre_tmp
                self.gt_cur = gt_cur_tmp

                #

            # ---------- update
            self.TPs_pre = TPs


if __name__ == '__main__':
    matcher = FeatureMatcher()
    matcher.run(viz_dir=None)  # '/mnt/diskc/even/viz'

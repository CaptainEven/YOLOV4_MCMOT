# encoding=utf-8

import os
import argparse
import cv2
from collections import defaultdict
from models import *
from utils.utils import map_resize_back, map_to_orig_coords
from tracker.multitracker import cos
from tracking_utils import visualization as vis
from mAPEvaluate.cmp_det_label_sf import box_iou
from demo import FindFreeGPU
from utils.datasets import LoadImages
from tqdm import tqdm


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
                                 default='cfg/yolov4-tiny-3l_no_group_id_tmp.cfg',
                                 help='*.cfg path')

        self.parser.add_argument('--weights',
                                 type=str,
                                 default='weights/tmp_track_last.weights',
                                 help='weights path')
        # ----------
        # -----

        # input seq videos
        self.parser.add_argument('--videos',
                                 type=str,
                                 default='/mnt/diskb/even/dataset/MCMOT_Evaluate',
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

        # ----- Set ReID feature map output layer ids
        self.parser.add_argument('--feat-out-ids',
                                 type=str,
                                 default='-1',  # '-5, -3, -1' or '-9, -5, -1' or '-1'
                                 help='reid feature map output layer ids.')

        self.parser.add_argument('--dim',
                                 type=int,
                                 default=128,
                                 help='reid feature map output embedding dimension')

        self.parser.add_argument('--bin-step',
                                 type=int,
                                 default=5,
                                 help='number of bins for cosine similarity statistics(10 or 5).')

        # -----
        self.parser.add_argument('--conf', type=float, default=0.2, help='object confidence threshold')
        self.parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
        # ----------
        self.parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
        self.parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

        self.opt = self.parser.parse_args()

        # class name to class id and class id to class name
        names = load_classes(self.opt.names)
        self.id2cls = defaultdict(str)
        self.cls2id = defaultdict(int)
        for cls_id, cls_name in enumerate(names):
            self.id2cls[cls_id] = cls_name
            self.cls2id[cls_name] = cls_id

        # video GT
        if not os.path.isdir(self.opt.videos):
            print('[Err]: invalid videos dir.')
            return

        self.videos = [self.opt.videos + '/' + x for x in os.listdir(self.opt.videos) if x.endswith('.mp4')]

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
                             emb_dim=self.opt.dim,
                             feat_out_ids=self.opt.feat_out_ids,
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

        # put model to device and set eval mode
        self.model.to(device).eval()

        # statistics
        self.correct_sim_bins_dict = defaultdict(int)
        self.wrong_sim_bins_dict = defaultdict(int)
        for edge in range(0, 100, self.opt.bin_step):
            self.correct_sim_bins_dict[edge] = 0
            self.wrong_sim_bins_dict[edge] = 0

        # gap of the same object class and different object class
        self.min_same_class_sim = 1.0   # init to the max
        self.max_diff_class_sim = -1.0  # init to the min

        print('Feature matcher init done.')

    def run(self, cls_id=0, img_w=1920, img_h=1080, viz_dir=None):
        """
        :param viz_dir:
        :return:
        """
        # create viz dir
        if viz_dir != None:
            if not os.path.isdir(viz_dir):
                os.makedirs(viz_dir)
            else:
                shutil.rmtree(viz_dir)
                os.makedirs(viz_dir)

        # traverse each video seq
        mean_precision = 0.0
        valid_seq_cnt = 0
        num_tps_total = 0
        for video_path in self.videos:  # .mp4
            if not os.path.isfile(video_path):
                print('[Warning]: {:s} not exists.'.format(video_path))
                continue

            # get video seq name
            seq_name = os.path.split(video_path)[-1][:-4]

            # current video seq's gt label
            self.darklabel_txt_path = video_path[:-4] + '_gt.txt'
            if not os.path.isfile(self.darklabel_txt_path):
                print('[Warning]: {:s} not exists.'.format(self.darklabel_txt_path))
                continue

            # current video seq's dataset
            self.dataset = LoadImages(video_path, self.opt.img_proc_method, self.opt.net_w, self.opt.net_h)

            # run a vide oseq
            print('Run seq {:s}...'.format(video_path))
            precision, num_tps = self.run_a_seq(seq_name, cls_id, img_w, img_h, viz_dir)
            mean_precision += precision
            num_tps_total += num_tps
            print('Seq {:s} done.\n'.format(video_path))

            valid_seq_cnt += 1

        mean_precision /= float(valid_seq_cnt)

        # histogram statistics
        num_correct = [self.correct_sim_bins_dict[x] for x in self.correct_sim_bins_dict]
        num_wrong = [self.wrong_sim_bins_dict[x] for x in self.wrong_sim_bins_dict]
        num_total_correct = sum(num_correct)
        num_total_wrong = sum(num_wrong)
        num_total = num_total_correct + num_total_wrong
        # print(num_total_wrong / num_total)

        print(self.correct_sim_bins_dict)
        print(self.wrong_sim_bins_dict)

        # detailed statistics
        for edge in range(0, 100, self.opt.bin_step):
            wrong_ratio = self.wrong_sim_bins_dict[edge] / num_total * 100.0
            print('Wrong   [{:d}, {:d}]: {:.3f}'.format(edge, edge + self.opt.bin_step, wrong_ratio))

        for edge in range(0, 100, self.opt.bin_step):
            correct_ratio = self.correct_sim_bins_dict[edge] / num_total * 100.0
            print('Correct [{:d}, {:d}]: {:.3f}'.format(edge, edge + self.opt.bin_step, correct_ratio))

        print('\nTotal {:d} true positives detected.'.format(num_tps_total))
        print('Total {:d} matches tested.'.format(num_total))
        print('Correct matched number: {:d}'.format(num_total_correct))
        print('Wrong matched number:   {:d}'.format(num_total_wrong))
        print('Mean precision:    {:.3f}%'.format(mean_precision * 100.0))
        print('Average precision: {:.3f}%'.format(num_total_correct / num_total * 100.0))
        print('Min same class similarity: {:.3f}'.format(self.min_same_class_sim))
        print('Max diff class similarity: {:.3f}'.format(self.max_diff_class_sim))

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

    def clip_bbox(self, bbox, w, h):
        """
        :param bbox: x1, y1, x2, y2
        :param w: max x
        :param h: max y
        :return:
        """
        x1, y1, x2, y2 = bbox

        if x1 >= x2 or y1 >= y2:
            print('[Err]: wrong bbox.')
            return

        x1 = x1 if x1 < w else w - 1
        x2 = x2 if x2 < w else w - 1
        y1 = y1 if y1 < h else h - 1
        y2 = y2 if y2 < h else h - 1

        bbox = x1, y1, x2, y2
        return bbox

    def get_tp_three_feat(self, fr_id, dets, yolo_inds, cls_id=0):
        """
        :param fr_id:
        :param dets:
        :param yolo_inds:
        :param cls_id:
        :return:
        """
        assert len(self.objs_gt) == self.dataset.nframes

        # get GT objs of current frame for specified object class
        fr_objs_gt = self.objs_gt[fr_id]
        objs_gt = [obj for obj in fr_objs_gt if obj[-1] == cls_id]

        dets = dets.tolist()
        det_ids = [dets.index(x) for x in dets if x[-1] == cls_id]
        objs_pred = [dets[id] for id in det_ids]
        yolo_inds = yolo_inds.squeeze().numpy()
        yolo_inds_pred = [yolo_inds[id] for id in det_ids]

        # compute TPs
        pred_match_flag = [False for n in range(len(objs_pred))]
        correct = 0
        TPs = []
        GT_tr_ids = []  # GT ids for each TP
        TP_yolo_inds = []
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

                # clipping predicted bbox
                objs_pred[best_pred_id][:4] = self.clip_bbox(objs_pred[best_pred_id][:4], self.img_w, self.img_h)

                TPs.append(objs_pred[best_pred_id])
                GT_tr_ids.append(obj_gt[4])
                TP_yolo_inds.append(yolo_inds_pred[best_pred_id])

        return TPs, GT_tr_ids, TP_yolo_inds

    def get_tp_one_feat(self, fr_id, dets, cls_id=0):
        """
        Compute true positives for the current frame and specified object class
        :param fr_id:
        :param dets: x1, y1, x2, y2, score, cls_id
        :param cls_id:
        :return:
        """
        assert len(self.objs_gt) == self.dataset.nframes

        # get GT objs of current frame for specified object class
        fr_objs_gt = self.objs_gt[fr_id]
        objs_gt = [obj for obj in fr_objs_gt if obj[-1] == cls_id]

        # get predicted objs of current frame for specified object class
        objs_pred = [det for det in dets if det[-1] == cls_id]

        # compute TPs
        pred_match_flag = [False for n in range(len(objs_pred))]
        correct = 0
        TPs = []
        GT_tr_ids = []  # GT ids for each TP
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

                # clipping predicted bbox
                objs_pred[best_pred_id][:4] = self.clip_bbox(objs_pred[best_pred_id][:4], self.img_w, self.img_h)

                TPs.append(objs_pred[best_pred_id])
                GT_tr_ids.append(obj_gt[4])

        return TPs, GT_tr_ids

    def get_feature(self, reid_feat_map,
                    feat_map_w, feat_map_h,
                    img_w, img_h,
                    x1, y1, x2, y2):
        """
        Get feature vector
        :param reid_feat_map:
        :param feat_map_w:
        :param feat_map_h:
        :param img_w:
        :param img_h:
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :return:
        """
        # get center point
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5

        # map center point from net scale to feature map scale(1/4 of net input size)
        center_x = center_x / float(img_w)
        center_x = center_x * float(feat_map_w)
        center_y = center_y / float(img_h)
        center_y = center_y * float(feat_map_h)

        # convert to int64 for indexing
        center_x = int(center_x + 0.5)
        center_y = int(center_y + 0.5)

        # to avoid the object center out of reid feature map's range
        center_x = np.clip(center_x, 0, feat_map_w - 1)
        center_y = np.clip(center_y, 0, feat_map_h - 1)

        # get reid feature vector and put into a dict
        reid_feat_vect = reid_feat_map[0, :, center_y, center_x]

        return reid_feat_vect

    def run_a_seq(self, seq_name, cls_id=0, img_w=1920, img_h=1080, viz_dir=None):
        """
        :param seq_name:
        :param cls_id:
        :param img_w:
        :param img_h:
        :param viz_dir:
        :return:
        """
        # define image width and height
        self.img_w, self.img_h = img_w, img_h

        # read net input width and height
        net_h, net_w = self.opt.net_h, self.opt.net_w

        # ---------- load GT for all frames
        self.objs_gt = self.load_gt(self.img_w, self.img_h, cls_id=cls_id)

        # ---------- iterate tracking results of each frame
        total = 0
        correct = 0
        sim_sum = 0.0
        num_tps = 0
        for fr_id, (path, img, img0, vid_cap) in tqdm(enumerate(self.dataset)):
            img = torch.from_numpy(img).to(self.opt.device)
            img = img.float()  # uint8 to fp32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # get current frames's image size
            img_h, img_w = img0.shape[:2]  # H×W×C

            with torch.no_grad():
                pred = None
                if len(self.model.feat_out_ids) == 3:
                    pred, pred_orig, reid_feat_out, yolo_inds = self.model.forward(img, augment=self.opt.augment)

                    # ----- get reid feature map: reid_feat_out: GPU -> CPU and L2 normalize
                    feat_tmp_list = []
                    for tmp in reid_feat_out:
                        # L2 normalize the feature map(feature map scale)
                        tmp = F.normalize(tmp, dim=1)

                        if fr_id == 0:
                            # feature map size
                            n, c, h, w = tmp.shape
                            print('Feature map size: {:d}×{:d}'.format(w, h))

                        # GPU -> CPU
                        tmp = tmp.detach().cpu().numpy()

                        feat_tmp_list.append(tmp)

                    reid_feat_out = feat_tmp_list

                elif len(self.model.feat_out_ids) == 1:
                    pred, pred_orig, reid_feat_out = self.model.forward(img, augment=self.opt.augment)

                    # ----- get reid feature map: reid_feat_out: GPU -> CPU and L2 normalize
                    reid_feat_map = reid_feat_out[0]

                    if fr_id == 0:
                        # feature map size
                        n, c, h, w = reid_feat_map.shape
                        print('Feature map size: {:d}×{:d}'.format(w, h))

                    # L2 normalize the feature map(feature map scale(1/4 or 1/8 of net input size))
                    reid_feat_map = F.normalize(reid_feat_map, dim=1)

                    reid_feat_map = reid_feat_map.detach().cpu().numpy()
                    b, reid_dim, feat_map_h, feat_map_w = reid_feat_map.shape

                # ----- apply NMS
                if len(self.model.feat_out_ids) == 3:
                    pred, pred_yolo_ids = non_max_suppression_with_yolo_inds(predictions=pred,
                                                                             yolo_inds=yolo_inds,
                                                                             conf_thres=self.opt.conf,
                                                                             iou_thres=self.opt.iou,
                                                                             merge=False,
                                                                             classes=self.opt.classes,
                                                                             agnostic=self.opt.agnostic_nms)
                    dets_yolo_ids = pred_yolo_ids[0]  # assume batch_size == 1 here

                elif len(self.model.feat_out_ids) == 1:
                    pred = non_max_suppression(predictions=pred,
                                               conf_thres=self.opt.conf,
                                               iou_thres=self.opt.iou,
                                               merge=False,
                                               classes=self.opt.classes,
                                               agnostic=self.opt.agnostic_nms)

                dets = pred[0]  # assume batch_size == 1 here
                if dets is None:
                    print('[Warning]: no objects detected.')
                    num_tps += 0
                    continue

                # ----- Rescale boxes from net size to img size
                if self.opt.img_proc_method == 'resize':
                    dets = map_resize_back(dets, net_w, net_h, img_w, img_h)
                elif self.opt.img_proc_method == 'letterbox':
                    dets = map_to_orig_coords(dets, net_w, net_h, img_w, img_h)

                dets = dets.detach().cpu().numpy()

            # # --- viz dets
            # if viz_dir != None:
            #     img_plot = vis.plot_detects(img0, dets, len(self.cls2id), fr_id, self.id2cls)
            #     det_img_path = viz_dir + '/' + str(fr_id) + '_det' + '.jpg'
            #     cv2.imwrite(det_img_path, img_plot)

            # ----- get GT for current frame
            self.gt_cur = self.objs_gt[fr_id]

            # # --- viz GTs
            # if viz_dir != None:
            #     objs_gt = np.array(self.objs_gt[fr_id])
            #     objs_gt[:, 4] = 1.0
            #     img_plot = vis.plot_detects(img0, objs_gt, len(self.cls2id), fr_id, self.id2cls)
            #     det_img_path = viz_dir + '/' + str(fr_id) + '_gt' + '.jpg'
            #     cv2.imwrite(det_img_path, img_plot)

            # ----- compute TPs for current frame
            if len(self.model.feat_out_ids) == 3:
                TPs, GT_tr_ids, TP_yolo_inds = self.get_tp_three_feat(fr_id, dets, dets_yolo_ids, cls_id=cls_id)
            elif len(self.model.feat_out_ids) == 1:
                TPs, GT_tr_ids = self.get_tp_one_feat(fr_id, dets, cls_id=cls_id)  # only for car(cls_id == 0)
            # print('{:d} true positive cars.'.format(len(TPs)))

            num_tps += len(TPs)

            # ----- build mapping from TP id to GT track id
            tpid_to_gttrid = [GT_tr_ids[x] for x in range(len(TPs))]

            # ---------- matching statistics
            if fr_id > 0:  # start from the second image
                # ----- get GT for the last frame
                objs_pre_gt = self.objs_gt[fr_id - 1]
                self.gt_pre = [obj for obj in objs_pre_gt if obj[-1] == cls_id]

                # ----- get intersection of pre and cur GT for the specified object class
                # filtering
                # tr_ids_cur = [x[4] for x in self.gt_cur]
                # tr_ids_pre = [x[4] for x in self.gt_pre]
                # tr_ids_gt_common = set(tr_ids_cur) & set(tr_ids_pre)  # GTs intersection
                # gt_pre_tmp = [x for x in self.gt_pre if x[4] in tr_ids_gt_common]
                # gt_cur_tmp = [x for x in self.gt_cur if x[4] in tr_ids_gt_common]
                # self.gt_pre = gt_pre_tmp
                # self.gt_cur = gt_cur_tmp

                # ----- get intersection between pre and cur TPs for the specified object class
                tr_ids_tp_common = set(self.GT_tr_ids_pre) & set(GT_tr_ids)
                TPs_ids_pre = [self.GT_tr_ids_pre.index(x) for x in self.GT_tr_ids_pre if x in tr_ids_tp_common]
                TPs_ids_cur = [GT_tr_ids.index(x) for x in GT_tr_ids if x in tr_ids_tp_common]

                TPs_pre = [self.TPs_pre[x] for x in TPs_ids_pre]
                TPs_cur = [TPs[x] for x in TPs_ids_cur]

                if len(self.model.feat_out_ids) == 3:
                    TP_yolo_inds_pre = [self.TP_yolo_inds_pre[x] for x in TPs_ids_pre]
                    TP_yolo_inds_cur = [TP_yolo_inds[x] for x in TPs_ids_cur]

                assert len(TPs_pre) == len(TPs_cur)

                # ----- update total pairs
                total += len(TPs_cur)

                # ----- greedy matching...
                # print('Frame {:d} start matching for {:d} TP pairs.'.format(fr_id, len(TPs_cur)))
                if len(self.model.feat_out_ids) == 1:  # one feature map layer
                    for tpid_cur, det_cur in zip(TPs_ids_cur, TPs_cur):  # current frame as row
                        x1_cur, y1_cur, x2_cur, y2_cur = det_cur[:4]
                        reid_feat_vect_cur = self.get_feature(reid_feat_map,
                                                              feat_map_w, feat_map_h,
                                                              img_w, img_h,
                                                              x1_cur, y1_cur, x2_cur, y2_cur)

                        best_sim = -1.0
                        best_tpid_pre = -1
                        for tpid_pre, det_pre in zip(TPs_ids_pre, TPs_pre):  # previous frame as col
                            x1_pre, y1_pre, x2_pre, y2_pre = det_pre[:4]

                            reid_feat_vect_pre = self.get_feature(self.reid_feat_map_pre,
                                                                  feat_map_w, feat_map_h,
                                                                  img_w, img_h,
                                                                  x1_pre, y1_pre, x2_pre, y2_pre)

                            # --- compute cosine of cur and pre corresponding feature vector
                            sim = cos(reid_feat_vect_cur, reid_feat_vect_pre)
                            if sim > best_sim:
                                best_sim = sim
                                best_tpid_pre = tpid_pre

                        # determine matched right or not
                        gt_tr_id_pre = self.tpid_to_gttrid_pre[best_tpid_pre]
                        gt_tr_id_cur = tpid_to_gttrid[tpid_cur]

                        # if matching correct
                        if gt_tr_id_pre == gt_tr_id_cur:
                            # update correct number
                            correct += 1
                            sim_sum += best_sim

                            # if do visualization for correct and wrong match
                            if viz_dir != None:
                                save_path = viz_dir + '/' \
                                            + 'correct_match_{:s}_fr{:d}id{:d}-fr{:d}id{:d}-sim{:.3f}.jpg' \
                                                .format(seq_name, fr_id - 1, gt_tr_id_pre, fr_id, gt_tr_id_cur, best_sim)

                            # do min similarity statistics of same object class
                            if best_sim < self.min_same_class_sim:
                                self.min_same_class_sim = best_sim

                            # do cosine similarity statistics
                            best_sim *= 100.0
                            edge = int(best_sim / self.opt.bin_step) * self.opt.bin_step
                            self.correct_sim_bins_dict[edge] += 1

                        else:  # visualize the wrong match:
                            # wrong match img saving path
                            if viz_dir != None:
                                save_path = viz_dir + '/' \
                                            + 'wrong_match_{:s}_fr{:d}id{:d}-fr{:d}id{:d}-sim{:.3f}.jpg' \
                                                .format(seq_name, fr_id - 1, gt_tr_id_pre, fr_id, gt_tr_id_cur, best_sim)

                            # do max similarity statistics of the different object class
                            if best_sim > self.max_diff_class_sim:
                                self.max_diff_class_sim = best_sim

                            # do cosine similarity statistics
                            best_sim *= 100.0
                            edge = int(best_sim / self.opt.bin_step) * self.opt.bin_step
                            self.wrong_sim_bins_dict[edge] += 1

                        if viz_dir != None:
                            # ----- plot
                            # text and line format
                            text_scale = max(1.0, img_w / 500.0)  # 1600.
                            text_thickness = 2
                            line_thickness = max(1, int(img_w / 500.0))

                            img0_pre = self.img0_pre.copy()
                            x1_pre, y1_pre, x2_pre, y2_pre = self.TPs_pre[best_tpid_pre][:4]  # best match bbox
                            cv2.rectangle(img0_pre,
                                          (int(x1_pre), int(y1_pre)),
                                          (int(x2_pre), int(y2_pre)),
                                          [0, 0, 255],
                                          thickness=line_thickness)
                            cv2.putText(img0_pre,
                                        'id{:d}'.format(gt_tr_id_pre),
                                        (int(x1_pre), int(y1_pre)),
                                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                                        fontScale=text_scale,
                                        color=[0, 255, 0],
                                        thickness=text_thickness)

                            img0_cur = img0.copy()
                            cv2.rectangle(img0_cur,
                                          (int(x1_cur), int(y1_cur)),
                                          (int(x2_cur), int(y2_cur)),
                                          [0, 0, 255],
                                          thickness=line_thickness)
                            cv2.putText(img0_cur,
                                        'id{:d}'.format(gt_tr_id_cur),
                                        (int(x1_cur), int(y1_cur)),
                                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                                        fontScale=text_scale,
                                        color=[0, 255, 0],
                                        thickness=text_thickness)

                            img_save = np.zeros((2 * img_h, img_w, 3), dtype=np.uint8)
                            img_save[:img_h, :, :] = img0_pre
                            img_save[img_h:2 * img_h, :, :] = img0_cur
                            cv2.imwrite(save_path, img_save)

                elif len(self.model.feat_out_ids) == 3:
                    for tpid_cur, det_cur, yolo_id_cur in zip(TPs_ids_cur, TPs_cur, TP_yolo_inds_cur):  # current frame as row
                        x1_cur, y1_cur, x2_cur, y2_cur = det_cur[:4]

                        reid_feat_map_cur = reid_feat_out[yolo_id_cur]
                        b, reid_dim, feat_map_h_cur, feat_map_w_cur = reid_feat_map_cur.shape

                        reid_feat_vect_cur = self.get_feature(reid_feat_map_cur,
                                                              feat_map_w_cur, feat_map_h_cur,
                                                              img_w, img_h,
                                                              x1_cur, y1_cur, x2_cur, y2_cur)

                        best_sim = -1.0
                        best_tpid_pre = -1
                        for tpid_pre, det_pre, yolo_id_pre in zip(TPs_ids_pre, TPs_pre, TP_yolo_inds_pre):  # previous frame as col
                            x1_pre, y1_pre, x2_pre, y2_pre = det_pre[:4]

                            reid_feat_map_pre = self.reid_feat_out_pre[yolo_id_pre]
                            b, reid_dim, feat_map_h_pre, feat_map_w_pre = reid_feat_map_pre.shape

                            reid_feat_vect_pre = self.get_feature(reid_feat_map_pre,
                                                                  feat_map_w_pre, feat_map_h_pre,
                                                                  img_w, img_h,
                                                                  x1_pre, y1_pre, x2_pre, y2_pre)

                            # --- compute cosine of cur and pre corresponding feature vector
                            sim = cos(reid_feat_vect_cur, reid_feat_vect_pre)
                            if sim > best_sim:
                                # update correct number
                                best_sim = sim
                                best_tpid_pre = tpid_pre

                        # determine matched right or not
                        gt_tr_id_pre = self.tpid_to_gttrid_pre[best_tpid_pre]
                        gt_tr_id_cur = tpid_to_gttrid[tpid_cur]

                        # update correct
                        if gt_tr_id_pre == gt_tr_id_cur:
                            correct += 1
                            sim_sum += best_sim

                            # if do visualization for correct and wrong match
                            if viz_dir != None:
                                save_path = viz_dir + '/' \
                                            + 'correct_match_{:s}_fr{:d}id{:d}-fr{:d}id{:d}-sim{:.3f}.jpg' \
                                                .format(seq_name, fr_id - 1, gt_tr_id_pre, fr_id, gt_tr_id_cur, best_sim)

                            # do cosine similarity statistics
                            best_sim *= 100.0
                            edge = int(best_sim / self.opt.bin_step) * self.opt.bin_step
                            self.correct_sim_bins_dict[edge] += 1

                        else:  # if wrong matching
                            if viz_dir != None:
                                save_path = viz_dir + '/' \
                                            + 'wrong_match_{:s}_fr{:d}id{:d}-fr{:d}id{:d}-sim{:.3f}.jpg' \
                                                .format(seq_name, fr_id - 1, gt_tr_id_pre, fr_id, gt_tr_id_cur, best_sim)

                            # do cosine similarity statistics
                            best_sim *= 100.0
                            edge = int(best_sim / self.opt.bin_step) * self.opt.bin_step
                            self.wrong_sim_bins_dict[edge] += 1

                        if viz_dir != None:
                            # ----- plot
                            # text and line format
                            text_scale = max(1.0, img_w / 500.0)  # 1600.
                            text_thickness = 2
                            line_thickness = max(1, int(img_w / 500.0))

                            img0_pre = self.img0_pre.copy()
                            x1_pre, y1_pre, x2_pre, y2_pre = self.TPs_pre[best_tpid_pre][:4]  # get best match bbox
                            cv2.rectangle(img0_pre,
                                          (int(x1_pre), int(y1_pre)),
                                          (int(x2_pre), int(y2_pre)),
                                          [0, 0, 255],
                                          thickness=line_thickness)
                            cv2.putText(img0_pre,
                                        'id{:d}'.format(gt_tr_id_pre),
                                        (int(x1_pre), int(y1_pre)),
                                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                                        fontScale=text_scale,
                                        color=[0, 255, 0],
                                        thickness=text_thickness)

                            img0_cur = img0.copy()
                            cv2.rectangle(img0_cur,
                                          (int(x1_cur), int(y1_cur)),
                                          (int(x2_cur), int(y2_cur)),
                                          [0, 0, 255],
                                          thickness=line_thickness)
                            cv2.putText(img0_cur,
                                        'id{:d}'.format(gt_tr_id_cur),
                                        (int(x1_cur), int(y1_cur)),
                                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                                        fontScale=text_scale,
                                        color=[0, 255, 0],
                                        thickness=text_thickness)

                            img_save = np.zeros((2 * img_h, img_w, 3), dtype=np.uint8)
                            img_save[:img_h, :, :] = img0_pre
                            img_save[img_h:2 * img_h, :, :] = img0_cur
                            cv2.imwrite(save_path, img_save)

            # ---------- update
            self.TPs_pre = TPs
            self.GT_tr_ids_pre = GT_tr_ids
            self.tpid_to_gttrid_pre = tpid_to_gttrid

            if len(self.model.feat_out_ids) == 1:
                self.reid_feat_map_pre = reid_feat_map  # contains 1 feature map
            elif len(self.model.feat_out_ids) == 3:
                self.reid_feat_out_pre = reid_feat_out  # contains 3 feature map
                self.TP_yolo_inds_pre = TP_yolo_inds

            self.img0_pre = img0

        precision = correct / total
        print('Precision: {:.3f}%, mean cos sim: {:.3f}, num_TPs: {:d}'
              .format(precision * 100.0, sim_sum / correct, num_tps))

        return precision, num_tps


if __name__ == '__main__':
    matcher = FeatureMatcher()
    matcher.run(cls_id=0, img_w=1920, img_h=1080, viz_dir=None)  # '/mnt/diskc/even/viz_one_feat'

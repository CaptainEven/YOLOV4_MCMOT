from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.utils import *
from tracking_utils.log import logger

from tracker.basetrack import BaseTrack, MCBaseTrack, TrackState
from models import *


# TODO: Multi-class Track class
class MCTrack(MCBaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, num_classes, cls_id, buff_size=30):
        """
        :param tlwh:
        :param score:
        :param temp_feat:
        :param num_classes:
        :param cls_id:
        :param buff_size:
        """
        # object class id
        self.cls_id = cls_id

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.track_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)

        # buffered features
        self.features = deque([], maxlen=buff_size)

        # fusion factor
        self.alpha = 0.9

    def update_features(self, feat):
        # L2 normalizing
        feat /= np.linalg.norm(feat)

        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat

        self.features.append(feat)

        # L2 normalizing
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])

            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = Track.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

    def reset_track_id(self):
        self.reset_track_count(self.cls_id)

    def activate(self, kalman_filter, frame_id):
        """Start a new track"""
        self.kalman_filter = kalman_filter  # assign a filter to each track?

        # update track id for the object class
        self.track_id = self.next_id(self.cls_id)

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.track_len = 0
        self.state = TrackState.Tracked  # set flag 'tracked'

        # self.is_activated = True
        if frame_id == 1:  # to record the first frame's detection result
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # kalman update
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        # feature vector update
        self.update_features(new_track.curr_feat)

        self.track_len = 0
        self.frame_id = frame_id

        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True

        if new_id:  # update track id for the object class
            self.track_id = self.next_id(self.cls_id)

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True  # set flag 'activated'

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()  # numpy中的.copy()是深拷贝
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_({}-{})_({}-{})'.format(self.cls_id, self.track_id, self.start_frame, self.end_frame)


class Track(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buff_size=30):
        """
        :param tlwh:
        :param score:
        :param temp_feat:
        :param buff_size:
        """

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.track_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buff_size)  # 指定了限制长度
        self.alpha = 0.9

    def update_features(self, feat):
        # L2 normalizing
        feat /= np.linalg.norm(feat)

        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat

        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])

            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = Track.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

    def reset_track_id(self):
        self.reset_track_count()

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter  # assign a filter to each tracklet?

        # update the track id
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.track_len = 0
        self.state = TrackState.Tracked  # set flag 'tracked'

        # self.is_activated = True
        if frame_id == 1:  # to record the first frame's detection result
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        self.update_features(new_track.curr_feat)
        self.track_len = 0
        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True
        self.frame_id = frame_id

        if new_id:  # update the track id
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True  # set flag 'activated'

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()  # numpy中的.copy()是深拷贝
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


# Multi-class JDETracker
from train import max_id_dict


class MCJDETracker(object):
    def __init__(self, opt):
        self.opt = opt

        # ---------- Init model
        # max_ids_dict = {
        #     0: 341,  # car
        #     1: 103,  # bicycle
        #     2: 104,  # person
        #     3: 329,  # cyclist
        #     4: 48  # tricycle
        # }
        # max_id_dict = {
        #     0: 330,
        #     1: 102,
        #     2: 104,
        #     3: 312,
        #     4: 53
        # }  # previous version

        ## read from .npy(max_id_dict.npy file)
        max_id_dict_file_path = '/mnt/diskb/even/dataset/MCMOT/max_id_dict.npz'
        if os.path.isfile(max_id_dict_file_path):
            load_dict = np.load(max_id_dict_file_path, allow_pickle=True)
        max_id_dict = load_dict['max_id_dict'][()]

        print(max_id_dict)

        # set device
        device = opt.device

        # model in track mode(do detection and reid feature vector extraction)
        self.model = Darknet(cfg=opt.cfg,
                             img_size=opt.img_size,
                             verbose=False,
                             max_id_dict=max_id_dict,
                             emb_dim=128,
                             mode='track').to(device)
        # print(self.model)

        # Load checkpoint
        if opt.weights.endswith('.pt'):  # pytorch format
            ckpt = torch.load(opt.weights, map_location=device)
            self.model.load_state_dict(ckpt['model'])
            if 'epoch' in ckpt.keys():
                print('Checkpoint of epoch {} loaded.\n'.format(ckpt['epoch']))
        else:  # darknet format
            load_darknet_weights(self.model, opt.weights)

        # # ----- for debugging...
        # w_f_out_path = '/mnt/diskb/even/w_pt.txt'
        # with open(w_f_out_path, 'w', encoding='utf-8') as f:
        #     for i, (name, child) in enumerate(self.model.module_list.named_children()):
        #         for j, param in enumerate(child.parameters()):
        #             if len(param.shape) == 4:
        #                 f.write('Layer_{:s}_{:s}, shape: {:d}×{:d}×{:d}×{:d}\n'
        #                         .format(name, str(j),
        #                                 param.shape[0], param.shape[1], param.shape[2], param.shape[3]))
        #                 if param.numel() > 64:
        #                     items = param.view(1, -1)[0, :64].squeeze()
        #                 else:
        #                     items = param.view(1, -1).squeeze()
        #             elif len(param.shape) == 2:
        #                 f.write('Layer_{:s}_{:s}, shape: {:d}×{:d}\n'
        #                         .format(name, str(j), param.shape[0], param.shape[1]))
        #                 if param.numel() > 64:
        #                     items = param.view(1, -1)[0, :64].squeeze()
        #                 else:
        #                     items = param.view(1, -1).squeeze()
        #             elif len(param.shape) == 1:
        #                 f.write('Layer_{:s}_{:s}, shape: {:d}\n'
        #                         .format(name, str(j), param.shape[0]))
        #                 if param.numel() > 64:
        #                     items = param.view(1, -1)[0, :64].squeeze()
        #                 else:
        #                     items = param.view(1, -1).squeeze()
        #             else:
        #                 print(param.shape)
        #
        #             # items = param.view(1, -1).squeeze()
        #             for k, item in enumerate(items):
        #                 if k != 0 and k % 8 == 0:
        #                     f.write('\n')
        #                 f.write('{:.6f} '.format(item))
        #             f.write('\n\n')
        #         f.write('\n\n\n')
        # # -----

        # Put model to device and set eval mode
        self.model.to(device).eval()
        # ----------

        # Define tracks dict
        self.tracked_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.lost_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.removed_tracks_dict = defaultdict(list)  # value type: list[Track]

        # init frame index
        self.frame_id = 0

        # init hyp
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(opt.track_buffer)
        self.max_time_lost = self.buffer_size
        # self.mean = np.array([0.408, 0.447, 0.470]).reshape(1, 1, 3)
        # self.std = np.array([0.289, 0.274, 0.278]).reshape(1, 1, 3)

        # init kalman filter(to stabilize tracking)
        self.kalman_filter = KalmanFilter()

    def reset(self):
        """
        :return:
        """
        # Reset tracks dict
        self.tracked_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.lost_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.removed_tracks_dict = defaultdict(list)  # value type: list[Track]

        # Reset frame id
        self.frame_id = 0

        # Reset kalman filter to stabilize tracking
        self.kalman_filter = KalmanFilter()

    def update_detection(self, img, img0):
        """
        :param img:
        :param img0:
        :return:
        """
        # ----- do detection only(reid feature vector will not be extracted)
        # only get aggregated result, not original YOLO output
        net_h, net_w = img.shape[2:]
        orig_h, orig_w, _ = img0.shape  # H×W×C

        with torch.no_grad():
            pred, pred_orig, _, _ = self.model.forward(img, augment=self.opt.augment)
            pred = pred.float()

            # apply NMS
            pred = non_max_suppression(pred,
                                       self.opt.conf_thres,
                                       self.opt.iou_thres,
                                       merge=False,
                                       classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            # print(pred)

            dets = pred[0]  # assume batch_size == 1 here

            # get reid feature for each object class
            if dets is None:
                print('[Warning]: no objects detected.')
                return None

            # ----- Rescale boxes from img_size to img0 size(from net input size to original size)
            # dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], img0.shape).round()
            dets = map_to_orig_coords(dets, net_w, net_h, orig_w, orig_h)

        return dets

    def update_tracking(self, img, img0):
        """
        Update tracking result of the frame
        :param img:
        :param img0:
        :return:
        """
        # update frame id
        self.frame_id += 1

        # ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            MCTrack.init_count(self.opt.num_classes)
        # -----

        # Get image size
        net_h, net_w = img.shape[2:]
        orig_h, orig_w, _ = img0.shape  # H×W×C

        # record tracking states of the current frame
        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        # ----- do detection and reid feature extraction
        # only get aggregated result, not original YOLO output
        with torch.no_grad():
            # t1 = torch_utils.time_synchronized()

            pred, pred_orig, reid_feat_out, yolo_ids = self.model.forward(img, augment=self.opt.augment)
            pred = pred.float()

            # L2 normalize feature map
            reid_feat_out[0] = F.normalize(reid_feat_out[0], dim=1)

            # apply NMS
            pred, pred_yolo_ids = non_max_suppression_with_yolo_inds(pred,
                                                                     yolo_ids,
                                                                     self.opt.conf_thres,
                                                                     self.opt.iou_thres,
                                                                     merge=False,
                                                                     classes=self.opt.classes,
                                                                     agnostic=self.opt.agnostic_nms)

            dets = pred[0]  # assume batch_size == 1 here
            dets_yolo_ids = pred_yolo_ids[0]  # # assume batch_size == 1 here

            # t2 = torch_utils.time_synchronized()
            # print('run time (%.3fs)' % (t2 - t1))

            # get reid feature for each object class
            if dets is None or dets_yolo_ids is None:
                print('[Warning]: no objects detected.')
                return None

            # Get reid feature vector for each detection
            b, c, h, w = img.shape  # net input img size
            id_vects_dict = defaultdict(list)
            for det, yolo_id in zip(dets, dets_yolo_ids):
                x1, y1, x2, y2, conf, cls_id = det

                ## ----- show box size and yolo index
                # print('box area {:.3f}, yolo {:d}'.format((y2-y1) * (x2-x1), int(yolo_id)))

                # get reid map for this bbox(corresponding yolo idx)
                reid_feat_map = reid_feat_out[yolo_id]

                # L2 normalize the feature map
                reid_feat_map = F.normalize(reid_feat_map, dim=1)

                b, reid_dim, h_id_map, w_id_map = reid_feat_map.shape
                assert b == 1  # make sure batch size is 1

                # map center point from net scale to feature map scale(1/4 of net input size)
                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5
                center_x *= float(w_id_map) / float(w)
                center_y *= float(h_id_map) / float(h)

                # convert to int64 for indexing
                center_x += 0.5  # round
                center_y += 0.5
                center_x = center_x.long()
                center_y = center_y.long()
                center_x.clamp_(0, w_id_map - 1)  # to avoid the object center out of reid feature map's range
                center_y.clamp_(0, h_id_map - 1)

                # get reid feature vector and put into a dict
                id_feat_vect = reid_feat_map[0, :, center_y, center_x]
                id_feat_vect = id_feat_vect.squeeze()
                id_feat_vect = id_feat_vect.cpu().numpy()
                id_vects_dict[int(cls_id)].append(id_feat_vect)  # put feat vect to dict(key: cls_id)

            # Rescale boxes from img_size to img0 size(from net input size to original size)
            # dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], img0.shape).round()
            dets = map_to_orig_coords(dets, net_w, net_h, orig_w, orig_h)

        # Process each object class
        for cls_id in range(self.opt.num_classes):
            cls_inds = torch.where(dets[:, -1] == cls_id)
            cls_dets = dets[cls_inds]  # n_objs × 6
            cls_id_feature = id_vects_dict[cls_id]  # n_objs × 128

            cls_dets = cls_dets.detach().cpu().numpy()
            cls_id_feature = np.array(cls_id_feature)

            if len(cls_dets) > 0:
                '''Detections, tlbrs: top left bottom right score'''
                cls_detections = [
                    MCTrack(MCTrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, self.opt.num_classes, cls_id, 30)
                    for (tlbrs, feat) in
                    zip(cls_dets[:, :5], cls_id_feature)]  # detection of current frame
            else:
                cls_detections = []

            ''' Add newly detected tracks(current frame) to tracked_tracks'''
            unconfirmed_dict = defaultdict(list)
            tracked_tracks_dict = defaultdict(list)
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)  # record unconfirmed tracks in this frame
                else:
                    tracked_tracks_dict[cls_id].append(track)  # record tracked tracks of this frame

            ''' Step 2: First association, with embedding'''
            # build track pool for the current frame by joining tracked_tracks and lost tracks
            track_pool_dict = defaultdict(list)
            track_pool_dict[cls_id] = join_tracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            # Predict the current location with KF
            # for track in track_pool:

            # kalman prediction for track_pool
            Track.multi_predict(track_pool_dict[cls_id])

            dists = matching.embedding_distance(track_pool_dict[cls_id], cls_detections)
            dists = matching.fuse_motion(self.kalman_filter, dists, track_pool_dict[cls_id], cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)  # thresh=0.7
            for i_tracked, i_det in matches:  # process matched pairs between track pool and current frame detection
                track = track_pool_dict[cls_id][i_tracked]
                det = cls_detections[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:  # re-activate the lost track
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            ''' Step 3: Second association, with IOU'''
            # match between track pool and unmatched detection in current frame
            cls_detections = [cls_detections[i] for i in u_detection]
            r_tracked_tracks = [track_pool_dict[cls_id][i]
                                for i in u_track if track_pool_dict[cls_id][i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_tracks, cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)  # thresh=0.5
            for i_tracked, i_det in matches:  # process matched tracks
                track = r_tracked_tracks[i_tracked]
                det = cls_detections[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)
            for it in u_track:  # process unmatched tracks for two rounds
                track = r_tracked_tracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()  # mark unmatched track as lost track
                    lost_tracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            cls_detections = [cls_detections[i] for i in u_detection]  # current frame's unmatched detection
            dists = matching.iou_distance(unconfirmed_dict[cls_id], cls_detections)  # iou matching
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)  # thresh=0.7
            for i_tracked, i_det in matches:
                unconfirmed_dict[cls_id][i_tracked].update(cls_detections[i_det], self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])
            for it in u_unconfirmed:
                track = unconfirmed_dict[cls_id][it]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)

            """ Step 4: Init new tracks"""
            for i_new in u_detection:  # current frame's unmatched detection
                track = cls_detections[i_new]
                if track.score < self.det_thresh:
                    continue

                # tracked but not activated
                track.activate(self.kalman_filter, self.frame_id)  # Note: activate do not set 'is_activated' to be True
                activated_tracks_dict[cls_id].append(
                    track)  # activated_tarcks_dict may contain track with 'is_activated' False

            """ Step 5: Update state"""
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)
            # print('Remained match {} s'.format(t4-t3))

            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])  # add activated track
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           refined_tracks_dict[cls_id])  # add refined track
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.tracked_tracks_dict[cls_id])  # update lost tracks
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

            # logger.debug('===========Frame {}=========='.format(self.frame_id))
            # logger.debug('Activated: {}'.format(
            #     [track.track_id for track in activated_tracks_dict[cls_id]]))
            # logger.debug('Refined: {}'.format(
            #     [track.track_id for track in refined_tracks_dict[cls_id]]))
            # logger.debug('Lost: {}'.format(
            #     [track.track_id for track in lost_tracks_dict[cls_id]]))
            # logger.debug('Removed: {}'.format(
            #     [track.track_id for track in removed_tracks_dict[cls_id]]))

        return output_tracks_dict


class JDETracker(object):
    def __init__(self, opt):
        self.opt = opt

        # Init model
        max_ids_dict = {
            0: 330,
            1: 102,
            2: 104,
            3: 312,
            4: 53
        }  # cls_id -> track id number for traning
        device = opt.device

        # model in track mode(do detection and reid feature vector extraction)
        self.model = Darknet(opt.cfg, opt.net_w, False, max_ids_dict, 128, 'track').to(device)

        # Load checkpoint
        if opt.weights.endswith('.pt'):  # pytorch format
            ckpt = torch.load(opt.weights, map_location=device)
            self.model.load_state_dict(ckpt['model'])
            if 'epoch' in ckpt.keys():
                print('Checkpoint of epoch {} loaded.'.format(ckpt['epoch']))
        else:  # darknet format
            load_darknet_weights(self.model, opt.weights)

        # Put model to device and set eval mode
        self.model.to(device).eval()

        # Define tracks dict
        self.tracked_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.lost_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.removed_tracks_dict = defaultdict(list)  # value type: list[Track]

        self.frame_id = 0

        self.det_thresh = opt.conf_thres
        self.buffer_size = int(opt.track_buffer)
        self.max_time_lost = self.buffer_size
        # self.mean = np.array([0.408, 0.447, 0.470]).reshape(1, 1, 3)
        # self.std = np.array([0.289, 0.274, 0.278]).reshape(1, 1, 3)

        # ----- using kalman filter to stabilize tracking
        self.kalman_filter = KalmanFilter()

    def reset(self):
        """
        :return:
        """
        # Reset tracks dict
        self.tracked_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.lost_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.removed_tracks_dict = defaultdict(list)  # value type: list[Track]

        # Reset frame id
        self.frame_id = 0

        # Reset kalman filter to stabilize tracking
        self.kalman_filter = KalmanFilter()

    def update_detection(self, img, img0):
        """
        :param img:
        :param img0:
        :return:
        """
        # ----- do detection only(reid feature vector will not be extracted)
        # only get aggregated result, not original YOLO output
        with torch.no_grad():
            pred, pred_orig, _, _ = self.model.forward(img, augment=self.opt.augment)
            pred = pred.float()

            # apply NMS
            pred = non_max_suppression(pred,
                                       self.opt.conf_thres,
                                       self.opt.iou_thres,
                                       merge=False,
                                       classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)

            dets = pred[0]  # assume batch_size == 1 here

            # get reid feature for each object class
            if dets is None:
                print('[Warning]: no objects detected.')
                return None

            # Rescale boxes from img_size to img0 size(from net input size to original size)
            dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], img0.shape).round()

        return dets

    def update_tracking(self, img, img0):
        """
        Update tracking result of the frame
        :param img:
        :param img0:
        :return:
        """
        # update frame id
        self.frame_id += 1

        # record tracking states
        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        # ----- do detection and reid feature extraction
        # only get aggregated result, not original YOLO output
        with torch.no_grad():
            # t1 = torch_utils.time_synchronized()

            pred, pred_orig, reid_feat_out, yolo_ids = self.model.forward(img, augment=self.opt.augment)
            pred = pred.float()

            # L2 normalize feature map
            reid_feat_out[0] = F.normalize(reid_feat_out[0], dim=1)

            # apply NMS
            pred, pred_yolo_ids = non_max_suppression_with_yolo_inds(pred,
                                                                     yolo_ids,
                                                                     self.opt.conf_thres,
                                                                     self.opt.iou_thres,
                                                                     merge=False,
                                                                     classes=self.opt.classes,
                                                                     agnostic=self.opt.agnostic_nms)

            dets = pred[0]  # assume batch_size == 1 here
            dets_yolo_ids = pred_yolo_ids[0].squeeze()

            # t2 = torch_utils.time_synchronized()
            # print('run time (%.3fs)' % (t2 - t1))

            # get reid feature for each object class
            if dets is None:
                print('[Warning]: no objects detected.')
                return None

            # Get reid feature vector for each detection
            b, c, h, w = img.shape  # net input img size
            id_vects_dict = defaultdict(list)
            for det, yolo_id in zip(dets, dets_yolo_ids):
                x1, y1, x2, y2, conf, cls_id = det

                # print('box area {:.3f}, yolo {:d}'.format((y2-y1) * (x2-x1), int(yolo_id)))

                # get reid map for this bbox(corresponding yolo idx)
                reid_feat_map = reid_feat_out[yolo_id]

                b, reid_dim, h_id_map, w_id_map = reid_feat_map.shape
                assert b == 1  # make sure batch size is 1

                # map center point from net scale to feature map scale(1/4 of net input size)
                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5
                center_x *= float(w_id_map) / float(w)
                center_y *= float(h_id_map) / float(h)

                # convert to int64 for indexing
                center_x += 0.5  # round
                center_y += 0.5
                center_x = center_x.long()
                center_y = center_y.long()
                center_x.clamp_(0, w_id_map - 1)  # avoid out of reid feature map's range
                center_y.clamp_(0, h_id_map - 1)

                id_feat_vect = reid_feat_map[0, :, center_y, center_x]
                id_feat_vect = id_feat_vect.squeeze()

                # L2 normalize the feature vector
                id_feat_vect = F.normalize(id_feat_vect, dim=0)

                id_feat_vect = id_feat_vect.cpu().numpy()
                id_vects_dict[int(cls_id)].append(id_feat_vect)  # put feat vect to dict(key: cls_id)

            # Rescale boxes from img_size to img0 size(from net input size to original size)
            dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], img0.shape).round()

        # Process each object class
        for cls_id in range(self.opt.num_classes):
            cls_inds = torch.where(dets[:, -1] == cls_id)
            cls_dets = dets[cls_inds]  # n_objs × 6
            cls_id_feature = id_vects_dict[cls_id]  # n_objs × 128

            cls_dets = cls_dets.detach().cpu().numpy()
            cls_id_feature = np.array(cls_id_feature)

            if len(cls_dets) > 0:
                '''Detections, tlbrs: top left bottom right score'''
                cls_detections = [Track(Track.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, buff_size=30)
                                  for (tlbrs, feat) in
                                  zip(cls_dets[:, :5], cls_id_feature)]  # detection of current frame
            else:
                cls_detections = []

            # reset the track ids for a different object class in the first frame
            if self.frame_id == 1:
                for track in cls_detections:
                    track.reset_track_id()

            ''' Add newly detected tracks(current frame) to tracked_tracks'''
            unconfirmed_dict = defaultdict(list)
            tracked_tracks_dict = defaultdict(list)
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)  # record unconfirmed tracks in this frame
                else:
                    tracked_tracks_dict[cls_id].append(track)  # record tracked tracks of this frame

            ''' Step 2: First association, with embedding'''
            # build track pool for the current frame by joining tracked_tracks and lost tracks
            track_pool_dict = defaultdict(list)
            track_pool_dict[cls_id] = join_tracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            # Predict the current location with KF
            # for track in track_pool:

            # kalman predict for track_pool
            Track.multi_predict(track_pool_dict[cls_id])

            dists = matching.embedding_distance(track_pool_dict[cls_id], cls_detections)
            dists = matching.fuse_motion(self.kalman_filter, dists, track_pool_dict[cls_id], cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)  # thresh=0.7
            for i_tracked, i_det in matches:  # process matched pairs between track pool and current frame detection
                track = track_pool_dict[cls_id][i_tracked]
                det = cls_detections[i_det]
                if track.state == TrackState.Tracked:
                    track.update(cls_detections[i_det], self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:  # re-activate the lost track
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            ''' Step 3: Second association, with IOU'''
            # match between track pool and unmatched detection in current frame
            cls_detections = [cls_detections[i] for i in
                              u_detection]  # get un-matched detections for following iou matching
            r_tracked_tracks = [track_pool_dict[cls_id][i]
                                for i in u_track if track_pool_dict[cls_id][i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_tracks, cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)  # thresh=0.5
            for i_tracked, i_det in matches:  # process matched tracks
                track = r_tracked_tracks[i_tracked]
                det = cls_detections[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)
            for it in u_track:  # process unmatched tracks for two rounds
                track = r_tracked_tracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()  # mark unmatched track as lost track
                    lost_tracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            cls_detections = [cls_detections[i] for i in u_detection]  # current frame's unmatched detection
            dists = matching.iou_distance(unconfirmed_dict[cls_id], cls_detections)  # iou matching
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for i_tracked, i_det in matches:
                unconfirmed_dict[cls_id][i_tracked].update(cls_detections[i_det], self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])
            for it in u_unconfirmed:
                track = unconfirmed_dict[cls_id][it]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)

            """ Step 4: Init new tracks"""
            for i_new in u_detection:  # current frame's unmatched detection
                track = cls_detections[i_new]
                if track.score < self.det_thresh:
                    continue

                # tracked but not activated
                track.activate(self.kalman_filter, self.frame_id)  # Note: activate do not set 'is_activated' to be True

                # activated_tarcks_dict may contain track with 'is_activated' False
                activated_tracks_dict[cls_id].append(track)

            """ Step 5: Update state"""
            # update removed tracks
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)
            # print('Remained match {} s'.format(t4-t3))

            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])  # add activated track
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           refined_tracks_dict[cls_id])  # add refined track
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.tracked_tracks_dict[cls_id])  # update lost tracks
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

            # logger.debug('===========Frame {}=========='.format(self.frame_id))
            # logger.debug('Activated: {}'.format(
            #     [track.track_id for track in activated_tracks_dict[cls_id]]))
            # logger.debug('Refined: {}'.format(
            #     [track.track_id for track in refined_tracks_dict[cls_id]]))
            # logger.debug('Lost: {}'.format(
            #     [track.track_id for track in lost_tracks_dict[cls_id]]))
            # logger.debug('Removed: {}'.format(
            #     [track.track_id for track in removed_tracks_dict[cls_id]]))

        return output_tracks_dict


def join_tracks(tracks_a, tracks_b):
    """
    join two track lists
    :param tracks_a:
    :param tracks_b:
    :return:
    """
    exists = {}
    join_tr_list = []

    for t in tracks_a:
        exists[t.track_id] = 1
        join_tr_list.append(t)

    for t in tracks_b:
        tr_id = t.track_id
        if not exists.get(tr_id, 0):
            exists[tr_id] = 1
            join_tr_list.append(t)

    return join_tr_list


def sub_tracks(tracks_a, tracks_b):
    tracks = {}

    for t in tracks_a:
        tracks[t.track_id] = t
    for t in tracks_b:
        tr_id = t.track_id
        if tracks.get(tr_id, 0):
            del tracks[tr_id]

    return list(tracks.values())


def remove_duplicate_tracks(tracks_a, tracks_b):
    p_dist = matching.iou_distance(tracks_a, tracks_b)
    pairs = np.where(p_dist < 0.15)
    dup_a, dup_b = list(), list()

    for a, b in zip(*pairs):
        time_a = tracks_a[a].frame_id - tracks_a[a].start_frame
        time_b = tracks_b[b].frame_id - tracks_b[b].start_frame
        if time_a > time_b:
            dup_b.append(b)  # choose short record time as duplicate
        else:
            dup_a.append(a)

    res_a = [t for i, t in enumerate(tracks_a) if not i in dup_a]
    res_b = [t for i, t in enumerate(tracks_b) if not i in dup_b]

    return res_a, res_b

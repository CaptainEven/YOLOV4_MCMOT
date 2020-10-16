from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.utils import *
from tracking_utils.log import logger

from .basetrack import BaseTrack, TrackState
from models import *


class STrack(BaseTrack):
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
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buff_size)  # 指定了限制长度
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * \
                               self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def reset_track_id(self):
        self.reset_track_count()

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter  # assign a filter to each tracklet?
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id

        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

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
        self.model = Darknet(opt.cfg, opt.img_size, False, max_ids_dict, 128, 'track').to(device)

        # Load checkpoint
        if opt.weights.endswith('.pt'):  # pytorch format
            ckpt = torch.load(opt.weights, map_location=device)
            self.model.load_state_dict(ckpt['model'])
            if 'epoch' in ckpt.keys():
                print('Checkpoint of epoch {} loaded.'.format(ckpt['epoch']))
        else:  # darknet format
            load_darknet_weights(self.model, opt.weights)

        # Set model to eval mode
        self.model.to(device).eval()

        # Define track_lets
        self.tracked_stracks_dict = defaultdict(list)  # value type: list[STrack]
        self.lost_stracks_dict = defaultdict(list)  # value type: list[STrack]
        self.removed_stracks_dict = defaultdict(list)  # value type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.mean = np.array([0.408, 0.447, 0.470]).reshape(1, 1, 3)
        self.std = np.array([0.289, 0.274, 0.278]).reshape(1, 1, 3)

        # ----- using kalman filter to stabilize tracking
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
        activated_starcks_dict = defaultdict(list)
        refind_stracks_dict = defaultdict(list)
        lost_stracks_dict = defaultdict(list)
        removed_stracks_dict = defaultdict(list)
        output_stracks_dict = defaultdict(list)

        # ----- do detection and reid feature extraction
        # only get aggregated result, not original YOLO output
        with torch.no_grad():
            t1 = torch_utils.time_synchronized()

            pred, pred_orig, reid_feat_out, anchor_inds = self.model.forward(img, augment=self.opt.augment)
            pred = pred.float()

            # L2 normalize feature map
            reid_feat_out[0] = F.normalize(reid_feat_out[0], dim=1)

            # apply NMS
            pred, pred_anchor_inds = non_max_suppression_with_anchor_inds(pred,
                                                                          anchor_inds,
                                                                          self.opt.conf_thres,
                                                                          self.opt.iou_thres,
                                                                          merge=False,
                                                                          classes=self.opt.classes,
                                                                          agnostic=self.opt.agnostic_nms)

            dets = pred[0]  # assume batch_size == 1 here
            dets_anchor_ids = pred_anchor_inds[0].squeeze()

            t2 = torch_utils.time_synchronized()
            print('run time (%.3fs)' % (t2 - t1))

            # get reid feature for each object class
            if dets is None:
                print('[Warning]: no objects detected.')
                return None

            # Get reid feature vector for each detection
            b, c, h, w = img.shape  # net input img size
            id_vects_dict = defaultdict(list)
            for det, anchor_id in zip(dets, dets_anchor_ids):
                x1, y1, x2, y2, conf, cls_id = det

                # print('box area {:.3f}, yolo {:d}'.format((y2-y1) * (x2-x1), int(anchor_id)))

                reid_feat_map = reid_feat_out[anchor_id]
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
                cls_detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, buff_size=30)
                                  for (tlbrs, feat) in zip(cls_dets[:, :5], cls_id_feature)]
            else:
                cls_detections = []

            # reset the track ids for a different object class in the first frame
            if self.frame_id == 0:
                for track in cls_detections:
                    track.reset_track_id()

            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed_dict = defaultdict(list)
            tracked_stracks_dict = defaultdict(list)
            for track in self.tracked_stracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)
                else:
                    tracked_stracks_dict[cls_id].append(track)

            ''' Step 2: First association, with embedding'''
            strack_pool_dict = defaultdict(list)
            strack_pool_dict[cls_id] = joint_stracks(tracked_stracks_dict[cls_id], self.lost_stracks_dict[cls_id])

            # Predict the current location with KF
            # for strack in strack_pool:
            STrack.multi_predict(strack_pool_dict[cls_id])
            dists = matching.embedding_distance(strack_pool_dict[cls_id], cls_detections)
            dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool_dict[cls_id], cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)  # thresh=0.7
            for i_tracked, i_det in matches:
                track = strack_pool_dict[cls_id][i_tracked]
                det = cls_detections[i_det]
                if track.state == TrackState.Tracked:
                    track.update(cls_detections[i_det], self.frame_id)
                    activated_starcks_dict[cls_id].append(track)  # for multi-class
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks_dict[cls_id].append(track)

            ''' Step 3: Second association, with IOU'''
            cls_detections = [cls_detections[i] for i in u_detection]
            r_tracked_stracks = [strack_pool_dict[cls_id][i]
                                 for i in u_track if strack_pool_dict[cls_id][i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_stracks, cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)  # thresh=0.5
            for i_tracked, i_det in matches:
                track = r_tracked_stracks[i_tracked]
                det = cls_detections[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks_dict[cls_id].append(track)
            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            cls_detections = [cls_detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed_dict[cls_id], cls_detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for i_tracked, i_det in matches:
                unconfirmed_dict[cls_id][i_tracked].update(cls_detections[i_det], self.frame_id)
                activated_starcks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])
            for it in u_unconfirmed:
                track = unconfirmed_dict[cls_id][it]
                track.mark_removed()
                removed_stracks_dict[cls_id].append(track)

            """ Step 4: Init new stracks"""
            for i_new in u_detection:
                track = cls_detections[i_new]
                if track.score < self.det_thresh:
                    continue
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks_dict[cls_id].append(track)

            """ Step 5: Update state"""
            for track in self.lost_stracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks_dict[cls_id].append(track)
            # print('Ramained match {} s'.format(t4-t3))

            self.tracked_stracks_dict[cls_id] = [t for t in self.tracked_stracks_dict[cls_id] if
                                                 t.state == TrackState.Tracked]
            self.tracked_stracks_dict[cls_id] = joint_stracks(self.tracked_stracks_dict[cls_id],
                                                              activated_starcks_dict[cls_id])
            self.tracked_stracks_dict[cls_id] = joint_stracks(self.tracked_stracks_dict[cls_id],
                                                              refind_stracks_dict[cls_id])
            self.lost_stracks_dict[cls_id] = sub_stracks(self.lost_stracks_dict[cls_id],
                                                         self.tracked_stracks_dict[cls_id])
            self.lost_stracks_dict[cls_id].extend(lost_stracks_dict[cls_id])
            self.lost_stracks_dict[cls_id] = sub_stracks(self.lost_stracks_dict[cls_id],
                                                         self.removed_stracks_dict[cls_id])
            self.removed_stracks_dict[cls_id].extend(removed_stracks_dict[cls_id])
            self.tracked_stracks_dict[cls_id], self.lost_stracks_dict[cls_id] = remove_duplicate_stracks(
                self.tracked_stracks_dict[cls_id],
                self.lost_stracks_dict[cls_id])

            # get scores of lost tracks
            output_stracks_dict[cls_id] = [track for track in self.tracked_stracks_dict[cls_id] if
                                           track.is_activated]

            # logger.debug('===========Frame {}=========='.format(self.frame_id))
            # logger.debug('Activated: {}'.format(
            #     [track.track_id for track in activated_starcks_dict[cls_id]]))
            # logger.debug('Refind: {}'.format(
            #     [track.track_id for track in refind_stracks_dict[cls_id]]))
            # logger.debug('Lost: {}'.format(
            #     [track.track_id for track in lost_stracks_dict[cls_id]]))
            # logger.debug('Removed: {}'.format(
            #     [track.track_id for track in removed_stracks_dict[cls_id]]))

        return output_stracks_dict


def joint_stracks(t_list_a, t_list_b):
    """
    join two track lists
    :param t_list_a:
    :param t_list_b:
    :return:
    """
    exists = {}
    res = []
    for t in t_list_a:
        exists[t.track_id] = 1
        res.append(t)
    for t in t_list_b:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()

    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]

    return resa, resb

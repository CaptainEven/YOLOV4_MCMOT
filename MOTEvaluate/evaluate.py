"""
2D MOT2016 Evaluation Toolkit
An python reimplementation of toolkit in
2DMOT16(https://motchallenge.net/data/MOT16/)

This file executes the evaluation.

usage:
python evaluate.py
    --bm                       Whether to evaluate multiple files(benchmarks)
    --seqmap [filename]        List of sequences to be evaluated
    --track  [dirname]         Tracking results directory: default path --
                               [dirname]/[seqname]/res.txt
    --gt     [dirname]         Groundtruth directory:      default path --
                               [dirname]/[seqname]/gt.txt
(C) Yiwen Liu(765305261@qq.com), 2020-10
"""
import os
import copy
import numpy as np
import argparse
# from sklearn.evaluate_utils.linear_assignment_ import linear_assignment
from collections import defaultdict
from scipy.optimize import linear_sum_assignment as linear_assignment
from easydict import EasyDict as edict
from MOTEvaluate.evaluate_utils.io import read_txt_to_struct, read_seqmaps, \
    extract_valid_gt_data, print_metrics
from MOTEvaluate.evaluate_utils.bbox import bbox_overlap
from MOTEvaluate.evaluate_utils.convert import cls2id, id2cls
from MOTEvaluate.evaluate_utils.measurements import clear_mot_metrics, id_measures


def filter_DB(trackDB, gtDB, distractor_ids, iou_thres, min_vis):
    """
    Preprocess the computed trajectory data.
    Matching computed boxes to ground-truth to remove distractors
    and low visibility data in both trackDB and gtDB
    trackDB: [npoints, 9] computed trajectory data
    gtDB: [npoints, 9] computed trajectory data
    distractor_ids: identities of distractors of the sequence
    iou_thres: bounding box overlap threshold
    min_vis: minimum visibility of groundtruth boxes, default set to zero
    because the occluded people are supposed to be interpolated for tracking.
    """

    # Get frames number for the seq: make sure track_frames is the same as gt frames
    track_frames = np.unique(trackDB[:, 0])
    gt_frames = np.unique(gtDB[:, 0])
    n_frames = min(len(track_frames), len(gt_frames))

    # keeping results: init to 1
    res_keep = np.ones((trackDB.shape[0],), dtype=float)  # number of res bbox

    for i in range(1, n_frames + 1):
        # find all data(one bbox correspond to one item of the data) in this frame
        res_in_frame = np.where(trackDB[:, 0] == i)[0]
        res_in_frame_data = trackDB[res_in_frame, :]
        gt_in_frame = np.where(gtDB[:, 0] == i)[0]
        gt_in_frame_data = gtDB[gt_in_frame, :]

        # ---------- IOU matching of res and gt bbox
        # get overlaps of bbox
        res_num = res_in_frame.shape[0]
        gt_num = gt_in_frame.shape[0]
        overlaps = np.zeros((res_num, gt_num), dtype=float)  # iou matrix
        for gt_id in range(gt_num):  # row: res, col: gt
            overlaps[:, gt_id] = bbox_overlap(
                res_in_frame_data[:, 2:6], gt_in_frame_data[gt_id, 2:6])

        # build cost matrix
        cost_matrix = 1.0 - overlaps

        # hungarian matching: return row_ind(res), col_ind(gt)
        matched_indices = linear_assignment(cost_matrix=cost_matrix)

        for matched in zip(*matched_indices):  # row_ind, col_ind
            # overlap lower than threshold, discard the pair
            if overlaps[matched[0], matched[1]] < iou_thres:
                continue

            # matched to distractors, discard the result box
            if distractor_ids is not None:
                if gt_in_frame_data[matched[1], 1] in distractor_ids:
                    res_keep[res_in_frame[matched[0]]] = 0

            # matched to a partial
            if gt_in_frame_data[matched[1], 8] < min_vis:
                res_keep[res_in_frame[matched[0]]] = 0

        # sanity check
        frame_id_pairs = res_in_frame_data[:, :2]  # pair: frame-id
        uniq_frame_id_pairs = np.unique(frame_id_pairs)
        has_duplicates = uniq_frame_id_pairs.shape[0] < frame_id_pairs.shape[0]
        assert not has_duplicates, \
            'Duplicate ID in same frame [Frame ID: {}].'.format(i)

    # filter res data
    keep_idx = np.where(res_keep == 1)[0]
    print('[TRACK PREPROCESSING]: remove distractors and low visibility boxes,'
          'remaining {}/{} computed boxes'.format(
        len(keep_idx), len(res_keep)))

    trackDB = trackDB[keep_idx, :]

    if distractor_ids is not None:
        print('Distractor IDs: {}'.format(
            ', '.join(list(map(str, distractor_ids.astype(int))))))

    if distractor_ids is not None:
        # filter gt data
        keep_idx = np.array([i for i in range(gtDB.shape[0]) if gtDB[i, 1] not in distractor_ids
                             and gtDB[i, 8] >= min_vis])  # distracor ids visibility ratio thresholding
    else:
        # distracor ids visibility ratio thresholding
        keep_idx = np.array(
            [i for i in range(gtDB.shape[0]) if gtDB[i, 8] >= min_vis])

    # keep_idx = np.array([i for i in range(gtDB.shape[0]) if gtDB[i, 6] != 0])
    print('[GT PREPROCESSING]: Removing distractor boxes, '
          'remaining {}/{} boxes'.format(len(keep_idx), gtDB.shape[0]))
    try:
        gtDB = gtDB[keep_idx, :]
    except Exception as e:
        print(e)

    return trackDB, gtDB


def evaluate_seq(resDB, gtDB, distractor_ids, iou_thresh=0.5, min_vis=0):
    """
    Evaluate single sequence
    trackDB: tracking result data structure
    gtDB: ground-truth data structure
    iou_thres: bounding box overlap threshold
    min_vis: minimum tolerent visibility
    """
    # filter out invalid items from the data
    resDB, gtDB = filter_DB(resDB, gtDB, distractor_ids, iou_thresh, min_vis)

    # ----- calculate all kinds of metrics
    # mme: mis-match error
    # tp: true positive
    # fp: false positive
    # gt_cnt: ground truth
    # fn: false negative
    # d: iou(or 1-distance), key: gt_tracked_id
    # M: matched dict, key: gt_track_id, col: res_track_id
    # all_fps: all frames' false positive
    # mme, tp, fp, gt_counts, fn, d, MatchedDicts, all_fps
    mme, tp, fp, gt_cnt, fn, d, M, all_fps = clear_mot_metrics(resDB, gtDB, iou_thresh)
    # -----

    gt_frames = np.unique(gtDB[:, 0])

    gt_ids = np.unique(gtDB[:, 1])
    res_ids = np.unique(resDB[:, 1])

    n_frames_gt = len(gt_frames)
    n_ids_gt = len(gt_ids)
    n_ids_res = len(res_ids)

    FN = sum(fn)  # false negative
    FP = sum(fp)  # false positive
    IDS = sum(mme)

    # MOTP = sum(iou) / # corrected boxes
    MOTP = (sum(sum(d)) / sum(tp)) * 100.0

    # MOTAL = 1.0 - (# fp + # fn + #log10(ids)) / # gts
    MOTAL = (1.0 - (sum(fp) + sum(fn) +
                    np.log10(sum(mme) + 1)) / sum(gt_cnt)) * 100.0

    sum_fp = sum(fp)
    sum_fn = sum(fn)
    sum_mme = sum(mme)
    sum_g = sum(gt_cnt)
    MOTA = (1.0 - (sum_fp + sum_fn + sum_mme) / sum_g) * 100.0
    # MOTA = (1.0 - (sum(fp) + sum(fn) + sum(mme)) / sum(g)) * 100.0
    # if MOTA < 0.0:
    #     print('[Debug here].')

    # recall = TP / (TP + FN) = # corrected boxes / # gt boxes
    recall = sum(tp) / sum(gt_cnt) * 100.0

    # precision = TP / (TP + FP) = # corrected boxes / # det boxes
    precision = sum(tp) / (sum(fp) + sum(tp)) * \
                100.0  # true positive / all_positive

    # FAR = sum(fp) / # number_frames
    FAR = sum(fp) / n_frames_gt
    MT_stats = np.zeros((n_ids_gt,), dtype=float)  # what's this?

    for fr_i in range(n_ids_gt):
        inds_of_the_gt_id = np.where(gtDB[:, 1] == gt_ids[fr_i])[0]
        n_frs_total_gt_id = len(inds_of_the_gt_id)
        frs_contain_gt_id = gtDB[inds_of_the_gt_id, 0].astype(
            int)  # part frames of the gtDB

        gt_frames_list = list(gt_frames)
        n_frs_matched_gt_id = sum([1 if fr_i in M[gt_frames_list.index(
            fr)].keys() else 0 for fr in frs_contain_gt_id])
        ratio = float(n_frs_matched_gt_id) / n_frs_total_gt_id

        if ratio < 0.2:
            MT_stats[fr_i] = 1
        elif ratio >= 0.8:
            MT_stats[fr_i] = 3
        else:
            MT_stats[fr_i] = 2

    # statistics of stats
    ML = len(np.where(MT_stats == 1)[0])
    PT = len(np.where(MT_stats == 2)[0])
    MT = len(np.where(MT_stats == 3)[0])

    # fragment
    fr = np.zeros((n_ids_gt,), dtype=int)
    M_arr = np.zeros((n_frames_gt, n_ids_gt), dtype=int)  # what's this?

    for fr_i in range(n_frames_gt):
        for gt_id in M[fr_i].keys():
            res_id = M[fr_i][gt_id]
            M_arr[fr_i, gt_id] = res_id + 1  # why res_id + 1?

    for fr_i in range(n_ids_gt):
        occur = np.where(M_arr[:, fr_i] > 0)[0]
        occur = np.where(np.diff(occur) != 1)[0]
        fr[fr_i] = len(occur)

    FRA = sum(fr)

    # -----
    id_metrics = id_measures(gtDB, resDB, iou_thresh)
    # -----

    metrics = [id_metrics.IDF1,
               id_metrics.IDP,
               id_metrics.IDR,
               recall,
               precision,
               FAR,
               n_ids_gt,
               MT, PT, ML, FP, FN, IDS, FRA,
               MOTA, MOTP, MOTAL]

    extra_info = edict()
    extra_info.mme = sum(mme)
    extra_info.c = sum(tp)
    extra_info.fp = sum(fp)
    extra_info.g = sum(gt_cnt)
    extra_info.missed = sum(fn)
    extra_info.d = d

    # extra_info.m = M
    extra_info.f_gt = n_frames_gt
    extra_info.n_gt = n_ids_gt
    extra_info.n_st = n_ids_res

    #    extra_info.allfps = allfps
    extra_info.ML = ML
    extra_info.PT = PT
    extra_info.MT = MT
    extra_info.FRA = FRA
    extra_info.idmetrics = id_metrics

    return metrics, extra_info


def evaluate_bm(all_metrics):
    """
    Evaluate whole benchmark, summaries all metrics
    """
    f_gt, n_gt, n_st = 0, 0, 0
    nbox_gt, nbox_st = 0, 0
    c, g, fp, missed, ids = 0, 0, 0, 0, 0
    IDTP, IDFP, IDFN = 0, 0, 0
    MT, ML, PT, FRA = 0, 0, 0, 0
    overlap_sum = 0
    for i in range(len(all_metrics)):
        nbox_gt += all_metrics[i].idmetrics.nbox_gt
        nbox_st += all_metrics[i].idmetrics.nbox_st

        # Total ID Measures
        IDTP += all_metrics[i].idmetrics.IDTP
        IDFP += all_metrics[i].idmetrics.IDFP
        IDFN += all_metrics[i].idmetrics.IDFN

        # Total ID Measures
        MT += all_metrics[i].MT
        ML += all_metrics[i].ML
        PT += all_metrics[i].PT
        FRA += all_metrics[i].FRA
        f_gt += all_metrics[i].f_gt
        n_gt += all_metrics[i].n_gt
        n_st += all_metrics[i].n_st
        c += all_metrics[i].c
        g += all_metrics[i].g
        fp += all_metrics[i].fp
        missed += all_metrics[i].missed
        ids += all_metrics[i].mme
        overlap_sum += sum(sum(all_metrics[i].d))

    # IDP = IDTP / (IDTP + IDFP)
    IDP = IDTP / (IDTP + IDFP) * 100

    # IDR = IDTP / (IDTP + IDFN)
    IDR = IDTP / (IDTP + IDFN) * 100

    # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    IDF1 = 2 * IDTP / (nbox_gt + nbox_st) * 100
    FAR = fp / f_gt
    MOTP = (overlap_sum / c) * 100

    # MOTAL = 1 - (# fp + # fn + #log10(ids)) / # gts
    MOTAL = (1 - (fp + missed + np.log10(ids + 1)) / g) * 100

    # MOTA = 1 - (# fp + # fn + # ids) / # gts
    MOTA = (1 - (fp + missed + ids) / g) * 100

    # recall = TP / (TP + FN) = # corrected boxes / # gt boxes
    recall = c / g * 100

    # precision = TP / (TP + FP) = # corrected boxes / # det boxes
    precision = c / (fp + c) * 100
    metrics = [IDF1, IDP, IDR, recall, precision, FAR, n_gt,
               MT, PT, ML, fp, missed, ids, FRA, MOTA, MOTP, MOTAL]
    return metrics


metric_names = ['IDF1', 'IDP', 'IDR',
                'Rcll', 'Prcn', 'FAR',
                'GT', 'MT', 'PT', 'ML',
                'FP', 'FN', 'IDs', 'FM',
                'MOTA', 'MOTP', 'MOTAL']


def evaluate_mcmot_seq(seq_name, gt_path, res_path):
    """
    :param seq_name:
    :param gt_path:
    :param res_path:
    :return:
    """
    if not (os.path.isfile(gt_path) and os.path.isfile(gt_path)):
        print('[Err]: invalid file path.')
        return

    # metric_name2id = defaultdict(int)
    # metric_id2name = defaultdict(str)
    # for id, name in enumerate(metric_names):
    #     metric_id2name[id] = name
    #     metric_name2id[name] = id

    # read txt file
    trackDB = read_txt_to_struct(res_path)
    gtDB = read_txt_to_struct(gt_path)

    # compute for each object class
    metrics = np.zeros((len(id2cls.keys()), len(metric_names)), dtype=float)
    for cls_id in id2cls.keys():
        selected = np.where(cls_id == gtDB[:, 7])[0]
        cls_gtDB = gtDB[selected]
        print('gt: {:d} items for object class {:s}'.format(len(cls_gtDB), id2cls[cls_id]))
        if len(cls_gtDB) == 0:
            continue

        selected = np.where(cls_id == trackDB[:, 7])[0]
        cls_resDB = trackDB[selected]
        print('res: {:d} items for object class {:s}'.format(len(cls_resDB), id2cls[cls_id]))
        if len(cls_resDB) == 0:
            continue

        # ---------- main function to do evaluation
        cls_metrics, cls_extra_info = evaluate_seq(cls_resDB, cls_gtDB, distractor_ids=None)
        metrics[cls_id] = cls_metrics
        # ----------

        print_metrics('Seq {:s} evaluation for class {:s}'.format(seq_name, id2cls[cls_id]), cls_metrics)

    # ---------- mean of the metrics
    mean_metrics = metrics.mean(axis=0)  # mean value of each column
    # ----------

    return mean_metrics


def evaluate_seqs(seqs, track_dir, gt_dir):
    all_info = []
    for seq_name in seqs:  # process every seq
        track_res = os.path.join(track_dir, seq_name, 'res.txt')
        gt_file = os.path.join(gt_dir, seq_name, 'gt.txt')
        assert os.path.exists(track_res) and os.path.exists(gt_file), \
            'Either tracking result {} or ' \
            'groundtruth directory {} does not exist'.format(track_res, gt_file)

        trackDB = read_txt_to_struct(track_res)  # track result
        gtDB = read_txt_to_struct(gt_file)  # ground truth

        # filtering for specific class id
        gtDB, distractor_ids = extract_valid_gt_data(gtDB)

        # ---------- main function to do evaluation
        metrics, extra_info = evaluate_seq(trackDB, gtDB, distractor_ids)
        # ----------

        print_metrics(seq_name + ' Evaluation', metrics)
        all_info.append(extra_info)

    all_metrics = evaluate_bm(all_info)
    print_metrics('Summary Evaluation', all_metrics)


def parse_args():
    parser = argparse.ArgumentParser(description='MOT Evaluation Toolkit')
    parser.add_argument('--bm',
                        help='Evaluation multiple videos',
                        action='store_true')
    parser.add_argument('--seqmap',
                        type=str,
                        default='seqmaps/test.txt',
                        help='seqmap file')
    parser.add_argument('--track',
                        default='data/',
                        type=str,
                        help='Tracking result directory')
    parser.add_argument('--gt',
                        default='data',
                        type=str,
                        help='Ground-truth annotation directory')
    args = parser.parse_args()
    return args


def evaluate_mcmot_seqs(test_root, default_fps=12):
    """
    :param test_root:
    :param default_fps: fps for sampling
    :return:
    """
    if not os.path.isdir(test_root):
        print('[Err]: invalid test root.')
        return

    seq_names = [x for x in os.listdir(test_root) if x.endswith('.mp4')]
    if len(seq_names) == 0 or seq_names is None:
        print('[Err]: no test videos detected.')
        return

    metrics = np.zeros((len(seq_names), len(metric_names)), dtype=float)
    for i, seq_name in enumerate(seq_names):
        seq_name = seq_name[:-4]
        gt_path = test_root + '/' + seq_name + '_gt_mot16' + '_fps' + str(default_fps) + '.txt'
        res_path = test_root + '/' + seq_name + '_results_fps' + str(default_fps) + '.txt'

        if not (os.path.isfile(gt_path) and os.path.isfile(res_path)):
            print('[Warning]: {:s} test file not exists.'.format(seq_name))
            continue

        # ---------
        seq_mean_metrics = evaluate_mcmot_seq(seq_name, gt_path, res_path)
        print_metrics('Seq {:s} evaluation mean metrics: '.format(seq_name), seq_mean_metrics)
        # ---------

        metrics[i] = seq_mean_metrics

    mean_metrics = metrics.mean(axis=0)  # mean value of each column
    print_metrics('All test seq evaluation mean metrics: '.format(seq_name), mean_metrics)



if __name__ == '__main__':
    # # ----- command line running
    # args = parse_args()
    # seqs = read_seqmaps(args.seqmap)
    # print('Seqs: ', sequences)

    # evaluate_seqs(seqs, args.track, args.gt)

    # ----- test running
    # evaluate_mcmot_seq(gt_path='F:/val_seq/val_1_gt_mot16_fps12.txt',
    #                    res_path='F:/val_seq/val_1_results_fps12.txt')

    evaluate_mcmot_seqs(test_root='/mnt/diskb/even/dataset/MCMOT_Evaluate',
                        default_fps=12)

    print('Done.')

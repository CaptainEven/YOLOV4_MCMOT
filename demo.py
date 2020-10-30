# encoding=utf-8

import os
import threading
import torch
import argparse
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from tracker.multitracker import JDETracker, MCJDETracker
from tracking_utils import visualization as vis
from tracking_utils.io import write_results_dict


def format_output(dets, w, h):
    """
    :param dets: detection result input: x1, y1, x2, y2, score, cls_id
    :param w: image's original width
    :param h: image's original height
    :return: list of items: cls_id, conf_score, center_x, center_y,  bbox_w, bbox_h, [0, 1]
    """
    if dets is None:
        return None

    out_list = []
    for det in dets:
        x1, y1, x2, y2, score, cls_id = det
        center_x = (x1 + x2) * 0.5 / float(w)
        center_y = (y1 + y2) * 0.5 / float(h)
        bbox_w = (x2 - x1) / float(w)
        bbox_h = (y2 - y1) / float(h)
        out_list.append([int(cls_id), score, center_x, center_y, bbox_w, bbox_h])
    return out_list


def run_detection(opt):
    """
    :param opt:
    :return:
    """
    print('Start detection...')

    # Set dataset and device
    if os.path.isfile(opt.source):
        with open(opt.source, 'r', encoding='utf-8') as r_h:
            paths = [x.strip() for x in r_h.readlines()]
            print('Total {:d} image files.'.format(len(paths)))
            dataset = LoadImages(path=paths, img_size=opt.img_size)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)

    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    opt.device = device

    # Set result output
    frame_dir = opt.save_img_dir + '/frame'
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir)
    else:
        shutil.rmtree(frame_dir)
        os.makedirs(frame_dir)

    # class name to class id and class id to class name
    names = load_classes(opt.names)
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    # Set MCMOT tracker
    # tracker = JDETracker(opt)  # Joint detection and embedding
    tracker = MCJDETracker(opt)  # Multi-class joint detection & embedding

    for fr_id, (path, img, img0, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(opt.device)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = torch_utils.time_synchronized()

        # update detection result of this frame
        dets = tracker.update_detection(img, img0)

        t2 = torch_utils.time_synchronized()
        print('%sdone, time (%.3fs)' % (path, t2 - t1))

        if opt.show_image:
            online_im = vis.plot_detects(img=img0,
                                         dets=dets,
                                         num_classes=opt.num_classes,
                                         frame_id=fr_id,
                                         id2cls=id2cls)

        if opt.save_img_dir is not None:
            save_path = os.path.join(frame_dir, '{:05d}.jpg'.format(fr_id))
            cv2.imwrite(save_path, online_im)

        # output results as .txt file
        if dets is None:
            print('\n[Warning]: non objects detected in {}, frame id {:d}\n' \
                  .format(os.path.split(path), fr_id))
            dets_list = []
        else:
            dets_list = format_output(dets, w=img0.shape[1], h=img0.shape[0])

        # output label(txt) to
        out_img_name = os.path.split(path)[-1]
        out_f_name = out_img_name.replace('.jpg', '.txt')
        out_f_path = opt.output_txt_dir + '/' + out_f_name
        with open(out_f_path, 'w', encoding='utf-8') as w_h:
            w_h.write('class prob x y w h total=' + str(len(dets_list)) + '\n')  # write the first row
            for det in dets_list:
                w_h.write('%d %f %f %f %f %f\n' % (det[0], det[1], det[2], det[3], det[4], det[5]))
        print('{} written'.format(out_f_path))

    print('Total {:d} images tested.'.format(fr_id + 1))


def run_tracking_of_videos_txt(opt):
    """
    :param opt:
    :return:
    """
    if not os.path.isdir(opt.videos):
        print('[Err]: invalid video directory.')
        return

    # set device
    opt.device = str(FindFreeGPU())
    print('Using gpu: {:s}'.format(opt.device))
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    opt.device = device

    # set result output
    frame_dir = opt.save_img_dir + '/frame'
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir)

    # class name to class id and class id to class name
    names = load_classes(opt.names)
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    # Set MCMOT tracker
    # tracker = JDETracker(opt)  # Joint detection and embedding
    tracker = MCJDETracker(opt)  # Multi-class joint detection & embedding

    out_fps = int(opt.outFPS / opt.interval)
    data_type = 'mot'
    video_path_list = [opt.videos + '/' + x for x in os.listdir(opt.videos) if x.endswith('.mp4')]
    video_path_list.sort()

    # tracking each input video
    for video_i, video_path in enumerate(video_path_list):
        # set MCMOT tracker
        if video_i > 0:
            tracker.reset()

        # set dataset
        dataset = LoadImages(video_path, img_size=opt.img_size)

        # set txt results path
        src_name = os.path.split(video_path)[-1]
        name, suffix = src_name.split('.')
        result_f_name = opt.save_img_dir + '/' + name + '_results_fps{:d}.txt'.format(out_fps)

        # set dict to store tracking results for txt output
        results_dict = defaultdict(list)

        # set sampled frame count
        fr_cnt = 0

        # iterate tracking results of each frame
        for fr_id, (path, img, img0, vid_cap) in enumerate(dataset):
            img = torch.from_numpy(img).to(opt.device)
            img = img.float()  # uint8 to fp32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # update tracking result of this frame
            if opt.interval == 1:
                online_targets_dict = tracker.update_tracking(img, img0)

                # aggregate current frame's results for each object class
                online_tlwhs_dict = defaultdict(list)
                online_ids_dict = defaultdict(list)

                # iterate each object class
                for cls_id in range(opt.num_classes):  # process each object class
                    online_targets = online_targets_dict[cls_id]
                    for track in online_targets:
                        online_tlwhs_dict[cls_id].append(track.tlwh)
                        online_ids_dict[cls_id].append(track.track_id)

                # collect result
                for cls_id in range(opt.num_classes):
                    results_dict[cls_id].append((fr_id + 1, online_tlwhs_dict[cls_id], online_ids_dict[cls_id]))
            else:
                if fr_id % opt.interval == 0:  # skip some frames
                    online_targets_dict = tracker.update_tracking(img, img0)

                    # aggregate current frame's results for each object class
                    online_tlwhs_dict = defaultdict(list)
                    online_ids_dict = defaultdict(list)

                    # iterate each object class
                    for cls_id in range(opt.num_classes):  # process each object class
                        online_targets = online_targets_dict[cls_id]
                        for track in online_targets:
                            online_tlwhs_dict[cls_id].append(track.tlwh)
                            online_ids_dict[cls_id].append(track.track_id)

                    # collect result
                    for cls_id in range(opt.num_classes):
                        results_dict[cls_id].append((fr_cnt + 1, online_tlwhs_dict[cls_id], online_ids_dict[cls_id]))

                    # update sampled frame count
                    fr_cnt += 1

        if opt.interval == 1:
            print('Total {:d} frames.'.format(fr_id + 1))
        else:
            print('Total {:d} frames.'.format(fr_cnt))

        # output track/detection results as txt(MOT16 format)
        write_results_dict(result_f_name, results_dict, data_type)  # write txt to opt.save_img_dir


def run_tracking_of_videos_img(opt):
    """
    :param opt:
    :return:
    """
    if not os.path.isdir(opt.videos):
        print('[Err]: invalid video directory.')
        return

    # set device
    opt.device = str(FindFreeGPU())
    print('Using gpu: {:s}'.format(opt.device))
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    opt.device = device

    # set result output
    frame_dir = opt.save_img_dir + '/frame'
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir)

    # class name to class id and class id to class name
    names = load_classes(opt.names)
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    # Set MCMOT tracker
    tracker = MCJDETracker(opt)  # Multi-class joint detection & embedding

    out_fps = int(opt.outFPS / opt.interval)
    data_type = 'mot'
    video_path_list = [opt.videos + '/' + x for x in os.listdir(opt.videos) if x.endswith('.mp4')]
    video_path_list.sort()

    # tracking each input video
    for video_i, video_path in enumerate(video_path_list):
        if video_i > 0:
            tracker.reset()

        # set dataset
        dataset = LoadImages(video_path, img_size=opt.img_size)

        # set txt results path
        src_name = os.path.split(video_path)[-1]
        name, suffix = src_name.split('.')
        result_f_name = opt.save_img_dir + '/' + name + '_results_fps{:d}.txt'.format(out_fps)

        # set dict to store tracking results for txt output
        results_dict = defaultdict(list)

        # set sampled frame count
        fr_cnt = 0

        # reset(clear) frame directory: write opt.save_img_dir
        shutil.rmtree(frame_dir)
        os.makedirs(frame_dir)

        # iterate tracking results of each frame
        for fr_id, (path, img, img0, vid_cap) in enumerate(dataset):
            img = torch.from_numpy(img).to(opt.device)
            img = img.float()  # uint8 to fp32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # update tracking result of this frame
            if opt.interval == 1:

                # ----- update tracking result of current frame
                online_targets_dict = tracker.update_tracking(img, img0)
                # -----

                # aggregate current frame's results for each object class
                online_tlwhs_dict = defaultdict(list)
                online_ids_dict = defaultdict(list)
                for cls_id in range(opt.num_classes):  # process each object class
                    online_targets = online_targets_dict[cls_id]
                    for track in online_targets:
                        online_tlwhs_dict[cls_id].append(track.tlwh)
                        online_ids_dict[cls_id].append(track.track_id)

                # to draw track/detection
                online_im = vis.plot_tracks(image=img0,
                                            tlwhs_dict=online_tlwhs_dict,
                                            obj_ids_dict=online_ids_dict,
                                            num_classes=opt.num_classes,
                                            frame_id=fr_id,
                                            id2cls=id2cls)

                if opt.save_img_dir is not None:
                    save_path = os.path.join(frame_dir, '{:05d}.jpg'.format(fr_id))
                    cv2.imwrite(save_path, online_im)
            else:  # interval > 1
                if fr_id % opt.interval == 0:  # skip some frames

                    # ----- update tracking result of current frame
                    online_targets_dict = tracker.update_tracking(img, img0)
                    # -----

                    # aggregate current frame's results for each object class
                    online_tlwhs_dict = defaultdict(list)
                    online_ids_dict = defaultdict(list)

                    # iterate each object class
                    for cls_id in range(opt.num_classes):  # process each object class
                        online_targets = online_targets_dict[cls_id]
                        for track in online_targets:
                            online_tlwhs_dict[cls_id].append(track.tlwh)
                            online_ids_dict[cls_id].append(track.track_id)

                    # to draw track/detection
                    online_im = vis.plot_tracks(image=img0,
                                                tlwhs_dict=online_tlwhs_dict,
                                                obj_ids_dict=online_ids_dict,
                                                num_classes=opt.num_classes,
                                                frame_id=fr_cnt,
                                                id2cls=id2cls)

                    if opt.save_img_dir is not None:
                        save_path = os.path.join(frame_dir, '{:05d}.jpg'.format(fr_cnt))
                        cv2.imwrite(save_path, online_im)  # write img to opt.save_img_dir

                    # update sampled frame count
                    fr_cnt += 1

        # output tracking result as video: read and write opt.save_img_dir
        result_video_path = opt.save_img_dir + '/' + name + '_track' + '_fps' + str(out_fps) + '.' + suffix
        cmd_str = 'ffmpeg -f image2 -r {:d} -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}' \
            .format(out_fps, frame_dir, result_video_path)
        os.system(cmd_str)


def run_tracking(opt):
    """
    :param opt:
    :return:
    """
    # Set dataset
    dataset = LoadImages(opt.source, img_size=opt.img_size)

    # set device
    opt.device = str(FindFreeGPU())
    print('Using gpu: {:s}'.format(opt.device))
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    opt.device = device

    # Set result output
    frame_dir = opt.save_img_dir + '/frame'
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir)
    else:
        shutil.rmtree(frame_dir)
        os.makedirs(frame_dir)

    # class name to class id and class id to class name
    names = load_classes(opt.names)
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    # Set MCMOT tracker
    # tracker = JDETracker(opt)  # Joint detection and embedding
    tracker = MCJDETracker(opt)  # Multi-class joint detection & embedding

    # Update tracking frames
    out_fps = int(opt.outFPS / opt.interval)
    data_type = 'mot'
    src_name = os.path.split(opt.source)[-1]
    name, suffix = src_name.split('.')
    result_f_name = opt.save_img_dir + '/' + name + '_results_fps{:d}.txt'.format(out_fps)
    results_dict = defaultdict(list)  # to store tracking results for txt output

    fr_cnt = 0
    for fr_id, (path, img, img0, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(opt.device)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # update tracking result of this frame
        if opt.interval == 1:

            # update tracking result of the current frame
            track_dict = tracker.update_tracking(img, img0)
            if track_dict is None:
                print('[Note]: empty track dict for frame {:d}.'.format(fr_id))
                continue

            # aggregate current frame's results for each object class
            online_tlwhs_dict = defaultdict(list)
            online_ids_dict = defaultdict(list)
            for cls_id in range(opt.num_classes):  # process each object class
                cls_tracks = track_dict[cls_id]
                for track in cls_tracks:
                    online_tlwhs_dict[cls_id].append(track.tlwh)
                    online_ids_dict[cls_id].append(track.track_id)

            # collect result
            for cls_id in range(opt.num_classes):
                results_dict[cls_id].append((fr_id + 1, online_tlwhs_dict[cls_id], online_ids_dict[cls_id]))

            # to draw track/detection
            if opt.show_image:
                if tracker.frame_id > 0:
                    online_im = vis.plot_tracks(image=img0,
                                                tlwhs_dict=online_tlwhs_dict,
                                                obj_ids_dict=online_ids_dict,
                                                num_classes=opt.num_classes,
                                                frame_id=fr_id,
                                                id2cls=id2cls)

            if opt.save_img_dir is not None:
                img_save_path = os.path.join(frame_dir, '{:05d}.jpg'.format(fr_id))
                cv2.imwrite(img_save_path, online_im)
        else:
            if fr_id % opt.interval == 0:  # skip some frames

                # update tracking results of current frame
                track_dict = tracker.update_tracking(img, img0)

                # aggregate current frame's results for each object class
                online_tlwhs_dict = defaultdict(list)
                online_ids_dict = defaultdict(list)
                for cls_id in range(opt.num_classes):  # process each object class
                    cls_tracks = track_dict[cls_id]
                    for track in cls_tracks:
                        online_tlwhs_dict[cls_id].append(track.tlwh)
                        online_ids_dict[cls_id].append(track.track_id)

                # collect result
                for cls_id in range(opt.num_classes):
                    results_dict[cls_id].append((fr_cnt + 1, online_tlwhs_dict[cls_id], online_ids_dict[cls_id]))

                # to draw track/detection
                if opt.show_image:
                    if tracker.frame_id > 0:
                        online_im = vis.plot_tracks(image=img0,
                                                    tlwhs_dict=online_tlwhs_dict,
                                                    obj_ids_dict=online_ids_dict,
                                                    num_classes=opt.num_classes,
                                                    frame_id=fr_cnt,
                                                    id2cls=id2cls)

                if opt.save_img_dir is not None:
                    img_save_path = os.path.join(frame_dir, '{:05d}.jpg'.format(fr_cnt))
                    cv2.imwrite(img_save_path, online_im)

                # update sampled frame count
                fr_cnt += 1

    if opt.interval == 1:
        print('Total {:d} frames.'.format(fr_id + 1))
    else:
        print('Total {:d} frames.'.format(fr_cnt + 1))

    # output track/detection results as txt(MOT16 format)
    write_results_dict(result_f_name, results_dict, data_type)

    # output tracking result as video
    result_video_path = opt.save_img_dir + '/' + name + '_track' + '_fps' + str(out_fps) + '.' + suffix
    cmd_str = 'ffmpeg -f image2 -r {:d} -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}' \
        .format(out_fps, frame_dir, result_video_path)  # set output frame rate 12 FPS
    os.system(cmd_str)


def FindFreeGPU():
    """
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_left_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]

    most_free_gpu_idx = np.argmax(memory_left_gpu)
    # print(str(most_free_gpu_idx))
    return int(most_free_gpu_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov4-tiny-3l_no_group_id_no_upsample.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/mcmot.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/track_last.pt', help='weights path')

    # input file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='data/samples/test30.mp4', help='source')
    parser.add_argument('--videos', type=str, default='data/samples/', help='')
    # parser.add_argument('--source', type=str, default='/users/duanyou/c5/all_pretrain/test.txt', help='source')

    # output detection results as txt file for mMAP computation
    parser.add_argument('--output-txt-dir', type=str, default='/users/duanyou/c5/results_new/results_all/tmp')
    parser.add_argument('--save-img-dir', type=str, default='./results', help='dir to save visualized results(imgs).')

    # task mode
    parser.add_argument('--task', type=str, default='track', help='task mode: track or detect')

    # output FPS interval
    parser.add_argument('--interval', type=int, default=1, help='The interval frame of tracking, default no interval.')

    # standard output FPS
    parser.add_argument('--outFPS', type=int, default=12, help='The FPS of output video.')

    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=768, help='inference size (pixels)')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of object classes.')

    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer frames')

    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='7', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--show-image', type=bool, default=True, help='whether to show results.')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    # ----------
    if opt.task == 'track':
        run_tracking(opt)
        # run_tracking_of_videos_txt(opt)
        # run_tracking_of_videos_img(opt)
    elif opt.task == 'detect':
        run_detection(opt)
    else:
        print("[Err]: un-recognized task mode, neither 'track' or 'detect'")
    # ----------

    # TestFreeGPU()

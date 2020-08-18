# encoding=utf-8

import os
import torch
import argparse
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis


def run(opt):
    """
    :param opt:
    :return:
    """

    # Set dataset and device
    dataset = LoadImages(opt.source, img_size=opt.img_size)
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    opt.device = device

    # Set result output
    frame_dir = opt.save_dir + '/frame'
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

    # Set tracker
    tracker = JDETracker(opt)  # Joint detectionand embedding

    for fr_id, (path, img, img0, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(opt.device)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # update tracking result of this frame
        online_targets_dict = tracker.update_tracking(img, img0)
        # print(online_targets_dict)

        # aggregate frame's results
        online_tlwhs_dict = defaultdict(list)
        online_ids_dict = defaultdict(list)
        for cls_id in range(opt.num_classes):
            # process each object class
            online_targets = online_targets_dict[cls_id]
            for track in online_targets:
                tlwh = track.tlwh
                t_id = track.track_id
                # vertical = tlwh[2] / tlwh[3] > 1.6  # box宽高比判断:w/h不能超过1.6?
                # if tlwh[2] * tlwh[3] > opt.min_box_area:  # and not vertical:
                online_tlwhs_dict[cls_id].append(tlwh)
                online_ids_dict[cls_id].append(t_id)

        if opt.show_image:
            if tracker.frame_id > 0:
                online_im = vis.plot_tracks(image=img0,
                                            tlwhs_dict=online_tlwhs_dict,
                                            obj_ids_dict=online_ids_dict,
                                            num_classes=opt.num_classes,
                                            frame_id=fr_id,
                                            id2cls=id2cls)

        if opt.save_dir is not None:
            save_path = os.path.join(frame_dir, '{:05d}.jpg'.format(fr_id))
            cv2.imwrite(save_path, online_im)

    # output tracking result as video
    src_name = os.path.split(opt.source)[-1]
    name, suffix = src_name.split('.')
    result_video_path = opt.save_dir + '/' + name + '_track' + '.' + suffix

    cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}' \
        .format(frame_dir, result_video_path)
    os.system(cmd_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov4-paspp-mcmot.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/mcmot.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='weights path')

    # input file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='data/samples/test3.mp4', help='source')

    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=768, help='inference size (pixels)')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of object classes.')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer frames')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='4', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--save-dir', type=str, default='./results', help='dir to save results(imgs).')
    parser.add_argument('--show-image', type=bool, default=True, help='whether to show results.')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    run(opt)

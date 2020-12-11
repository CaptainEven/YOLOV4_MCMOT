import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch

import pickle
import numpy as np

from models import *
from utils.datasets import *
from utils.utils import *
from auto_weighted_loss import AutomaticWeightedLoss

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# Hyper-parameters
hyp = {
    'giou': 3.54,  # g_iou loss_funcs gain
    'cls': 37.4,  # cls loss_funcs gain
    'cls_pw': 1.0,  # cls BCELoss positive_weight
    'obj': 64.3,  # obj loss_funcs gain (*=img_size/320 if img_size != 320)
    'reid': 0.1,  # reid loss_funcs weight
    'obj_pw': 1.0,  # obj BCELoss positive_weight
    'iou_t': 0.20,  # iou training threshold
    'lr0': 0.0001,  # initial learning rate (SGD=5E-3, Adam=5E-4), default: 0.01
    'lrf': 0.0001,  # final learning rate (with cos scheduler)
    'momentum': 0.937,  # SGD momentum
    'weight_decay': 0.000484,  # optimizer weight decay
    'fl_gamma': 0.0,  # focal loss_funcs gamma (efficientDet default is gamma=1.5)
    'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
    'degrees': 1.98 * 0,  # image rotation (+/- deg)
    'translate': 0.05 * 0,  # image translation (+/- fraction)
    'scale': 0.5,  # image scale (+/- gain)
    'shear': 0.641 * 0  # image shear (+/- deg)
}

# automatically generate the max_ids_dict
global max_id_dict
# max_id_dict = {
#     0: 341,  # car
#     1: 103,  # bicycle
#     2: 104,  # person
#     3: 329,  # cyclist
#     4: 48  # tricycle
# }
#
# max_id_dict = {
#     0: 330,
#     1: 102,
#     2: 104,
#     3: 312,
#     4: 53
# }  # previous version

# ----- max_id_dict read from .npy(max_id_dict.npy file)
max_id_dict_file_path = '/mnt/diskb/even/dataset/MCMOT/max_id_dict.npz'
if os.path.isfile(max_id_dict_file_path):
    load_dict = np.load(max_id_dict_file_path, allow_pickle=True)
max_id_dict = load_dict['max_id_dict'][()]
print(max_id_dict)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss_funcs if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])


def train():
    global max_id_dict

    print('Task mode: {}'.format(opt.task))

    last = wdir + opt.task + '_last.pt'

    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)

    # Image Sizes
    gs = 64  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if opt.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = grid_min * gs, grid_max * gs
    img_size = imgsz_max  # initialize with max size

    # Configure run
    init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Dataset
    if opt.task == 'pure_detect':
        dataset = LoadImagesAndLabels(train_path,
                                      img_size,
                                      batch_size,
                                      augment=True,
                                      hyp=hyp,  # augmentation hyper parameters
                                      rect=opt.rect,  # rectangular training
                                      cache_images=opt.cache_images,
                                      single_cls=opt.single_cls)
    else:
        dataset = LoadImgsAndLbsWithID(train_path,
                                       img_size,
                                       batch_size,
                                       augment=True,
                                       hyp=hyp,  # augmentation hyper parameters
                                       rect=opt.rect,  # rectangular training
                                       cache_images=opt.cache_images,
                                       single_cls=opt.single_cls)

    # Initialize model
    if opt.task == 'pure_detect':
        model = Darknet(cfg=cfg,
                        img_size=img_size,
                        verbose=False,
                        max_id_dict=max_id_dict,  # after dataset's statistics
                        emb_dim=128,
                        mode=opt.task).to(device)
    else:
        max_id_dict = dataset.max_ids_dict
        model = Darknet(cfg=cfg,
                        img_size=img_size,
                        verbose=False,
                        max_id_dict=max_id_dict,  # using priori knowledge
                        emb_dim=128,
                        mode=opt.task).to(device)
    # print(model)
    print(max_id_dict)

    # Optimizer definition and model parameters registration
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    # do not succeed...
    if opt.auto_weight:
        if opt.task == 'pure_detect' or opt.task == 'detect':
            awl = AutomaticWeightedLoss(3)
        elif opt.task == 'track':
            awl = AutomaticWeightedLoss(4)

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

    if opt.auto_weight:
        optimizer.add_param_group({'params': awl.parameters(), 'weight_decay': 0})  # auto weighted params

    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0
    # attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        chkpt = torch.load(weights, map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e
        if 'epoch' in chkpt.keys():
            print('Checkpoint of epoch {} loaded.'.format(chkpt['epoch']))

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    # load dark-net format weights
    elif len(weights) > 0:
        load_darknet_weights(model, weights)

    # freeze weights of some previous layers(for yolo detection only)
    for layer_i, (name, child) in enumerate(model.module_list.named_children()):
        if layer_i < 51:
            for param in child.parameters():
                param.requires_grad = False
        else:
            print('Layer ', name, ' requires grad.')

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # see link below
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822

    ## Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9997',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layer_inds = model.module.yolo_layer_inds  # move yolo layer indices to top level

    # Data loader
    batch_size = min(batch_size, len(dataset))

    nw = 0  # for debugging
    if not opt.isdebug:
        nw = 8  # min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              num_workers=nw,
                                              shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                              pin_memory=True,
                                              collate_fn=dataset.collate_fn)

    # Test loader
    if opt.task == 'pure_detect':
        test_loader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path,
                                                                      imgsz_test,
                                                                      batch_size,
                                                                      hyp=hyp,
                                                                      rect=True,  # True
                                                                      cache_images=opt.cache_images,
                                                                      single_cls=opt.single_cls),
                                                  batch_size=batch_size,
                                                  num_workers=nw,
                                                  pin_memory=True,
                                                  collate_fn=dataset.collate_fn)
    else:
        test_loader = torch.utils.data.DataLoader(LoadImgsAndLbsWithID(test_path,
                                                                       imgsz_test,
                                                                       batch_size,
                                                                       hyp=hyp,
                                                                       rect=True,
                                                                       cache_images=opt.cache_images,
                                                                       single_cls=opt.single_cls),
                                                  batch_size=batch_size,
                                                  num_workers=nw,
                                                  pin_memory=True,
                                                  collate_fn=dataset.collate_fn)

    # Define model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyper-parameters to model
    model.gr = 1.0  # g_iou loss_funcs ratio (obj_loss = 1.0 or g_iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights

    # Model EMA: exponential moving average
    ema = torch_utils.ModelEMA(model)

    # Start training
    nb = len(data_loader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'

    t0 = time.time()

    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g data_loader workers' % nw)
    print('Starting training for %g epochs...' % epochs)

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()  # train mode

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        if opt.task == 'pure_detect' or opt.task == 'detect':
            m_loss = torch.zeros(4).to(device)  # mean losses
        elif opt.task == 'track':
            m_loss = torch.zeros(5).to(device)  # mean losses
        else:
            print('[Err]: unrecognized task mode.')
            return

        if opt.task == 'track':
            print(('\n' + '%10s' * 9) % (
                'Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'reid', 'total', 'targets', 'img_size'))
        elif opt.task == 'detect' or opt.task == 'pure_detect':
            print(('\n' + '%10s' * 8) % (
                'Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        else:
            print('[Err]: unrecognized task mode.')
            return

        p_bar = tqdm(enumerate(data_loader), total=nb)  # progress bar
        if opt.task == 'pure_detect' or opt.task == 'detect':
            for batch_i, (imgs, targets, paths,
                          shape) in p_bar:  # batch -------------------------------------------------------------
                ni = batch_i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
                targets = targets.to(device)

                # Burn-in
                if ni <= n_burn * 2:
                    model.gr = np.interp(ni, [0, n_burn * 2],
                                         [0.0, 1.0])  # giou loss_funcs ratio (obj_loss = 1.0 or giou)
                    if ni == n_burn:  # burn_in complete
                        print_model_biases(model)

                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])

                # Multi-Scale
                if opt.multi_scale:
                    if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                        img_size = random.randrange(grid_min, grid_max + 1) * gs
                    sf = img_size / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                pred = model.forward(imgs)

                # Loss
                loss, loss_items = compute_loss(pred, targets, model)

                if not torch.isfinite(loss):
                    print('[Warning]: infinite loss_funcs, ending training ', loss_items)
                    return results

                # Backward
                loss *= batch_size / 64.0  # scale loss_funcs
                if mixed_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Optimize
                if ni % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update(model)

                # Print
                m_loss = (m_loss * batch_i + loss_items) / (batch_i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *m_loss, len(targets), img_size)
                p_bar.set_description(s)

                # Plot
                if ni < 1:
                    f = 'train_batch%g.jpg' % batch_i  # filename
                    plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer:
                        tb_writer.add_image(f, cv2.imread(f)[:, :, ::-1], dataformats='HWC')
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

                # Save model
                if ni != 0 and ni % 300 == 0:  # save checkpoint every 100 batches
                    save = (not opt.nosave) or (not opt.evolve)
                    if save:
                        chkpt = {'epoch': epoch,
                                 'batch': ni,
                                 'best_fitness': best_fitness,
                                 'model': ema.ema.module.state_dict() \
                                     if hasattr(model, 'module') else ema.ema.state_dict(),
                                 'optimizer': optimizer.state_dict()}

                        # Save last, best and delete
                        torch.save(chkpt, last)
                        print('{:s} saved.'.format(last))
                        del chkpt

                        # Save .weights file
                        wei_f_path = wdir + opt.task + '_last.weights'
                        save_weights(model, wei_f_path)
                        print('{:s} saved.'.format(wei_f_path))

        elif opt.task == 'track':
            for batch_i, (imgs, targets, paths, shape,
                          track_ids) in p_bar:  # batch -------------------------------------------------------------
                ni = batch_i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
                targets = targets.to(device)
                track_ids = track_ids.to(device)

                # Burn-in
                if ni <= n_burn * 2:
                    model.gr = np.interp(ni, [0, n_burn * 2],
                                         [0.0, 1.0])  # giou loss_funcs ratio (obj_loss = 1.0 or giou)
                    if ni == n_burn:  # burnin complete
                        print_model_biases(model)

                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])
                    # print('Lr {:.3f}'.format(x['lr']))

                # Multi-Scale
                if opt.multi_scale:
                    if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                        img_size = random.randrange(grid_min, grid_max + 1) * gs
                    sf = img_size / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                pred, reid_feat_out = model.forward(imgs)

                # Loss
                loss, loss_items = compute_loss_no_upsample(pred, reid_feat_out, targets, track_ids, model)

                if opt.auto_weight:
                    loss = awl.forward(loss_items[0], loss_items[1], loss_items[2], loss_items[3])

                if not torch.isfinite(loss_items[3]):
                    print('[Warning]: infinite reid loss.')
                    loss_items[3:] = torch.zeros((1, 1), device=device)
                if not torch.isfinite(loss):
                    for i in range(loss_items.shape[0]):
                        loss_items[i] = torch.zeros((1, 1), device=device)
                    print('[Warning] infinite loss_funcs', loss_items)  # ending training
                    return results

                # Backward
                loss *= batch_size / 64.0  # scale loss_funcs
                if mixed_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Optimize
                if ni % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update(model)

                # Print
                m_loss = (m_loss * batch_i + loss_items) / (batch_i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 7) % ('%g/%g' % (epoch, epochs - 1), mem, *m_loss, len(targets), img_size)
                p_bar.set_description(s)

                # Plot
                if ni < 1:
                    f = 'train_batch%g.jpg' % batch_i  # filename
                    plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer:
                        tb_writer.add_image(f, cv2.imread(f)[:, :, ::-1], dataformats='HWC')
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

                # Save model
                if ni != 0 and ni % 300 == 0:  # save checkpoint every 100 batches
                    save = (not opt.nosave) or (not opt.evolve)
                    if save:
                        chkpt = {'epoch': epoch,
                                 'batch': ni,
                                 'best_fitness': best_fitness,
                                 'model': ema.ema.module.state_dict() \
                                     if hasattr(model, 'module') else ema.ema.state_dict(),
                                 'optimizer': optimizer.state_dict()}

                        # Save last, best and delete
                        torch.save(chkpt, last)
                        print('{:s} saved.'.format(last))
                        del chkpt

                        # Save .weights file
                        wei_f_path = wdir + opt.task + '_last.weights'
                        save_weights(model, wei_f_path)
                        print('{:s} saved.'.format(wei_f_path))

                # end batch ------------------------------------------------------------------------------------------------
        else:
            print('[Err]: unrecognized task mode.')
            return

        # Update scheduler
        scheduler.step()

        # Process epoch results
        ema.update_attr(model)

        final_epoch = epoch + 1 == epochs

        if not opt.notest or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, maps = test.test(cfg,
                                      data,
                                      batch_size=batch_size,
                                      img_size=imgsz_test,
                                      model=ema.ema,
                                      save_json=final_epoch and is_coco,
                                      single_cls=opt.single_cls,
                                      data_loader=test_loader,
                                      task=opt.task)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        if len(opt.name) and opt.bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Tensorboard
        if tb_writer:
            tags = ['train/giou_loss',
                    'train/obj_loss',
                    'train/cls_loss',
                    'train/reid_loss',
                    'metrics/precision',
                    'metrics/recall',
                    'metrics/mAP_0.5',
                    'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(m_loss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            # create checkpoint: whithin an epoch, no results yet, donot save results.txt
            chkpt = {'epoch': epoch,
                     'best_fitness': best_fitness,
                     'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                     'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(chkpt, last)
            if (best_fitness == fi) and not final_epoch:
                torch.save(chkpt, best)
            del chkpt

            # # Save .weights file
            # wei_f_path = wdir + opt.task + '_last.weights'
            # save_weights(model, wei_f_path)
            # print('{:s} saved.'.format(wei_f_path))

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    n = opt.name
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

    if not opt.evolve:
        plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=8)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size',
                        nargs='+',
                        type=int,
                        default=[384, 832, 768],
                        help='[min_train, max-train, test]')  # [320, 640]
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyper parameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')

    parser.add_argument('--data',
                        type=str,
                        default='data/mcmot.data',
                        help='*.data path')

    # ---------- weights and cfg file
    parser.add_argument('--cfg',
                        type=str,
                        default='cfg/yolov4-tiny-3l_no_group_id_no_upsample.cfg',
                        help='*.cfg path')

    parser.add_argument('--weights',
                        type=str,
                        default='./weights/track_last.weights',
                        help='initial weights path')
    # ----------

    parser.add_argument('--name',
                        default='yolov4-mobilenetv2',
                        help='renames results.txt to results_name.txt if supplied')

    parser.add_argument('--device',
                        default='6',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')

    # Set 3 task mode: pure_detect | detect | track
    # pure detect means the dataset do not contains ID info.
    # detect means the dataset contains ID info, but do not load for training. (i.e. do detection in tracking)
    # track means the dataset contains both detection and ID info, use both for training. (i.e. detect & reid)
    parser.add_argument('--task',
                        type=str,
                        default='track',
                        help='pure_detect, detect or track mode.')

    parser.add_argument('--auto-weight', type=bool, default=False, help='Whether use auto weight tuning')

    # use debug mode to enforce the parameter of worker number to be 0
    parser.add_argument('--isdebug',
                        type=bool,
                        default=True,
                        help='whether in debug mode or not')

    opt = parser.parse_args()
    opt.weights = last if opt.resume else opt.weights
    check_git_status()
    print(opt)

    # ----- Set image size for training and testing
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)

    # ----- Set device
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    # scale hyp['obj'] by img_size (evolved at 320)
    # hyp['obj'] *= opt.img_size[0] / 320.

    tb_writer = None
    if not opt.evolve:  # Train normally
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(comment=opt.name)
        train()  # train normally

    else:  # Evolve hyper parameters (optional)
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                if method == 1:
                    v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
                elif method == 2:
                    v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
                elif method == 3:
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train()

            # Write mutation results
            print_mutation(hyp, results, opt.bucket)

            # Plot results
            # plot_evolution_results(hyp)

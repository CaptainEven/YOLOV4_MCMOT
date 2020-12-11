from utils.google_utils import *
from utils.layers import *
from utils.parse_config import *

ONNX_EXPORT = False


# Parse cfg file, create every layer
def create_modules(module_defs, img_size, cfg, id_classifiers=None):
    # Constructs module list of layer blocks from module configuration in module_defs

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels

    # define modules to register
    module_list = nn.ModuleList()

    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('Conv2d',
                                   nn.Conv2d(in_channels=output_filters[-1],
                                             out_channels=filters,
                                             kernel_size=k,
                                             stride=stride,
                                             padding=k // 2 if 'pad' in mdef else 0,
                                             groups=mdef['groups'] if 'groups' in mdef else 1,
                                             bias=not bn))

            else:  # multiple-size conv
                modules.add_module('MixConv2d',
                                   MixConv2d(in_ch=output_filters[-1],
                                             out_ch=filters,
                                             k=k,
                                             stride=stride,
                                             bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-5))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU(inplace=True))
            elif mdef['activation'] == 'logistic':  # Add logistic activation support
                modules.add_module('activation', nn.Sigmoid())
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())

        # To parse deconvolution for learnable up-sampling
        elif mdef['type'] == 'deconvolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('DeConv2d', nn.ConvTranspose2d(in_channels=output_filters[-1],
                                                                  out_channels=filters,
                                                                  kernel_size=k,
                                                                  stride=stride,
                                                                  padding=k // 2 if mdef['pad'] else 0,
                                                                  groups=mdef['groups'] if 'groups' in mdef else 1,
                                                                  bias=not bn))
            else:  # multiple-size conv
                modules.add_module('MixDeConv2d', MixDeConv2d(in_ch=output_filters[-1],
                                                              out_ch=filters,
                                                              k=k,
                                                              stride=stride,
                                                              bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-5))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU(inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())

        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        # Add support for global average pooling layer
        elif mdef['type'] == 'avgpool':
            modules = GlobalAvgPool()

        # Add support for dropout layer
        elif mdef['type'] == 'dropout':
            prob = mdef['probability']
            modules = Dropout(prob=prob)

        # Add support for scale channels
        elif mdef['type'] == 'scale_channels':
            layers = mdef['from']
            filters = output_filters[-1]  #
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ScaleChannels(layers=layers)

        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])

        # Add support for group route
        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            # layers = mdef['layers']
            # filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            # routs.extend([i + l if l < 0 else l for l in layers])
            # modules = FeatureConcat(layers=layers)

            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])

            if 'groups' in mdef:
                groups = mdef['groups']
                group_id = mdef['group_id']
                modules = RouteGroup(layers, groups, group_id)
                filters //= groups
            else:
                modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'route_lhalf':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers]) // 2
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat_l(layers=layers)

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])

            # modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

            # ----- to merge a shortcut layer and an activation layer to one layer
            modules.add_module('WeightedFeatureFusion',
                               WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef))

            # ----- add activation layer after a shortcut layer
            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [8, 16, 32]  # P5, P4, P3 strides
            if any(x in cfg for x in ['yolov4-tiny', 'mobile', 'Mobile', 'enet', 'Enet']):  # stride order reversed
                stride = [32, 16, 8]

            layers = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_idx=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = layers[yolo_index] if 'from' in mdef else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3, 85)
                bias[:, 4] += -4.5  # obj
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # ---------- Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)
        # ----------

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True

    return module_list, routs_binary


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_idx, layers, stride):
        """
        :param anchors:
        :param nc:
        :param img_size:
        :param yolo_idx:
        :param layers:
        :param stride:
        """
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.index = yolo_idx  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y grid points
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, pred, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.inde
            x, self.nl  # index in layers, number of layers
            pred = out[self.layers[i]]
            bs, _, ny, nx = pred.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), pred.device)

            # outputs and weights
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(pred[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            pred = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    pred += w[:, j:j + 1] * \
                            F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear',
                                          align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = pred.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids(ng=(nx, ny), device=pred.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, na, ny, nx, no(classes + xywh))
        pred = pred.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return pred

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            pred = pred.view(m, self.no)
            xy = torch.sigmoid(pred[:, 0:2]) + grid  # x, y
            wh = torch.exp(pred[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(pred[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(pred[:, 5:self.no]) * torch.sigmoid(pred[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = pred.clone()  # inference output

            # process pred to io
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh YOLO method
            io[..., :4] *= self.stride  # map from YOLO layer's scale to net input's scale
            torch.sigmoid_(io[..., 4:])  # sigmoid for confidence score and cls pred

            # gathered pred output: io: view [1, 3, 13, 13, 85] as [1, 507, 85]
            io = io.view(bs, -1, self.no)

            # yolo inds
            # yolo_inds = torch.full((io.size(0), io.size(1), 1), self.index, dtype=torch.long)

            return io, pred  # , yolo_inds


class Darknet(nn.Module):
    # YOLOv3/v4 object detection model
    def __init__(self,
                 cfg,
                 img_size=(416, 416),
                 verbose=False,
                 max_id_dict=None,
                 emb_dim=128,
                 mode='detect'):
        """
        :param cfg:
        :param img_size:
        :param verbose:
        :param max_id_dict:
        :param emb_dim: record max id numbers for each object class, used to do reid classification
        :param mode: output detection or tracking(detection + reid vector)
        """
        super(Darknet, self).__init__()

        self.mode = mode

        # ---------- parsing cfg file
        self.module_defs = parse_model_cfg(cfg)

        # create module list from cfg file
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)

        # ----- Define ReID classifiers
        if max_id_dict is not None:
            self.max_id_dict = max_id_dict
            self.emb_dim = emb_dim  # dimension of embedding feature vector
            self.id_classifiers = nn.ModuleList()  # num_classes layers of FC

            for cls_id, nID in self.max_id_dict.items():
                # choice 1: use normal FC layers as classifiers
                self.id_classifiers.append(nn.Linear(self.emb_dim, nID))  # FC layers

            # add reid classifiers(nn.ModuleList) to self.module_list to be registered
            self.module_list.append(self.id_classifiers)

        self.yolo_layer_inds = get_yolo_layers(self)
        # torch_utils.initialize_weights(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, augment=False, verbose=False):
        if not augment:
            return self.forward_once(x, verbose=verbose)

        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out, reid_feat_out = [], [], []  # 3(or 2) yolo laers correspond to 3(or 2) reid feature map layers
        if verbose:
            print('0', x.shape)
            str = ''

        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)

        # ---------- traverse the network(by traversing the module_list)
        use_output_layers = ['WeightedFeatureFusion',  # Shortcut(add)
                             'FeatureConcat',  # Route(concatenate)
                             'FeatureConcat_l',
                             'RouteGroup',
                             'ScaleChannels']
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in use_output_layers:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])

                x = module.forward(x, out)

            elif name == 'YOLOLayer':  # x: current layer, out: previous layers output
                yolo_out.append(module.forward(x, out))

            elif name == 'ModuleList':  # last 5 layers of FC: reid classifiers
                continue

            # We need to process a shortcut layer combined with a activation layer
            # followed by a activation layer
            elif name == 'Sequential':
                for layer in module:
                    layer_name = layer.__class__.__name__
                    if layer_name in use_output_layers:
                        x = layer.forward(x, out)
                    else:
                        x = layer.forward(x)

            # run module directly, i.e. mtype = 'upsample', 'maxpool', 'batchnorm2d' etc.
            else:
                x = module(x)

            # ----------- record previous output layers
            out.append(x if self.routs[i] else [])
            # out.append(x)  # for debugging...

            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''
        # ----------

        # ----------for debugging...
        # net_out_path = '/mnt/diskb/even/net_out_pt.txt'
        # with open(net_out_path, 'w', encoding='utf-8') as f:
        #     for i, layer in enumerate(out):
        #         # if i != 36 and i != 43 and i != 50:
        #         #     continue
        #
        #         f.write('Layer {:d}, shape: {:d}×{:d}×{:d}×{:d}\n'
        #                 .format(i, layer.shape[0], layer.shape[1], layer.shape[2], layer.shape[3]))
        #
        #         if layer.numel() < 64:
        #             tmp = layer.view(1, -1).squeeze()
        #         else:
        #             tmp = layer[0, 0, 0, :64]
        #         for j, k in enumerate(tmp):
        #             if j != 0 and j % 8 == 0:
        #                 f.write('\n')
        #             f.write('{:.6f} '.format(k.item()))
        #
        #         f.write('\n\n\n')
        # # ----------

        # Get 3 or 2 feature map layers for reid feature vector extraction
        # reid_feat_out.append(out[-5])  # the 1st YOLO scale feature map
        # reid_feat_out.append(out[-3])  # the 2nd YOLO scale feature map
        # reid_feat_out.append(out[-1])  # the 3rd YOLO scale feature map

        # @even: Get feature maps(corresponding to yolo layers)
        yolo_inds = [-1 - i * 2 for i in range(len(self.yolo_layer_inds))]
        yolo_inds.sort()
        for yolo_idx in yolo_inds:
            yolo_layer = out[yolo_idx]
            reid_feat_out.append(yolo_layer)

        # 3(or 2) yolo output layers and 3 feature layers
        # return out[36], out[43], out[50], out[-5], out[-3], out[-1]  # for yolov4-tiny-3l
        # return out[69], out[79], out[-3], out[-1]  # for mbv2-2l
        # return out[69], out[79], out[89], out[-5], out[-3], out[-1]  # for mbv2-3l

        # ----- Output mode
        if self.training:  # train
            if self.mode == 'pure_detect' or self.mode == 'detect':
                return yolo_out
            elif self.mode == 'track':
                return yolo_out, reid_feat_out
            else:
                print('[Err]: unrecognized task mode.')
                return None
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output

            # ----- record anchor inds
            for yolo_i, yolo_out in enumerate(x):
                yolo_inds_i = torch.full((yolo_out.size(0), yolo_out.size(1), 1), yolo_i, dtype=torch.long)
                if yolo_i == 0:
                    yolo_inds = yolo_inds_i
                else:
                    yolo_inds = torch.cat((yolo_inds, yolo_inds_i), 1)

            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)

            if self.mode == 'pure_detect' or self.mode == 'detect':
                return x, p
            elif self.mode == 'track':
                return x, p, reid_feat_out, yolo_inds
            else:
                print('[Err]: un-recognized mode, return None.')
                return None

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        children = list(self.children())[0]
        for ch_i, a in enumerate(children):
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        try:
                            fused = torch_utils.fuse_conv_and_bn(conv, b)
                        except Exception as e:
                            print(e)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training
        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    # for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
    for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
        # if i > 51:
        #     break

        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases

                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb

                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb

                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb

                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()

                try:
                    conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                except Exception as e:
                    print(e)

                conv.bias.data.copy_(conv_b)
                ptr += nb

            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        multi_gpu = type(self) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
        if multi_gpu:
            self.module.version.tofile(f)  # (int32) version info: major, minor, revision
            self.module.seen.tofile(f)  # (int64) number of images seen during training

            # Iterate through layers
            for i, (mdef, module) in enumerate(zip(self.module.module_defs[:cutoff], self.module.module_list[:cutoff])):
                # for i, (mdef, module) in enumerate(zip(self.module.module_defs, self.module.module_list)):

                if mdef['type'] == 'convolutional':
                    conv_layer = module[0]
                    # If batch norm, load bn first
                    if mdef['batch_normalize']:
                        bn_layer = module[1]
                        bn_layer.bias.data.cpu().numpy().tofile(f)
                        bn_layer.weight.data.cpu().numpy().tofile(f)
                        bn_layer.running_mean.data.cpu().numpy().tofile(f)
                        bn_layer.running_var.data.cpu().numpy().tofile(f)
                    # Load conv bias
                    else:
                        conv_layer.bias.data.cpu().numpy().tofile(f)
                    # Load conv weights
                    conv_layer.weight.data.cpu().numpy().tofile(f)
        else:
            self.version.tofile(f)  # (int32) version info: major, minor, revision
            self.seen.tofile(f)  # (int64) number of images seen during training

            # Iterate through layers
            # for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
                if mdef['type'] == 'convolutional':
                    conv_layer = module[0]
                    # If batch norm, load bn first
                    if mdef['batch_normalize']:
                        bn_layer = module[1]
                        bn_layer.bias.data.cpu().numpy().tofile(f)
                        bn_layer.weight.data.cpu().numpy().tofile(f)
                        bn_layer.running_mean.data.cpu().numpy().tofile(f)
                        bn_layer.running_var.data.cpu().numpy().tofile(f)
                    # Load conv bias
                    else:
                        conv_layer.bias.data.cpu().numpy().tofile(f)
                    # Load conv weights
                    conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov4-pacsp.cfg', weights='weights/yolov4-pacsp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {
            'epoch': -1,
            'best_fitness': None,
            'training_results': None,
            'model': model.state_dict(),
            'optimizer': None
        }

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip()
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if len(weights) > 0 and not os.path.isfile(weights):
        d = {'': ''}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)

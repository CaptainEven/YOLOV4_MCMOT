 python3 ./train.py --cfg ./cfg/MobileNetV2-YOLO_2l_one_feat_fuse.cfg \
                    --weights ./weights/mobile_207000.weights \
                    --cutoff 80 \
                    --stop-freeze-layer-idx 81 \
                    --batch-size 48 \
                    --adam 1 \
                    --nw 8 \
                    --debug 0 \
                    --lr 1e-4 \
                    --name mcmot_mbv2_2l
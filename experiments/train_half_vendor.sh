 python3 ./train.py --cfg ./cfg/yolov4_half_one_feat_fuse.cfg \
                    --weights ./weights/mcmot_half_vendor_track_last.weights \
                    --cutoff 161 \
                    --stop-freeze-layer-idx 162 \
                    --batch-size 32 \
                    --adam 1 \
                    --nw 8 \
                    --debug 0 \
                    --lr 1.0e-4 \
                    --name mcmot_half_vendor
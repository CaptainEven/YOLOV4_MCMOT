 python3 ./train.py --cfg ./cfg/yolov4_half_one_feat_fuse.cfg \
                    --weights ./weights/mcmot_half_track_last_210508.weights \
                    --cutoff 161 \
                    --stop-freeze-layer-idx 162 \
                    --batch-size 20 \
                    --adam 1 \
                    --nw 16 \
                    --debug 0 \
                    --lr 5.0e-5 \
                    --name mcmot_half
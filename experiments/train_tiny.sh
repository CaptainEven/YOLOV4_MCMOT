 python3 ./train.py --cfg ./cfg/yolov4-tiny-3l_no_group_id_SE_one_feat_fuse.cfg \
                    --weights ./weights/mcmot_tiny_track_last.pt \
                    --cutoff 48 \
                    --stop-freeze-layer-idx 49 \
                    --batch-size 56 \
                    --adam 1 \
                    --nw 16 \
                    --debug 0 \
                    --lr 1e-4
 python3 ./train.py --cfg ./cfg/yolov4_half_plm_one_feat_fuse.cfg \
                    --weights ./weights/yolov4_half_plm_34000.weights \
                    --cutoff 161 \
                    --stop-freeze-layer-idx 162 \
                    --batch-size 22 \
                    --adam 1 \
                    --nw 8 \
                    --debug 0 \
                    --lr 1.0e-4 \
                    --name mcmot_half_plm
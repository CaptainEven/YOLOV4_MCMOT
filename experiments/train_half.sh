 python3 ./train.py --cfg ./cfg/yolov4_half_one_feat_fuse.cfg \
                    --weights ./weights/yolov4_half_276000.weights \
                    --cutoff 161 \
                    --stop-freeze-layer-idx 162 \
                    --batch-size 32 \
                    --adam 1 \
                    --nw 16 \
                    --debug 0 \
                    --lr 1.0e-4 \
                    --name mcmot_half
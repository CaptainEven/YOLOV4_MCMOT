 python3 ./train.py --cfg ./cfg/yolov4_new_tiny_mcmot.cfg \
                    --weights ./weights/mcmot_new_tiny_track_last.pt \
                    --cutoff 64 \
                    --stop-freeze-layer-idx 65 \
                    --batch-size 28 \
                    --adam 1 \
                    --nw 16 \
                    --debug 0 \
                    --lr 1e-4 \
                    --name mcmot_new_tiny
python test.py --weights checkpoints/model_epoch_50.pkl --img_path 000000435081.jpg --conf_threshold 0.3

python vis.py --weights checkpoints/model_epoch_50.pkl --img_path test.png --conf_threshold 0.3


python train.py --epochs 2 --batch_size 4 --lr 1e-4 

python train.py --subset_size 100
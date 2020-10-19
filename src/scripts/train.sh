cd ..

python train.py 2 --batch_size 8 --num_epochs 30 --lr 0.01 --gpus 1 --num_workers 1
python train.py 3 --batch_size 8 --num_epochs 30 --lr 0.01 --gpus 1 --num_workers 1
python train.py 4 --batch_size 8 --num_epochs 30 --lr 0.001 --gpus 1 --num_workers 1 --optimizer RAdam
python train.py 6 --batch_size 8 --num_epochs 30 --lr 0.001 --gpus 1 --num_workers 1 --optimizer RAdam --limit
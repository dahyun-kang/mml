# Memory classification

## Conda installation command
```bash
conda env create -f environment.yml --prefix $YOURPREFIX
```
`$YOUPREFIX` is typically `/home/anaconda3`

## Training command for cifar10/cifar100/food101/places365/fgvcaircraft/stl10
```python
python main.py \
    --datapath $YOURDATASETPATH \
    --backbone {resnet18/resnet50/clipRN50/clipvitb} \
    --dataset {cifar10/cifar100/food101/places365/fgvcaircraft/stl10} \
    --logpath $YOURLOGPATH
```

## Training command for imagenetLT/placesLT
```python
python main.py \
    --datapath $YOURDATASETPATH \
    --backbone {resnet18/resnet50/clipRN50/clipvitb} \
    --dataset {imagenetLT/placesLT} \
    --LT \
    --sampler {Classaware/SquareRoot} \
    --logpath $YOURLOGPATH
```

## Reproduce Experiments
### Joint learning
```python
python main.py --datapath data --dataset imagenetLT --backbone resnext50 --lr 0.025 --maxepochs 90 --Decoupled joint --batchsize 64
```
### cRT learning
```python
python main.py --datapath data --dataset imagenetLT --backbone resnext50 --lr 0.025 --maxepochs 100 --Decoupled cRT --batchsize 64 --sampler ClassAware
```
### feat_extract (Do before tau-normalize)
```python
python main.py --datapath data --dataset imagenetLT --backbone resnext50 --Decoupled feat_extract --nowandb --batchsize 64
```
### tau-normalize
```python
python main.py --datapath data --dataset imagenetLT --backbone resnext50 --Decoupled tau --nowandb
```


## Flags
- Raise `--nowandb` for no wandb logging
- Raise `--eval` for evaluating the best checkpoint of the corresponding `--logpath` experiment
- Raise `--resume` for resume from the last checkpoint of the corresponding `--logpath` experiment

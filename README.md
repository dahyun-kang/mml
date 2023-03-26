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

## Flags
- Raise `--nowandb` for no wandb logging
- Raise `--eval` for evaluating the best checkpoint of the corresponding `--logpath` experiment
- Raise `--resume` for resume from the last checkpoint of the corresponding `--logpath` experiment

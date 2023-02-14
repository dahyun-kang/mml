# memory classification

## conda installation command
```bash
conda env create -f environment.yml --prefix $YOURPREFIX
```
`$YOUPREFIX` is typically `/home/anaconda3`


## training command
```python
python main.py \
    --datapath $YOURDATASETPATH \
    --backbone {resnet18/resnet50/clipRN50/clipvitb} \
    --dataset {cifar10/cifar100/food101/places365/fgvcaircraft/stl10} \
    --logpath $YOURLOGPATH
```

raise `--nowandb` for no wandb logging

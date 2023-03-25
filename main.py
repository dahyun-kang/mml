""" Memory Referencing Classification """
import os
import argparse
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from datamodule import return_datamodule
from model.memclslearner import MemClsLearner

from model.decoupled import Decoupled_learner
from callbacks import CustomCheckpoint



if __name__ == '__main__':
    seed_everything(7)

    parser = argparse.ArgumentParser(description='Query-Adaptive Memory Referencing Classification')
    parser.add_argument('--datapath', type=str, default='/ssd1t/datasets', help='Dataset root path')
    parser.add_argument('--dataset', type=str, default=None, help='Experiment dataset')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone; clip-trained model should have the keywoard \"clip\"')
    parser.add_argument('--logpath', type=str, default='', help='Checkpoint saving dir identifier')
    parser.add_argument('--batchsize', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--k', type=int, default=10, help='K KNN')
    parser.add_argument('--ntokens', type=int, default=0, help='Number of tokens')
    parser.add_argument('--maxepochs', type=int, default=500, help='Max iterations')
    parser.add_argument('--nowandb', action='store_true', help='Flag not to log at wandb')
    parser.add_argument('--nakata22', action='store_true', help='Flag to run Nataka et al., ECCV 2022')
    parser.add_argument('--LT', action='store_true', help='Flag to run Longtailed Learning')
    parser.add_argument('--sampler', type=str, default=None, choices=['ClassAware', 'SquareRoot'], help='Choose your sampler for training')
    parser.add_argument('--Decoupled', action='store_true', help='Flag to run reproducing expriement of Decoupled Learning')
    parser.add_argument('--eval', action='store_true', help='Flag for evaluation')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from the last point of logpath')

    args = parser.parse_args()

    args.many_shot_thr = 100
    args.low_shot_thr = 20

    if args.dataset == 'places365':
        args.datapath = os.path.join(args.datapath, 'places365')

    dm = return_datamodule(args.datapath, args.dataset, args.batchsize, args.backbone, args.sampler)
    if args.Decoupled:
        model = Decoupled_learner(args, dm=dm)
    else:
        model = MemClsLearner(args, dm=dm)
        if args.nakata22:
            model.forward = model.forward_nakata22

    checkpoint_callback = CustomCheckpoint(args)
    trainer = Trainer(
        max_epochs=args.maxepochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=CSVLogger(save_dir='logs') if args.nowandb else WandbLogger(name=args.logpath, save_dir='logs', project=f'qamr-{args.dataset}-{args.backbone}'),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10), checkpoint_callback],
        num_sanity_val_steps=0,
        # gradient_clip_val=5.0,
    )

    if args.nakata22:
        # non-differentiable majority voting method, Nakata et al., ECCV 2022
        trainer.test(model, datamodule=dm)
    else:
        if args.eval:
            modelpath = checkpoint_callback.modelpath
            model = MemClsLearner.load_from_checkpoint(modelpath, args=args, dm=dm)
            trainer.test(model=model, datamodule=dm)
        else:
            trainer.fit(model, dm)

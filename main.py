""" Memory Referencing Classification """
import math
import os
import argparse
import torch
import numpy as np

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from datamodule import return_datamodule
from model.mmltrainer import MemoryModularLearnerTrainer
from callbacks import CustomCheckpoint

import submitit
from submitit.helpers import RsyncSnapshot


if __name__ == '__main__':
    seed_everything(7)

    parser = argparse.ArgumentParser(description='Query-Adaptive Memory Referencing Classification')
    parser.add_argument('--datapath', type=str, default='/ssd1t/datasets', help='Dataset root path')
    parser.add_argument('--dataset', type=str, default=None, help='Experiment dataset')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone; clip-trained model should have the keywoard \"clip\"')
    parser.add_argument('--logpath', type=str, default='', help='Checkpoint saving dir identifier')
    parser.add_argument('--batchsize', type=int, default=256, help='Batch size')
    parser.add_argument('--shot', type=int, default=16, help='M shots')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--multemp', type=float, default=16., help='Multiplying temperature')
    parser.add_argument('--ik', type=int, default=16, help='K KNN')
    parser.add_argument('--tk', type=int, default=16, help='K KNN')
    parser.add_argument('--ntokens', type=int, default=0, help='Number of tokens')
    parser.add_argument('--maxepochs', type=int, default=1000, help='Max iterations')
    parser.add_argument('--nowandb', action='store_true', help='Flag not to log at wandb')
    parser.add_argument('--runfree', type=str, default=None, choices=['nakata22', 'naiveproto', 'clipzeroshot'], help="Run a model don't have any differentiable parameters")
    parser.add_argument('--eval', action='store_true', help='Flag for evaluation')
    parser.add_argument('--episodiceval', action='store_true', help='Flag for episodic evaluation')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from the last point of logpath')
    parser.add_argument('--usefewshot', action='store_true', help='use few-shot images')
    parser.add_argument('--jobid', type=int, default=0, help='Slurm job ID')

    args = parser.parse_args()

    if args.dataset == 'places365':
        args.datapath = os.path.join(args.datapath, 'places365')

    checkpoint_callback = CustomCheckpoint(args)
    dm = return_datamodule(args.datapath, args.dataset, args.batchsize, args.shot)
    model = MemoryModularLearnerTrainer(args, dm=dm)

    if args.runfree:
        if args.runfree == 'nakata22':
            model.learner.forward = model.learner.forward_nakata22
        elif args.runfree == 'naiveproto':
            model.learner.forward = model.learner.forward_naive_protomatching
        elif args.runfree == 'clipzeroshot':
            model.learner.forward = model.learner.forward_clipzeroshot

    trainer = Trainer(
        max_epochs=args.maxepochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=CSVLogger(save_dir='logs') if args.nowandb else WandbLogger(name=args.logpath, save_dir='logs', project=f'qamr-{args.dataset}-{args.backbone}', config=args),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10), checkpoint_callback],
        num_sanity_val_steps=0,
        resume_from_checkpoint=checkpoint_callback.lastmodelpath,
        # gradient_clip_val=5.0,
    )

    if args.episodiceval:
        modelpath = checkpoint_callback.modelpath
        model = MemoryModularLearnerTrainer.load_from_checkpoint(modelpath, args=args, dm=dm)
        nepisode = 600
        acc_list = torch.zeros(nepisode)
        for i in range(nepisode):
            seed_everything(i)
            testresult = trainer.test(model=model, datamodule=dm)
            acc = testresult[0]['tst/acc']
            acc_list[i] = acc
            avg_acc = torch.mean(acc_list[:i + 1])
            std_acc = torch.std(acc_list[:i + 1])
            ci = std_acc * 1.96 / math.sqrt(i + 1)  # 95% confidence interval
            print(f'{i + 1}/{nepisode} avg acc: {avg_acc} +- {ci}')
        exit()

    if args.eval:
        if not args.runfree:
            modelpath = checkpoint_callback.modelpath
            model = MemoryModularLearnerTrainer.load_from_checkpoint(modelpath, args=args, dm=dm)
        trainer.test(model=model, datamodule=dm)
    else:
        snapshot_dir = os.path.join('snapshots', args.logpath)
        with RsyncSnapshot(snapshot_dir=snapshot_dir):
            trainer.fit(model, dm)

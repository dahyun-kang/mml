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
from model.decoupled import Decoupled_learner
from model.tau_normalize import tau_normalizer
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
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--multemp', type=float, default=16., help='Multiplying temperature')
    parser.add_argument('--ik', type=int, default=16, help='K KNN')
    parser.add_argument('--tk', type=int, default=16, help='K KNN')
    parser.add_argument('--ntokens', type=int, default=0, help='Number of tokens')
    parser.add_argument('--maxepochs', type=int, default=1000, help='Max iterations')
    parser.add_argument('--nowandb', action='store_true', help='Flag not to log at wandb')
    # parser.add_argument('--nakata22', action='store_true', help='Flag to run Nataka et al., ECCV 2022')
    parser.add_argument('--runfree', type=str, default=None, choices=['nakata22', 'naiveproto', 'clipzeroshot'], help="Run a model don't have any differentiable parameters")
    parser.add_argument('--LT', action='store_true', help='Flag to run Longtailed Learning')
    parser.add_argument('--sampler', type=str, default=None, choices=['ClassAware', 'SquareRoot'], help='Choose your sampler for training')
    parser.add_argument('--Decoupled', type=str, default=None, choices=['joint', 'cRT', 'tau', 'feat_extract'], help='Flag to run reproducing expriement of Decoupled Learning')
    parser.add_argument('--eval', action='store_true', help='Flag for evaluation')
    parser.add_argument('--episodiceval', action='store_true', help='Flag for episodic evaluation')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from the last point of logpath')
    parser.add_argument('--usefewshot', action='store_true', help='use few-shot images')
    parser.add_argument('--jobid', type=int, default=0, help='Slurm job ID')

    args = parser.parse_args()

    args.many_shot_thr = 100
    args.low_shot_thr = 20

    if args.dataset == 'places365':
        args.datapath = os.path.join(args.datapath, 'places365')

    # Only for reproducing experiment of paper
    # Kang, B., Xie, S., Rohrbach, M., Yan, Z., Gordo, A., Feng, J., Kalantidis, Y.: Decoupling representation and classifier for long-tailed recognition. In: Proc. Int. Conf. Learn. Representations (2019)
    if args.Decoupled:
        checkpoint_callback = CustomCheckpoint(args)
        dm = return_datamodule(args.datapath, args.dataset, args.batchsize, args.backbone, args.sampler)

        if args.Decoupled == 'Joint':
            model = Decoupled_learner(args, dm=dm)
        else:
            modelpath = checkpoint_callback.modelpath
            model = Decoupled_learner.load_from_checkpoint(modelpath, args=args, dm=dm)

        if args.Decoupled == 'tau':
            dirpath = os.path.join('logs', args.dataset, args.backbone, args.logpath)
            train_file  = 'trainfeat_all.pkl'
            test_files   = ['valfeat_all.pkl', 'testfeat_all.pkl']

            for test_file in test_files:
                print(f'tau_normalize result of {test_file}')
                tau_trainer = tau_normalizer(args, model, dirpath, train_file, test_file)
                log = []
                for tau in np.linspace(0,2,2001):
                    result = tau_trainer.test(tau, log=False)
                    log.append([tau]+result)

                log = np.array(log)
                maxidx = np.argmax(log, axis=0)

                stages = ['top1', 'many', 'medium', 'few']
                for idx, stage in zip(maxidx[1:], stages):
                    print(f'Tau for Top {stage}_acc \t| tau: {log[idx][0]:.3f} | all: {log[idx][1]:.2f} | many: {log[idx][2]:.2f} | medium: {log[idx][3]:.2f} | few: {log[idx][4]:.2f}')



        else:
            trainer = Trainer(
                max_epochs=args.maxepochs,
                accelerator="auto",
                devices=1 if torch.cuda.is_available() else None,
                logger=CSVLogger(save_dir='logs') if args.nowandb else WandbLogger(name=args.logpath, save_dir='logs', project=f'qamr-{args.dataset}-{args.backbone}'),
                callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10), checkpoint_callback],
                num_sanity_val_steps=0,
                resume_from_checkpoint=checkpoint_callback.lastmodelpath,
                # gradient_clip_val=5.0,
            )

            if args.eval:
                modelpath = checkpoint_callback.modelpath
                model = Decoupled_learner.load_from_checkpoint(modelpath, args=args, dm=dm)
                trainer.test(model=model, datamodule=dm)
            elif args.Decoupled == 'feat_extract':
                phases = [['test', dm.test_dataloader()], ['val', dm.val_dataloader()], ['train', dm.unshuffled_train_dataloader()]]
                for phase in phases:
                    print(f"\nFeature(backbone) extract from {phase[0]} dataloader")
                    model.feat_extract_phase = phase[0]
                    trainer.test(model=model, dataloaders=phase[1])
            else:
                trainer.fit(model, dm)

    else:
        checkpoint_callback = CustomCheckpoint(args)
        dm = return_datamodule(args.datapath, args.dataset, args.batchsize, args.backbone, args.sampler)
        model = MemoryModularLearnerTrainer(args, dm=dm)

        # if args.nakata22:
        #     model.forward = model.forward_nakata22
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

        # if args.nakata22:
        #     # non-differentiable majority voting method, Nakata et al., ECCV 2022
        #     trainer.test(model, datamodule=dm)
        if args.runfree and not args.episodiceval:
            trainer.test(model, datamodule=dm)
        elif args.eval:
            modelpath = checkpoint_callback.modelpath
            model = MemoryModularLearnerTrainer.load_from_checkpoint(modelpath, args=args, dm=dm)
            trainer.test(model=model, datamodule=dm)
        elif args.episodiceval:
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
        else:
            snapshot_dir = os.path.join('snapshots', args.logpath)
            with RsyncSnapshot(snapshot_dir=snapshot_dir):
                trainer.fit(model, dm)

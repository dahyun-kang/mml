import os
import pickle
import numpy as np
import torch

def taunorm(weights, tau):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], tau)
    return ws
    
def dotproduct_similarity(A, B):
    AB = torch.mm(A, B.t())
    return AB

def logits2preds(logits, labels):
    _, nns = logits.max(dim=1)
    preds = np.array([labels[i] for i in nns])
    return preds
class tau_normalizer():
    def __init__(self, args, model, root, train_feat_path, test_feat_path):
        self.args = args
        self.model = model

        self.weight = model.classifier.weight
        self.bias = model.classifier.bias

        self.root = root
        self.train_feat_path = train_feat_path
        self.test_feat_path = test_feat_path


        with open(os.path.join(self.root, self.train_feat_path), 'rb') as f:    
            self.trainset = pickle.load(f)
        with open(os.path.join(self.root, self.test_feat_path), 'rb') as f:     
            self.testset = pickle.load(f)

        self.testsize = len(self.testset['feats'])

        c_labels = []
        for i in np.unique(self.trainset['labels']):
            c_labels.append(i)
        self.c_labels = np.array(c_labels)

    def preds2accs(self, preds):
        testlabel = self.testset['labels']
        trainlabel = self.trainset['labels']

        trainlabel = np.array(trainlabel).astype(int)

        # top1 acc
        top1_all = (preds == testlabel).sum().item() / len(testlabel)

        # many, medium, few acc
        train_class_count = []
        test_class_count = []
        class_correct = []
        for l in np.unique(testlabel):
            train_class_count.append(len(trainlabel[trainlabel == l]))
            test_class_count.append(len(testlabel[testlabel == l]))
            class_correct.append((preds[testlabel == l] == testlabel[testlabel == l]).sum())

        many_shot = []
        medium_shot = []
        few_shot = []

        for i in range(len(train_class_count)):
            if train_class_count[i] > self.args.many_shot_thr:
                many_shot.append((class_correct[i] / test_class_count[i]))
            elif train_class_count[i] < self.args.low_shot_thr:
                few_shot.append((class_correct[i] / test_class_count[i]))
            else:
                medium_shot.append((class_correct[i] / test_class_count[i]))  

        if len(many_shot) == 0: many_shot.append(0)
        if len(medium_shot) == 0: medium_shot.append(0)
        if len(few_shot) == 0: few_shot.append(0)

        top1_acc = top1_all*100.
        many_acc = np.mean(many_shot)*100.
        medium_acc = np.mean(medium_shot)*100.
        few_acc = np.mean(few_shot)*100.

        result = f"all: {top1_acc:.2f} | many: {many_acc:.2f} | medium: {medium_acc:.2f} | few: {few_acc:.2f}"
        return result, top1_acc, many_acc, medium_acc, few_acc
        
    def forward(self, weights):
        total_logits = []
        for i in range(self.testsize // self.args.batchsize + 1):
            # if i%10 == 0:
            #     print('{}/{}'.format(i, testsize // batch_size + 1))
            feat = self.testset['feats'][self.args.batchsize*i:min(self.args.batchsize*(i+1), self.testsize)]
            feat = torch.Tensor(feat)

            logits = dotproduct_similarity(feat, weights)
            total_logits.append(logits)

        total_logits = torch.cat(total_logits)
        return total_logits
    
    def test(self, tau, log=True):
        ws = taunorm(self.weight, tau)
        logits = self.forward(ws)
        preds = logits2preds(logits, self.c_labels)
        result, top1, many, medium, few = self.preds2accs(preds)

        if log:
            print(f'tau: {tau:.3f} | ' + result)
        return [top1, many, medium, few]

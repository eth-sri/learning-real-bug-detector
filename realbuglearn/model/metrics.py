import torch

class MyMetrics:
    def __init__(self, task, target):
        self.task = task
        self.target = target

        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        self.tp_loc = 0
        self.fp_loc = 0
        self.tp_tgt = 0
        self.fp_tgt = 0

    def add(self, outputs, inputs):
        logits_cls, logits_loc, logits_tgt = outputs.logits_cls, outputs.logits_loc, outputs.logits_tgt

        if self.target == 'cls':
            predicteds, labels = torch.argmax(logits_cls, dim=1), inputs['cls_labels']
            for predicted, label in zip(predicteds, labels):
                if predicted == 0:
                    if label == 0:
                        self.tn += 1
                    else:
                        self.fn += 1
                else:
                    if label == 1:
                        self.tp += 1
                    else:
                        self.fp += 1
        elif self.target == 'loc':
            loc, loc_masks = torch.argmax(logits_loc, dim=1), inputs['loc_correct_masks']
            for i in range(loc.size(0)):
                if loc[i] == 0:
                    if loc_masks[i][0] == 1:
                        self.tn += 1
                    else:
                        self.fn += 1
                else:
                    if loc_masks[i][0] != 1:
                        self.tp += 1
                    else:
                        self.fp += 1

                    if loc_masks[i][0] != 1 and loc_masks[i][loc[i]] == 1:
                        self.tp_loc += 1
                    else:
                        self.fp_loc += 1
        elif self.target == 'loc-tgt':
            loc, loc_masks = torch.argmax(logits_loc, dim=1), inputs['loc_correct_masks']
            tgt, tgt_masks = torch.argmax(logits_tgt, dim=1), inputs['tgt_correct_masks']
            for i in range(loc.size(0)):
                if loc[i] == 0:
                    if loc_masks[i][0] == 1:
                        self.tn += 1
                    else:
                        self.fn += 1
                else:
                    if loc_masks[i][0] != 1:
                        self.tp += 1
                    else:
                        self.fp += 1

                    if loc_masks[i][0] != 1 and loc_masks[i][loc[i]] == 1:
                        self.tp_loc += 1
                    else:
                        self.fp_loc += 1

                    if loc_masks[i][0] != 1 and loc_masks[i][loc[i]] == 1 and tgt_masks[i][tgt[i]] == 1:
                        self.tp_tgt += 1
                    else:
                        self.fp_tgt += 1
        elif self.target == 'cls-locpos':
            predicteds, labels = torch.argmax(logits_cls, dim=1), inputs['cls_labels']
            loc, loc_masks = torch.argmax(logits_loc, dim=1), inputs['loc_correct_masks']

            for i in range(predicteds.size(0)):
                predicted, label = predicteds[i], labels[i]
                if predicted == 0:
                    if label == 0:
                        self.tn += 1
                    else:
                        self.fn += 1
                else:
                    if label == 1:
                        self.tp += 1
                    else:
                        self.fp += 1

                    if label == 1 and loc_masks[i][loc[i]] == 1:
                        self.tp_loc += 1
                    else:
                        self.fp_loc += 1
        elif self.target == 'cls-locpos-tgt':
            predicteds, labels = torch.argmax(logits_cls, dim=1), inputs['cls_labels']
            loc, loc_masks = torch.argmax(logits_loc, dim=1), inputs['loc_correct_masks']
            tgt, tgt_masks = torch.argmax(logits_tgt, dim=1), inputs['tgt_correct_masks']

            for i in range(predicteds.size(0)):
                predicted, label = predicteds[i], labels[i]
                if predicted == 0:
                    if label == 0:
                        self.tn += 1
                    else:
                        self.fn += 1
                else:
                    if label == 1:
                        self.tp += 1
                    else:
                        self.fp += 1

                    if label == 1 and loc_masks[i][loc[i]] == 1:
                        self.tp_loc += 1
                    else:
                        self.fp_loc += 1

                    if label == 1 and loc_masks[i][loc[i]] == 1 and tgt_masks[i][tgt[i]] == 1:
                        self.tp_tgt += 1
                    else:
                        self.fp_tgt += 1
        else:
            assert(False)

    @property
    def prec(self):
        prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) != 0 else 0
        return prec

    @property
    def recall(self):
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) != 0 else 0
        return recall

    @property
    def f1(self):
        p, r = self.prec, self.recall
        if p == 0 and r == 0:
            return 0
        else:
            return 2 * p * r / (p + r)

    @property
    def loc_prec(self):
        prec = self.tp_loc / (self.tp_loc + self.fp_loc) if (self.tp_loc + self.fp_loc) != 0 else 0
        return prec

    @property
    def loc_recall(self):
        recall = self.tp_loc / (self.tp + self.fn) if (self.tp_loc + self.fn) != 0 else 0
        return recall

    @property
    def loc_f1(self):
        p, r = self.loc_prec, self.loc_recall
        if p == 0 and r == 0:
            return 0
        else:
            return 2 * p * r / (p + r)

    @property
    def tgt_prec(self):
        prec = self.tp_tgt / (self.tp_tgt + self.fp_tgt) if (self.tp_tgt + self.fp_tgt) != 0 else 0
        return prec

    @property
    def tgt_recall(self):
        recall = self.tp_tgt / (self.tp + self.fn) if (self.tp_tgt + self.fn) != 0 else 0
        return recall

    @property
    def tgt_f1(self):
        p, r = self.tgt_prec, self.tgt_recall
        if p == 0 and r == 0:
            return 0
        else:
            return 2 * p * r / (p + r)

    def __str__(self):
        pos = self.tp + self.fn
        neg = self.tn + self.fp
        total = pos + neg
        overall = f'total {total}, pos {pos}, neg {neg}'

        if self.target == 'cls':
            cls = f'cls_prec {self.prec:.4f}, cls_recall {self.recall:.4f}, cls_f1 {self.f1:.4f}'
            return ', '.join([overall, cls])
        elif self.target == 'loc':
            cls = f'cls_prec {self.prec:.4f}, cls_recall {self.recall:.4f}, cls_f1 {self.f1:.4f}'
            loc = f'loc_prec {self.loc_prec:.4f}, loc_recall {self.loc_recall:.4f}, loc_f1 {self.loc_f1:.4f}'
            return ', '.join([overall, cls, loc])
        elif self.target == 'loc-tgt':
            cls = f'cls_prec {self.prec:.4f}, cls_recall {self.recall:.4f}, cls_f1 {self.f1:.4f}'
            loc = f'loc_prec {self.loc_prec:.4f}, loc_recall {self.loc_recall:.4f}, loc_f1 {self.loc_f1:.4f}'
            tgt = f'tgt_prec {self.tgt_prec:.4f}, tgt_recall {self.tgt_recall:.4f}, tgt_f1 {self.tgt_f1:.4f}'
            return ', '.join([overall, cls, loc, tgt])
        elif self.target == 'cls-locpos':
            cls = f'cls_prec {self.prec:.4f}, cls_recall {self.recall:.4f}, cls_f1 {self.f1:.4f}'
            loc = f'loc_prec {self.loc_prec:.4f}, loc_recall {self.loc_recall:.4f}, loc_f1 {self.loc_f1:.4f}'
            return ', '.join([overall, cls, loc])
        elif self.target == 'cls-locpos-tgt':
            cls = f'cls_prec {self.prec:.4f}, cls_recall {self.recall:.4f}, cls_f1 {self.f1:.4f}'
            loc = f'loc_prec {self.loc_prec:.4f}, loc_recall {self.loc_recall:.4f}, loc_f1 {self.loc_f1:.4f}'
            tgt = f'tgt_prec {self.tgt_prec:.4f}, tgt_recall {self.tgt_recall:.4f}, tgt_f1 {self.tgt_f1:.4f}'
            return ', '.join([overall, cls, loc, tgt])
        else:
            assert(False)

import torch
import random
import pickle
from sklearn.utils import shuffle

from .sparse_sample import desparsify_sample

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, d=None):
        if d is None:
            with open(data_path, 'rb') as f:
                d = pickle.load(f)
        self.data = d

    def stat(self):
        total, pos, neg, repos = 0, 0, 0, set()
        if 'cls_labels' in self.data:
            for i in range(len(self.data['cls_labels'])):
                total += 1
                if self.data['cls_labels'][i] == 1:
                    pos += 1
                else:
                    neg += 1
                repos.add((self.data['user'][i], self.data['repo'][i]))
        else:
            for i in range(len(self.data['input_ids1'])):
                total += 2
                pos += 1
                neg += 1
                repos.add((self.data['user1'][i], self.data['repo1'][i]))
        return total, pos, neg, len(repos)

    def subset_percent_by_repo(self, percent):
        if 'user' in self.data:
            user_key, repo_key = 'user', 'repo'
        else:
            user_key, repo_key = 'user1', 'repo1'

        all_repos = set()
        for i in range(len(self.data[user_key])):
            all_repos.add((self.data[user_key][i], self.data[repo_key][i]))

        all_repos = list(sorted(all_repos))
        random.shuffle(all_repos)
        subset_repos = set(all_repos[:round(len(all_repos)/100*percent)])

        new_data = dict()
        for i in range(len(self.data[user_key])):
            if (self.data[user_key][i], self.data[repo_key][i]) not in subset_repos: continue
            for key in self.data:
                if key not in new_data:
                    new_data[key] = list()
                new_data[key].append(self.data[key][i])

        self.data = new_data

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.data.items()}
        return item

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])

    def shuffle(self):
        keys = list(sorted(self.data.keys()))
        vals = list()
        for key in keys:
            vals.append(self.data[key])
        vals = shuffle(*vals)
        for key, val in zip(keys, vals):
            self.data[key] = val

    def subset(self, num):
        old_data = self.data
        self.data = dict()
        for key in old_data:
            self.data[key] = old_data[key][:num]

    def prepare(self, target):
        if target == 'cls':
            self.pop('loc_candidate_masks')
            self.pop('loc_correct_masks')
            self.pop('tgt_candidate_masks')
            self.pop('tgt_correct_masks')
        elif target == 'loc':
            self.prepare_loc()
            self.pop('cls_labels')
            self.pop('tgt_candidate_masks')
            self.pop('tgt_correct_masks')
        elif target == 'loc-tgt':
            self.prepare_loc()
            self.pop('cls_labels')
        elif target == 'cls-locpos':
            self.pop('tgt_candidate_masks')
            self.pop('tgt_correct_masks')
        elif target == 'cls-locpos-tgt':
            pass
        elif target == 'cls-locpos-tgt-contrastive':
            pass
        else:
            assert(False)

    def prepare_loc(self):
        for i in range(len(self.data['cls_labels'])):
            self.data['loc_candidate_masks'][i]['_sparse'] = [0] + self.data['loc_candidate_masks'][i]['_sparse']
            if self.data['cls_labels'][i] == 0:
                self.data['loc_correct_masks'][i]['_sparse'] = [0] + self.data['loc_correct_masks'][i]['_sparse']

    def append(self, other):
        for key in other.data:
            self.data[key] += other.data[key]

    def pop(self, key, default=None):
        self.data.pop(key, default)

    def only_pos(self):
        self.only_label(1)

    def only_neg(self):
        self.only_label(0)

    def only_label(self, label):
        new_data = dict()
        for key in self.data:
            new_data[key] = list()

        for i in range(len(self.data['cls_labels'])):
            if self.data['cls_labels'][i] != label: continue

            for key in self.data:
                new_data[key].append(self.data[key][i])

        self.data = new_data

    def neg_pos_ratio(self, ratio):
        total_pos, total_neg = 0, 0
        for i in range(len(self.data['cls_labels'])):
            if self.data['cls_labels'][i] == 1:
                total_pos += 1
            else:
                total_neg += 1

        neg, new_data = 0, dict()
        for key in self.data:
            new_data[key] = list()
        for i in range(len(self.data['cls_labels'])):
            if self.data['cls_labels'][i] != 1:
                neg += 1
                if neg > total_pos * ratio:
                    continue
            for key in self.data:
                new_data[key].append(self.data[key][i])

        self.data = new_data

class MyDataCollator:
    NO_EFFECT_KEYS = {
        'user', 'repo', 'path', 'code', 'op', 'char_starts', 'char_ends',
        'user1', 'repo1', 'path1', 'code1', 'op1', 'char_starts1', 'char_ends1',
        'user2', 'repo2', 'path2', 'code2', 'op2', 'char_starts2', 'char_ends2'
    }

    def __init__(self):
        pass

    def __call__(self, samples):
        res = dict()
        for key in samples[0]:
            val = list()
            for sample in samples:
                sample = desparsify_sample(sample)
                if key.startswith('tgt_candidate_masks'):
                    val.append(sample[key])
                elif key in self.NO_EFFECT_KEYS:
                    val.append(sample[key])
                elif key.startswith('cls_labels'):
                    val.append(sample[key])
                else:
                    val.append(torch.LongTensor(sample[key]))
            if key in self.NO_EFFECT_KEYS:
                res[key] = val
            elif key.startswith('cls_labels'):
                res[key] = torch.LongTensor(val)
            elif key.startswith('tgt_candidate_masks'):
                res[key] = val
            else:
                res[key] = torch.stack(val)
        return res

def stack_output(outputs, new_outputs):
    for key, val in new_outputs.items():
        if key not in outputs:
            outputs[key] = list()
        outputs[key] += new_outputs[key]

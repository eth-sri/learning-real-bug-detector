import torch
import pickle
import argparse
from sklearn.metrics import PrecisionRecallDisplay

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_file', dest='res_file', type=str, required=True)
    args = parser.parse_args()
    return args

args = get_args()

def main():
    with open(args.res_file, 'rb') as f:
        res = pickle.load(f)
    labels = res['cls_labels']
    cls_pred = torch.tensor(res['cls'])
    cls_prob = torch.tensor(res['cls_prob'])
    cls_prob = cls_pred * cls_prob + (1 - cls_pred) * (1 - cls_prob)
    pr = PrecisionRecallDisplay.from_predictions(labels, cls_prob)
    print(pr.average_precision)

if __name__ == '__main__':
    main()
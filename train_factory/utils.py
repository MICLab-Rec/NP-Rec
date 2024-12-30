import copy
import math
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import Sampler


def get_train_set(dataset, max_len):
    train_set = copy.copy(dataset)
    train_seq = []
    for user in range(len(dataset)):
        seq = dataset[user]
        for i in reversed(range(len(seq))):
            if not seq[i][-1]:
                break
        n_sample = math.floor((i + max_len - 1) / max_len)
        for k in range(n_sample):
            if (i - k * max_len) > max_len:
                trg = seq[i - (k + 1) * max_len: i - k * max_len]
                src = seq[i - (k + 1) * max_len - 1: i - k * max_len - 1]
                train_seq.append((src, trg))
            else:
                break
    train_set.user_seq = train_seq
    return train_set


def get_test_set(dataset, max_len):
    test_set = copy.copy(dataset)
    test_seq = []
    for user in range(len(dataset)):
        seq = dataset[user]
        for i in reversed(range(len(seq))):
            if not seq[i][-1]:
                break
        trg = seq[i: i + 1]
        ctx = seq[max(1, i - max_len + 1) : i]
        src = seq[max(0, i - max_len): i]
        test_seq.append((src, trg, ctx))
    test_set.user_seq = test_seq
    return test_set


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def pad_sequence(seq, max_len):
    seq = list(seq)
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[-max_len:]
    return torch.tensor(seq)


class LadderSampler(Sampler):
    def __init__(self, data_source, batch_size, fix_order=False):
        super(LadderSampler, self).__init__(data_source)
        self.data = [len(e[0]) for e in data_source]
        self.batch_size = batch_size * 100
        self.fix_order = fix_order

    def __iter__(self):
        if self.fix_order:
            d = zip(self.data, np.arange(len(self.data)), np.arange(len(self.data)))
        else:
            d = zip(self.data, np.random.permutation(len(self.data)), np.arange(len(self.data)))
        d = sorted(d, key=lambda e: (e[1] // self.batch_size, e[0]), reverse=True)
        return iter(e[2] for e in d)

    def __len__(self):
        return len(self.data)


def gen_train_batch(batch, data_source, max_len):
    src_seq, trg_seq = zip(*batch)
    items, data_size = [], []
    for e in src_seq:
        _, i_, _ = zip(*e)
        items.append(pad_sequence(i_, max_len))
        data_size.append(len(_))
    src_items = torch.stack(items)
    data_size = torch.tensor(data_size)
    items = []
    for e in trg_seq:
        _, i_, _ = zip(*e)
        items.append(pad_sequence(i_, max_len))
    trg_items = torch.stack(items)
    return src_items, trg_items, data_size


def gen_eval_batch(batch, data_source, max_len):
    src_seq, trg_seq, ctx_seq = zip(*batch)
    items, data_size = [], []
    for e in src_seq:
        _, i_, _ = zip(*e)
        items.append(pad_sequence(i_, max_len))
        data_size.append(len(_))
    src_items = torch.stack(items)
    data_size = torch.tensor(data_size)
    items = []
    for e in trg_seq:
        _, i_, _ = zip(*e)
        items.append(pad_sequence(i_, 1))
    trg_items = torch.stack(items)
    items = []
    for e in ctx_seq:
        _, i_, _ = zip(*e)
        items.append(pad_sequence(i_, max_len))
    ctx_items = torch.stack(items)
    return src_items, ctx_items, trg_items, data_size


def count(logits, trg, cnt):
    output = logits.clone()
    for i in range(trg.size(0)):
        output[i][0] = logits[i][trg[i]]
        output[i][trg[i]] = logits[i][0]
    idx = output.sort(descending=True, dim=-1)[1]
    order = idx.topk(k=1, dim=-1, largest=False)[1]
    cnt.update(order.squeeze().tolist())
    return cnt


def calculate(cnt, array):
    for k, v in cnt.items():
        array[k] = v
    hr = array.cumsum()
    ndcg = 1 / np.log2(np.arange(0, len(array)) + 2)
    ndcg = ndcg * array
    ndcg = ndcg.cumsum() / hr.max()
    hr = hr / hr.max()
    return hr, ndcg
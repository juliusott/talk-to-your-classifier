import torch
import sys

def get_batch(x, vocab, device="cpu"):
    go_x, x_eos = [], []
    max_len = max([len(s) for s in x])
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        go_x.append([vocab.go] + s_idx + padding)
        x_eos.append(s_idx + [vocab.eos] + padding)
    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch


def get_batch_new(x, vocab):
    go_x, x_eos = [], []
    max_len = 30
    w_old = ""
    s_idx= []
    for s in x:
        for w in s:
            if w_old == w:
                break
            elif w in vocab.word2idx:
                s_idx.append(vocab.word2idx[w])
            else:
                s_idx.append(vocab.unk)
            
            w_old = w
        padding = [vocab.pad] * (max_len - len(s_idx))
        go_x.append([vocab.go] + s_idx + padding)
        x_eos.append(s_idx + [vocab.eos] + padding)
    assert len(go_x[0]) == max_len +1, f" max len is 30, s_idx {len(s_idx)}, pad {len(padding)}"
    return torch.LongTensor(go_x).t().contiguous(), \
           torch.LongTensor(x_eos).t().contiguous() # time * batch

def get_batches(data, vocab, batch_size, device=None):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]))
    order, data = zip(*z)

    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        batches.append(get_batch_new(data[i: j], vocab))
        i = j
    return batches, order

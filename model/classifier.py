import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_dim, vocab_size, softmax=False):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(input_dim, vocab_size, bias=False)
        self.softmax = softmax

    def forward(self, x, mode='training'):
        logits = self.classifier(x)
        if mode == 'training':
            if self.softmax:
                if self.softmax:
                    probs = F.softmax(logits, dim=-1)
                    return probs
                else:
                    return logits
        else:
            logits = logits.mean(dim=2)
            if self.softmax:
                probs = F.softmax(logits, dim=-1)
                return probs
            else:
                return logits

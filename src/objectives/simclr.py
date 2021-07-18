import math
import torch
import numpy as np

from src.utils.utils import l2_normalize
import torch


class SimCLRObjective(torch.nn.Module):

    def __init__(self, outputs1, outputs2, t, push_only=False):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.t = t
        self.push_only = push_only

    def get_loss_and_acc(self):
        batch_size = self.outputs1.size(0)  # batch_size x out_dim
        witness_score = torch.sum(self.outputs1 * self.outputs2, dim=-1)
        outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        # [num_examples, 2 * num_examples]
        witness_norm_raw = self.outputs1 @ outputs12.T
        witness_norm = torch.logsumexp(
            witness_norm_raw / self.t, dim=1) - math.log(2 * batch_size)
        loss = -torch.mean(witness_score / self.t - witness_norm)

        # Witness score should be 2nd highest if correct (1st highest is the same example).
        accuracy = torch.isclose(witness_score, torch.topk(witness_norm_raw, 2, dim=-1).values[:, 1].float()).float().mean()
        return loss, accuracy
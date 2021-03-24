import math
import torch
import numpy as np

from src.utils.utils import l2_normalize


class NoiseConstrastiveEstimation(object):

    def __init__(self, indices, outputs, memory_bank, k=4096, t=0.07, m=0.5, **kwargs):
        self.k, self.t, self.m = k, t, m

        self.indices = indices.detach()
        self.outputs = l2_normalize(outputs, dim=1)

        self.memory_bank = memory_bank
        self.device = outputs.device
        self.data_len = memory_bank.size

    def updated_new_data_memory(self):
        data_memory = self.memory_bank.at_idxs(self.indices)
        new_data_memory = data_memory * self.m + (1 - self.m) * self.outputs
        return l2_normalize(new_data_memory, dim=1)

    def get_loss(self, *args, **kwargs):
        batch_size = self.indices.size(0)

        witness_score = self.memory_bank.get_dot_products(self.outputs, self.indices)

        noise_indx = torch.randint(0, self.data_len, (batch_size, self.k),
                                   device=self.device)  # U(0, data_len)
        noise_indx = noise_indx.long()
        witness_norm = self.memory_bank.get_dot_products(self.outputs, noise_indx)
        witness_norm = torch.logsumexp(witness_norm / self.t, dim=1) - math.log(self.k)

        loss = -torch.mean(witness_score / self.t - witness_norm)
        return loss

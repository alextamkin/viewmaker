import torch
import numpy as np

from src.utils.utils import l2_normalize


class MemoryBank(torch.nn.Module):
    """For efficiently computing the background vectors."""

    def __init__(self, size, dim, dtype=float):
        super().__init__()
        self.size = size
        self.dim = dim
        self.register_buffer('_bank', self._create())
        if dtype == int:
            self._bank = torch.zeros_like(self._bank, dtype=int)

    def _create(self):
        # initialize random weights
        mb_init = torch.rand(self.size, self.dim, requires_grad=False)
        std_dev = 1. / np.sqrt(self.dim / 3)
        mb_init = mb_init * (2 * std_dev) - std_dev
        # L2 normalise so that the norm is 1
        mb_init = l2_normalize(mb_init, dim=1)
        return mb_init

    def as_tensor(self):
        return self._bank

    def at_idxs(self, idxs):
        return torch.index_select(self._bank, 0, idxs)

    def get_all_distances(self, emb_batch):
        '''Returns a tensor of L2-distances between each given embedding and all the embeddings in the bank
        
        Args:
            emb_batch: [batch_size, emb_dim] Tensor of embeddings
        
        Returns:
            [batch_size, memory_bank_size] Tensor of L2-norm distances
        '''
        assert len(emb_batch.shape) == 2

        differences = self._bank.unsqueeze(0) - emb_batch.unsqueeze(1)
        # Broadcasted elementwise dot product.
        distances = torch.sqrt(torch.einsum('abc,abc->ab', differences, differences))
        return distances

    def get_all_dot_products(self, vec):
        # [bs, dim]
        assert len(vec.size()) == 2
        return torch.matmul(vec, torch.transpose(self._bank, 1, 0))

    def get_dot_products(self, vec, idxs):
        vec_shape = list(vec.size())    # [bs, dim]
        idxs_shape = list(idxs.size())  # [bs, ...]

        assert len(idxs_shape) in [1, 2]
        assert len(vec_shape) == 2
        assert vec_shape[0] == idxs_shape[0]

        if len(idxs_shape) == 1:
            with torch.no_grad():
                memory_vecs = torch.index_select(self._bank, 0, idxs)
                memory_vecs_shape = list(memory_vecs.size())
                assert memory_vecs_shape[0] == idxs_shape[0]
        else:  # len(idxs_shape) == 2
            with torch.no_grad():
                batch_size, k_dim = idxs.size(0), idxs.size(1)
                flat_idxs = idxs.view(-1)
                memory_vecs = torch.index_select(self._bank, 0, flat_idxs)
                memory_vecs = memory_vecs.view(batch_size, k_dim, self._bank.size(1))
                memory_vecs_shape = list(memory_vecs.size())

            vec_shape[1:1] = [1] * (len(idxs_shape) - 1)
            vec = vec.view(vec_shape)  # [bs, 1, dim]

        prods = memory_vecs * vec
        assert list(prods.size()) == memory_vecs_shape

        return torch.sum(prods, dim=-1)

    def update(self, indices, data_memory):
        # in lieu of scatter-update operation
        data_dim = data_memory.size(1)
        self._bank = self._bank.scatter(0, indices.unsqueeze(1).repeat(1, data_dim),
                                        data_memory.detach())

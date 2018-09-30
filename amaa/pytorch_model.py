import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EpisodicGRU(nn.Module):

    def __init__(self, input_size, hidden_size, layers=1):
        super().__init__()
        self.cell = nn.GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

        if layers != 1:
            raise NotImplementedError()

    def forward(self, x, att_weights):
        is_packed = isinstance(x, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            assert isinstance(att_weights, torch.nn.utils.rnn.PackedSequence)
            data, batch_sizes = x
            att_weight_data, att_weight_batch_sizes = att_weights
            assert torch.all(batch_sizes == att_weight_batch_sizes)
        else:
            raise NotImplementedError()

        # Note: Understanding the code below requires understanding
        # how PackedSequence works.
        #
        # The data in a PackedSequence is arranged so that you can do
        # seq[:batch_size] and get the first element in each sequence
        # for each batch. Meaning that if your original data looked
        # like:
        #
        # [[a, b, c],
        #  [d, e, 0],
        #  [f, 0, 0]]
        #
        # The packed data would be:
        #
        # [a, d, f, b, e, c]

        hidden = torch.zeros(batch_sizes[0], self.hidden_size)
        start_index = 0
        for batch_size in batch_sizes:
            batch = data[start_index:start_index + batch_size]
            weights = att_weight_data[start_index:start_index + batch_size]
            weights = weights.unsqueeze(1)
            old_hidden = hidden[:batch_size]
            new_hidden = self.cell(batch, old_hidden)
            new_hidden = (new_hidden * weights) + (1 - weights) * old_hidden
            hidden[:batch_size] = new_hidden
            start_index += batch_size

        return hidden
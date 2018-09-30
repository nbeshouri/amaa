import torch
import torch.nn as nn
from amaa.pytorch_model import EpisodicGRU


def test_attention_weights():

    # SETUP

    cell = EpisodicGRU(1, 3)
    att_weights = torch.tensor(
        [[1, 0, 1],
         [1, 1, 0],
         [0, 0, 0]],
        dtype=torch.float)
    x = torch.tensor(
        [[1, 2, 3],
         [1, 3, 0],
         [5, 0, 0]],
        dtype=torch.float)
    batch_sizes = [3, 2, 1]
    att_weights = nn.utils.rnn.pack_padded_sequence(
        att_weights, batch_sizes, batch_first=True)
    x = x.unsqueeze(2)  # Fake embeddings.
    x = nn.utils.rnn.pack_padded_sequence(x, batch_sizes, batch_first=True)
    output = cell(x, att_weights)

    # TESTING

    # The weights [1, 0, 1] applied to the first story mean that its
    # second sentence should effect the state at all and we should
    # get the same output from the cell as we would if that step
    # wasn't there (which is what the second story is).
    assert torch.all(output[0] == output[1])
    # No sentence in the last has any weight, so the resulting episode
    # vector should be zero.
    assert torch.all(output[2] == torch.zeros_like(output[2]))



import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from itertools import chain
import time
from . import data
from sklearn.metrics import accuracy_score


class EpisodicGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.cell = nn.GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

        if num_layers != 1:
            raise NotImplementedError()

    def forward(self, x, att_weights):

        assert len(x.size()) == 3
        assert len(att_weights.size()) == 2

        is_packed = isinstance(x, torch.nn.utils.rnn.PackedSequence)
        if is_packed:

            # TODO: To mirror the standard GRU layer, this should return
            # a packed sequence.

            assert isinstance(att_weights, torch.nn.utils.rnn.PackedSequence)
            data, batch_sizes = x
            att_weight_data, att_weight_batch_sizes = att_weights
            assert torch.all(batch_sizes == att_weight_batch_sizes)

            # Note: Understanding the code below requires understanding
            # how PackedSequence works.
            #
            # The data in a PackedSequence is arranged so that you can do
            # seq[:batch_size] and get the first element in each sequence
            # for the first batch. Meaning that if your original data looked
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

        hiddens = torch.zeros(x.size(0), x.size(1), self.hidden_size)
        old_hidden = torch.zeros(x.size(0), self.hidden_size)
        for word_i in range(x.size(1)):
            batch = x[:, word_i]
            weights = att_weights[:, word_i:word_i + 1]
            new_hidden = self.cell(batch, old_hidden)
            new_hidden = (new_hidden * weights) + (1 - weights) * old_hidden
            hiddens[:, word_i] = new_hidden
            old_hidden = new_hidden

        # PyTorch hidden states should have shape (layers, batch_size,
        # hidden_size) and old_hidden has shape (batch_size, hidden_size).
        # Right now, I'm only doing a single layer, but I'll need comeback
        # to this.
        state = old_hidden.unsqueeze(0)

        return hiddens, state


class DMNEncoder(nn.Module):

    def __init__(
            self, recur_size=128, embedding_matrix=None,
            train_embeddings=False, iterations=3):
        super().__init__()
        self.recur_size = recur_size
        self.iterations = iterations
        # Setup embedding lookup.
        if not isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = torch.FloatTensor(embedding_matrix)
        vocab_size = embedding_matrix.size(0)
        embedding_size = embedding_matrix.size(1)
        self.embeddings = nn.Embedding.from_pretrained(embedding_matrix, train_embeddings)

        # Setup story/question encoder.
        self.encoder = nn.GRU(
            input_size=embedding_size,
            hidden_size=recur_size,
            num_layers=1,
            batch_first=True)

        self.gate_linear = nn.Linear(
            in_features=recur_size * 4,
            out_features=1)

        self.ep_encoder = EpisodicGRU(
            input_size=recur_size,
            hidden_size=recur_size,
            num_layers=1)

        self.decoder = nn.GRU(
            input_size=recur_size,
            hidden_size=vocab_size,
            num_layers=1)

    @staticmethod
    def get_question_mask(x_questions):
        mask = torch.zeros_like(x_questions, dtype=torch.uint8)
        for question_i, question in enumerate(x_questions):
            for word_i, word in enumerate(question):
                if word_i == len(question) - 1 or question[word_i + 1] == 0:
                    mask[question_i, word_i] = 1
                    break
        return mask

    def forward(self, x_stories, x_story_masks, x_questions):

        # zero_state = torch.zeros(x_stories.size(0), self.recur_size)

        # Encode stories.
        x = self.embeddings(x_stories)
        # TODO: Can't this one be done with packed sequences too?
        encoded_stories, _ = self.encoder(x)
        sent_counts = np.sum(np.array(x_story_masks), axis=1)  # (batch_size,)
        masked_outputs = encoded_stories[x_story_masks]  # (num_sents_in_batch, recur_size)

        encoded_story_sents = torch.zeros(  # (batch_size, max_num_sents, recur_size)
            x_stories.size(0),
            max(sent_counts),
            self.recur_size,
            dtype=torch.float)

        # `masked_outputs` is a flat tensor of encoded sentences that
        # have been selected with boolean masking. Here we unravel those
        # a `(batch_size, max_num_sents, recur_size)` tensor for encoding
        # into episode vectors. They can't just be reshaped because the
        # size of the array is changing (stories are of different lengths).
        masked_sent_index = 0
        for story_id, sent_count in enumerate(sent_counts):
            for sent_id in range(sent_count):
                encoded_story_sents[story_id, sent_id] = masked_outputs[masked_sent_index]
                masked_sent_index += 1

        x = self.embeddings(x_questions)
        questions_vecs, _ = self.encoder(x)
        questions_mask = self.get_question_mask(x_questions)
        questions_vecs = questions_vecs[questions_mask]  # (batch_size, recur_size)
        padded_questions_vecs = questions_vecs.unsqueeze(1)  # (batch_size, 1, recur_size)

        # Get attention weights.
        pointwise1 = encoded_story_sents * padded_questions_vecs  # (batch_size, num_sents, recur_size)
        delta1 = (encoded_story_sents - padded_questions_vecs)**2
        memory_vecs = padded_questions_vecs

        for i in range(self.iterations):
            pointwise2 = encoded_story_sents * memory_vecs
            delta2 = (encoded_story_sents - memory_vecs)**2
            feature_vectors = torch.cat(  # (batch_size, max_sents, recur_size * 4)
                [pointwise1, pointwise2, delta1, delta2], dim=2)
            x = self.gate_linear(feature_vectors)  # (batch_size, max_sents, 1)
            gate_weights = torch.sigmoid(x)
            gate_weights = gate_weights.squeeze(2)  # (batch_size, max_sents)
            episode_vecs, ep_encoder_state = self.ep_encoder(encoded_story_sents, gate_weights)
            episode_vecs = episode_vecs[:, -1, :]
            # TODO: I'm just selecting the last timestep. I should
            # be selecting the value at the last timestep with input. Or
            # I could them into a packed sequence...

            # TODO: Here, I'm totally ignoring the separate memory
            # encoder. It should probably be optional.
            memory_vecs = episode_vecs.unsqueeze(1)  # (batch_size, 1, recur_size)
            mem_encoder_state = ep_encoder_state

        return questions_vecs, mem_encoder_state, gate_weights


class DMNDecoder(nn.Module):

    def __init__(self, embeddings, recur_size=128):
        super().__init__()
        vocab_size = embeddings.num_embeddings
        embedding_size = embeddings.weight.size(1)
        self.embeddings = embeddings

        self.decoder = nn.GRU(
            input_size=recur_size + embedding_size,
            hidden_size=recur_size,
            num_layers=1,
            batch_first=True)

        self.word_predictor = nn.Linear(
            recur_size,
            vocab_size)


    def forward(self, prev_words, question_vecs, memory_encoder_state):
        """

        Args:
            memory_vecs: (num_layers, batch_size, hidden_size)
            question_vecs: (batch_size, hidden_size)

        Returns:

        """
        prev_words = self.embeddings(prev_words)
        question_vecs = question_vecs.unsqueeze(1)
        repeated_question_vecs = question_vecs.expand(-1, prev_words.size(1), -1)
        decoder_inputs = torch.cat([prev_words, repeated_question_vecs], dim=2)
        decoder_outputs, state = self.decoder(decoder_inputs, memory_encoder_state)
        predicted_words = self.word_predictor(decoder_outputs)
        return predicted_words, state


class BabAIDataset(Dataset):

    def __init__(
            self, X_stories, X_story_masks,
            X_questions, y, X_decoder=None, y_att=None):
        self.X_stories = torch.LongTensor(X_stories)
        self.X_questions = torch.LongTensor(X_questions)
        self.X_story_masks = torch.ByteTensor(X_story_masks)
        self.X_decoder = torch.LongTensor(X_decoder)
        if y_att is not None:
            y_att = torch.FloatTensor(self.get_att_target(y_att, X_story_masks))
        self.y_att = y_att
        self.y = y

    def __len__(self):
        return len(self.X_stories)

    @staticmethod
    def get_att_target(sent_indices, story_masks):
        """
        Convert attention targets from lists of sentence indices to
        boolean masks.

        TODO: This really should be part of the data structure that
        defines the data, but the existing version of the masks was
        written assuming pre-padding.
        """
        sent_counts = np.sum(np.array(story_masks), axis=1, dtype=int)
        att = np.zeros((story_masks.shape[0], max(sent_counts)))
        for example_index, indicies in enumerate(sent_indices):
            for sent_index in indicies:
                att[example_index, sent_index] = 1
        return att

    def __getitem__(self, item):
        batch = [self.X_stories[item], self.X_story_masks[item], self.X_questions[item], self.y[item]]
        if self.X_decoder is not None:
            batch.append(self.X_decoder[item])
        if self.y_att is not None:
            batch.append(self.y_att[item])
        return batch


# def debug(data_munch):
#     index = 0
#     example_story = data_munch.X_train_stories[index]
#     example_question = data_munch.X_train_questions[index]
#     example_hint = data_munch.y_train_att[index]
#     print(data.ids_to_text(example_story, data_munch.id_to_word))
#     print(data.ids_to_text(example_question, data_munch.id_to_word))
#     print(example_hint)
#     print(data_munch.train_hints[index])
#     print(data_munch.X_train_story_masks[index])


def debug(data_munch, dataset, id_to_word):
    index = 2
    example_story = dataset.X_stories[index]
    example_question = dataset.X_questions[index]
    print(data.ids_to_text(np.array(example_story), id_to_word))
    print(data.ids_to_text(np.array(example_question), id_to_word))
    print(dataset.y_att[index])
    print(data_munch.train_hints[index])


def predict(encoder_model, decoder_model, stories, questions):
    pass


def train(encoder_model, decoder_model, data, epochs, log_freq=1, gate_supervision=True):

    train_dataset = BabAIDataset(
        data.X_train_stories,
        data.X_train_story_masks,
        data.X_train_questions,
        data.y_train,
        data.X_train_decoder,
        data.train_hints)

    train_dataloader = DataLoader(train_dataset, batch_size=64)

    val_dataset = BabAIDataset(
        data.X_val_stories,
        data.X_val_story_masks,
        data.X_val_questions,
        data.y_val,
        data.X_val_decoder,
        data.val_hints)

    val_dataloader = DataLoader(val_dataset)

    # debug(data, train_dataset, data.id_to_word)
    # return

    criterion = nn.CrossEntropyLoss()
    att_criterion = nn.BCELoss()

    # The embedding layer is in both and is trainable, so make it's not
    # fed to the optimizer more than once.
    params = set(chain(encoder_model.parameters(), decoder_model.parameters()))
    optimizer = torch.optim.RMSprop(params)

    for epoch in range(epochs):
        print(f'Starting epoch {epoch}.')
        epoch_start_time = time.time()
        running_loss = 0.0
        running_accuracy = 0.0
        encoder_model.train()
        decoder_model.train()
        for i, train_batch in enumerate(train_dataloader):
            X_stories, X_story_masks, X_questions, y, X_decoder, y_att = train_batch

            optimizer.zero_grad()
            question_vecs, encoder_state, att_pred = encoder_model(X_stories, X_story_masks, X_questions)
            y_pred, _ = decoder_model(X_decoder, question_vecs, encoder_state)
            y_pred = y_pred.reshape(-1, y_pred.size(2))
            y = y.reshape(-1)
            decoder_loss = criterion(y_pred, y)

            # TODO: The next step is going to split the two losses so that you
            # can see the differences.

            total_loss = decoder_loss
            if gate_supervision:
                trunc_y_att = y_att[:, :att_pred.size(1)]
                att_loss = att_criterion(att_pred, trunc_y_att)
                # total_loss += att_loss
                total_loss = total_loss + att_loss

            accuracy = accuracy_score(y, y_pred.argmax(dim=-1))
            total_loss.backward()
            optimizer.step()

            # Periodically print dthe loss and prediction accuracy. Usually
            # with a language model you'd also show perplexity, but as it's
            # a function of our cross entropy loss and not that intuitive,
            # I've elected not to.
            running_loss += decoder_loss.item()
            running_accuracy += accuracy
            # if i % log_freq == log_freq - 1:
            # print((i + 1) % log_freq)
            if (i + 1) % log_freq == 0:
                average_loss = running_loss / log_freq
                average_accuracy = running_accuracy / log_freq
                log_str = f'Mini-batch: {i + 1}/{len(train_dataloader)} '
                if gate_supervision:
                    log_str += (f'Decoder loss: {decoder_loss:.5f} Attention Loss: {att_loss:.5f} '
                                f'Total loss: {total_loss:.5f} Accuracy: {average_accuracy:.5f}')
                else:
                    log_str += f'Loss: {average_loss:.5f} Accuracy: {average_accuracy:.5f}'
                print(log_str)
                running_loss = 0.0
                running_accuracy = 0.0

    # Log elapsed_time for the epoch.
    elapsed_time = time.time() - epoch_start_time
    print(f'Epoch {epoch} completed in {elapsed_time // 60:.0f} minutes '
          f'{elapsed_time % 60:.0f} seconds.\n')

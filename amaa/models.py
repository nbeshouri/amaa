import os
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Dense,
    Input,
    Embedding,
    Bidirectional,
    GRU,
    concatenate,
    RepeatVector,
    Lambda, 
    TimeDistributed,
    Flatten,
    Permute
)
from keras import backend as K
import numpy as np
import socket
import keras.layers as layers
import pandas as pd
from . import data
from .layers import EpisodicGRU

data_dir_path = os.path.join(os.path.dirname(__file__), 'data')


#
# BASELINE MODEL
#


def build_baseline_model(story_length, question_length, answer_length, 
                         embedding_matrix, recur_size=256, recurrent_layers=1):
    """
    Build and return the baseline sequence-to-sequence model.

    In this model, the both the story and the question are encoded using
    the same multi-layer GRU based encoder. The state of the encoder is
    then passed off to a similar decoder which takes as input the question
    vector and the previous word in the answer and predicts the next word
    in the answer.

    Args:
        story_length: The number of tokens in each story.
        question_length: The number of tokens in each question.
        answer_length: The number of tokens in each answer.
        embedding_matrix: A numpy matrix with shape 
            `(vocab_size, embedding_size)`
        recur_size: The size of the hidden space used by the recurrent
            layers.
        recurrent_layers: The number of stacked recurrent layers to use.

    Returns:
        (tuple): tuple containing:
            train_model: The Keras model used to train the weights.
            encoder_model: The Keras model used to encode input during
                prediction.
            decoder_model: The Keras model used to decode input during
                prediction.

    """
    story_input = Input(shape=(story_length,), name='story_input')
    question_input = Input(shape=(question_length,), name='question_input')
    
    decoder_input = Input(shape=(answer_length,), name='decoder_input')
    embedding_lookup = Embedding(embedding_matrix.shape[0],
                                 embedding_matrix.shape[1],
                                 weights=[embedding_matrix],
                                 trainable=False,
                                 name='embedding_lookup')
                                             
    encoders = [] 
    for i in range(recurrent_layers):
        return_seq = False if i == recurrent_layers - 1 else True
        encoder = GRU(recur_size // 2, return_sequences=return_seq, name=f'encoder{i}')
        encoder = Bidirectional(encoder)
        encoders.append(encoder)
                        
    decoders = []
    for i in range(recurrent_layers):
        decoder = GRU(
            recur_size, return_state=True, 
            return_sequences=True, name=f'decoder{i}')
        decoders.append(decoder)
    
    word_predictor = Dense(embedding_matrix.shape[0], activation='softmax', name='word_predictor')
    
    # Encode story.
    x = embedding_lookup(story_input)
    get_last_time_step = Lambda(lambda t: t[:, -1])
    encoder_states = []
    for encoder in encoders:
        x = encoder(x)  # <--- this is the line that causes seg fault.
        if encoder is not encoders[-1]:
            encoder_state = get_last_time_step(x)
        else:
            encoder_state = x
        encoder_states.append(encoder_state)
    
    # Encode question.
    
    # TODO: Find out if state is retained in recurrent layers for 
    # repeated uses within the same batch. If so, ww don't need to 
    # pass in these 0s here.
    x = embedding_lookup(question_input)
    zeros = K.zeros_like(encoder_states[-1])
    state_1 = K.slice(zeros, (0, 0), (-1, recur_size // 2))
    state_2 = K.slice(zeros, (0, recur_size // 2), (-1, -1))
    for encoder in encoders:
        x = encoder(x, initial_state=[state_1, state_2])
    encoded_question = x
    
    # Decode answer.
    x = embedding_lookup(decoder_input)
    repeated_question = RepeatVector(answer_length)(encoded_question)
    x = concatenate([x, repeated_question])
    for decoder, encoder_state in zip(decoders, encoder_states):
        x, _ = decoder(x, initial_state=encoder_state)
    
    outputs = TimeDistributed(word_predictor)(x)
    
    # Build training model.
    inputs = story_input, question_input, decoder_input
    train_model = Model(inputs=inputs, outputs=outputs)
    train_model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer='rmsprop', metrics=['accuracy'])
    
    # Build encoder model.
    inputs = story_input, question_input
    outputs = encoder_states + [encoded_question]
    encoder_model = Model(inputs=inputs, outputs=outputs)
    
    # Build decoder model.
    decoder_prev_predict_input = Input(shape=(1,), name='decoder_prev_predict_input')
    decoder_question_input = Input(shape=(recur_size,), name='decoder_question_input')
    decoder_state_inputs = [Input(shape=(recur_size,), name=f'decoder_state_input_{i}') 
                            for i in range(recurrent_layers)]    
    
    x = embedding_lookup(decoder_prev_predict_input)
    repeated_question = RepeatVector(1)(decoder_question_input)
    x = concatenate([x, repeated_question])
    
    decoder_states = []
    for decoder, decoder_state_input in zip(decoders, decoder_state_inputs):
        x, decoder_state = decoder(x, initial_state=decoder_state_input)
        decoder_states.append(decoder_state)
    x = word_predictor(x)
    
    inputs = [decoder_prev_predict_input, decoder_question_input] + decoder_state_inputs
    outputs = [x] + decoder_states
    decoder_model = Model(inputs=inputs, outputs=outputs)    
    
    return train_model, encoder_model, decoder_model


#
# DMN MODEL
#


def build_dmn_model(num_story_sentences, story_sentence_length, question_length,
                    answer_length, embedding_matrix, recur_size=256,
                    recurrent_layers=1, iterations=3, gate_dense_size=128,
                    use_mem_gru=False, gate_supervision=True,
                    return_att_model=False, reuse_ep_encoder_state=False,
                    apply_attention_to_hidden_state=True):
    """
    Build and return a Dynamic Memory Network.

    For details on this architecture, please see:
    https://arxiv.org/abs/1506.07285.

    Args:
        num_story_sentences: The number of sentences in the story.
        story_sentence_length: The number of tokens in each of those
            sentences.
        question_length: The number of tokens in each question.
        answer_length: The number of tokens in each answer.
        embedding_matrix: A numpy matrix with shape
            `(vocab_size, embedding_size)`
        recur_size: The size of the hidden space used by the recurrent
            layers.
        recurrent_layers: The number of stacked recurrent layers to use.
        iterations: The number of passes the episode generating GRU
            makes over the input.
        gate_dense_size: The number of hidden units in the dense network
             that generates the attention gate weights.
        use_mem_gru: Whether or not to consolidate the episodic memories
            into a memory vector that is retained after each iteration.
            This is how they did it in the paper, but I found the network
            converged faster without it. Defaults to `False`.
        gate_supervision: Whether to build the network so that the
            gate weights of the final iteration are part of the output
            and loss function for the training model. If this option is
            used, the correct gate weights must be passed in during
            training. Defaults to `True`.
        return_att_model: Return an additional model that can but used
            to get the attention weights during prediction. These
            you to debug/visualize the attention gate weights on
            arbitrary inputs.  Defaults to `False`.
        reuse_ep_encoder_state: Whether or not to reset the state of
            the episode generating GRU after each iteration. Defaults
            to `False`.
        apply_attention_to_hidden_state: If `True`, attention weights
            are applied to the hidden states of the episode generating
            GRU between timesteps. If a sentence has an attention
            weight of 1, then the hidden state resulting from that
            timestep is passed forward unmodified. If its weight is
            0, then the hidden state from the previous timestep is
            passed forward and the sentence at the current timestep has
            no impact on the network. Values between 0 and 1 will have
            intermediate effects.

            If `False`, then the attention weights are applied to the
            the sentence vectors directly before they're feed to the
            episode generating GRU.

            Defaults to `True` as this was the architecture used
            in the original paper.

    Returns:
        (tuple): tuple containing:
            train_model: The Keras model used to train the weights.
            encoder_model: The Keras model used to encode input during
                prediction.
            decoder_model: The Keras model used to decode input during
                prediction.
            attention_model: Optionally, a model that can output the
                attention gate weights for arbitrary input.

    """
    story_sent_inputs = [Input(shape=(story_sentence_length,), name=f'story_sentence_{i}_input')
                         for i in range(num_story_sentences)]
    question_input = Input(shape=(question_length,), name='question_input')
    decoder_input = Input(shape=(answer_length,), name='decoder_input')

    embedding_lookup = Embedding(embedding_matrix.shape[0],
                                 embedding_matrix.shape[1],
                                 weights=[embedding_matrix],
                                 trainable=False,
                                 name='embedding_lookup')

    #
    # Setup reused layers.
    #

    encoders = []
    for i in range(recurrent_layers):
        return_seq = False if i == recurrent_layers - 1 else True
        encoders.append(GRU(recur_size, return_sequences=return_seq,
                            return_state=True, name=f'encoder_{i}'))

    ep_encoders = []
    ep_gru_class = EpisodicGRU if apply_attention_to_hidden_state else GRU
    for i in range(recurrent_layers):
        return_seq = False if i == recurrent_layers - 1 else True
        ep_encoders.append(ep_gru_class(recur_size, return_sequences=return_seq,
                                        return_state=True, name=f'ep_encoder_{i}'))
    if use_mem_gru:
        mem_gru = GRU(recur_size)

    decoders = []
    for i in range(recurrent_layers):
        decoders.append(GRU(recur_size, return_state=True,
                            return_sequences=True, name=f'decoder_{i}'))

    gate_dense1 = Dense(gate_dense_size, activation='tanh')
    gate_dense2 = Dense(1, activation='sigmoid')  # Sigmoid makes more sense.
    word_predictor = Dense(embedding_matrix.shape[0], activation='softmax',
                           name='word_predictor')
    flatten = Flatten()
    repeat_recur_size_times = RepeatVector(recur_size)
    permute = Permute((2, 1))
    repeat_num_sents_times = RepeatVector(num_story_sentences)
    repeat_once = RepeatVector(1)

    #
    # Encode input.
    #

    # Encode story sents.
    initial_state = None
    encoded_sents = []
    for sent in story_sent_inputs:
        x = embedding_lookup(sent)
        for encoder in encoders:
            x, state = encoder(x, initial_state)
            if initial_state is None:
                initial_state = K.zeros_like(state)
        encoded_sents.append(x)

    # Merge encoded sents.
    reshaped_sents = [repeat_once(sent) for sent in encoded_sents]
    merged_sents = concatenate(reshaped_sents, axis=1)

    # Encode question.
    x = embedding_lookup(question_input)
    for encoder in encoders:
        x, _ = encoder(x, initial_state=initial_state)
    question_vector = x

    #
    # Generate memory vector.
    #

    per_layer_memory_vectors = [question_vector] * recur_size
    per_layer_episodes = [initial_state] * len(ep_encoders)
    question_vectors = repeat_num_sents_times(question_vector)
    pointwise1 = layers.multiply([merged_sents, question_vectors])
    delta1 = layers.subtract([merged_sents, question_vectors])
    delta1 = layers.multiply([delta1, delta1])

    attention_outputs = []
    for iteration in range(iterations):
        # Calculate the weights for the gate vectors.
        memory_vectors = repeat_num_sents_times(per_layer_memory_vectors[-1])
        pointwise2 = layers.multiply([merged_sents, memory_vectors])
        delta2 = layers.subtract([merged_sents, memory_vectors])
        delta2 = layers.multiply([delta2, delta2])

        gate_feature_vectors = concatenate([pointwise1, pointwise2, delta1, delta2])
        x = gate_dense1(gate_feature_vectors)
        attention_weights = gate_dense2(x)  # Shape: (None, num_story_sents, 1)
        flattened_attention_weights = flatten(attention_weights)  # Shape: (None, num_story_sents)
        attention_outputs.append(flattened_attention_weights)

        if apply_attention_to_hidden_state:
            ep_encoder_input = concatenate([merged_sents, attention_weights], axis=-1)
        else:
            x = repeat_recur_size_times(x)  # Shape: (None, recur_size, num_story_sents)
            gate_weights = permute(x)  # (None, num_story_sents, recur_size)
            ep_encoder_input = layers.multiply([merged_sents, gate_weights])

        new_per_layer_episodes = []
        x = ep_encoder_input
        for ep_encoder, prev_ep in zip(ep_encoders, per_layer_episodes):
            state = prev_ep if reuse_ep_encoder_state else initial_state
            x, episode = ep_encoder(x, initial_state=state)
            new_per_layer_episodes.append(episode)
        per_layer_episodes = new_per_layer_episodes

        # TODO: If you're using the multi-layer and mem_gru options
        # together, the way the mem_gru is resused here is not ideal.

        if use_mem_gru:
            new_per_layer_memory_vectors = []
            for memory_vector, episode in zip(per_layer_memory_vectors, per_layer_episodes):
                episode = repeat_once(episode)
                memory_vector = mem_gru(episode, initial_state=memory_vector)
                new_per_layer_memory_vectors.append(memory_vector)
            per_layer_memory_vectors = new_per_layer_memory_vectors
        else:
            per_layer_memory_vectors = per_layer_episodes

    # Decode answer.
    repeated_question = RepeatVector(answer_length)(question_vector)
    x = embedding_lookup(decoder_input)
    x = concatenate([x, repeated_question])
    for decoder, memory_vector in zip(decoders, per_layer_memory_vectors):
        x, _ = decoder(x, initial_state=memory_vector)
    answer_prediction = word_predictor(x)

    #
    # Build models.
    #

    # Build training model.
    inputs = story_sent_inputs + [question_input, decoder_input]
    outputs = [answer_prediction]
    losses = ['sparse_categorical_crossentropy']
    if gate_supervision:
        outputs.append(attention_outputs[-1])
        losses.append('binary_crossentropy')
    train_model = Model(inputs=inputs, outputs=outputs)
    train_model.compile(loss=losses, optimizer='rmsprop', metrics=['accuracy'])

    # Build encoder model.
    inputs = story_sent_inputs + [question_input]
    outputs = per_layer_memory_vectors + [question_vector]
    encoder_model = Model(inputs=inputs, outputs=outputs)

    # Build decoder model.
    decoder_prev_predict_input = Input(shape=(1,), name='decoder_prev_predict_input')
    decoder_question_input = Input(shape=(recur_size,), name='decoder_question_input')
    decoder_state_inputs = [Input(shape=(recur_size,), name=f'decoder_state_input_{i}')
                            for i in range(recurrent_layers)]

    x = embedding_lookup(decoder_prev_predict_input)
    repeated_question = repeat_once(decoder_question_input)
    x = concatenate([x, repeated_question])

    decoder_states = []
    for decoder, decoder_state_input in zip(decoders, decoder_state_inputs):
        x, decoder_state = decoder(x, initial_state=decoder_state_input)
        decoder_states.append(decoder_state)
    x = word_predictor(x)

    inputs = [decoder_prev_predict_input, decoder_question_input] + decoder_state_inputs
    outputs = [x] + decoder_states
    decoder_model = Model(inputs=inputs, outputs=outputs)

    models = [train_model, encoder_model, decoder_model]

    # Build attention mdoel.
    if return_att_model:
        inputs = story_sent_inputs + [question_input]
        att_model = Model(inputs=inputs, outputs=attention_outputs)
        models.append(att_model)

    return models


#
# SHARED UTILS
#


def model_is_sent_level(model):
    """Returns whether the model has different inputs for each story sentence."""
    for input in model.inputs:
        if 'sentence' in input.name:
            return True
    return False


def build_encoder_inputs(encoder_model, stories, questions):
    encoder_inputs = {}
    if model_is_sent_level(encoder_model):
        for i in range(stories.shape[1]):
            encoder_inputs[f'story_sentence_{i}_input'] = stories[:, i]
    else:
        encoder_inputs['story_input'] = stories
    encoder_inputs['question_input'] = questions
    return encoder_inputs
    

def predict(encoder_model, decoder_model, stories, questions,
            max_answer_length, story_masks=None):
    """Returns the predictions as indicies."""

    encoder_inputs = build_encoder_inputs(encoder_model, stories, questions)

    *encoded_stories, encoded_questions = encoder_model.predict(encoder_inputs)
    batch_size = encoded_stories[0].shape[0]
    preds = np.zeros((batch_size, 1)) + 2  # 2 == <START>
    decoder_states = encoded_stories
    pred_list = []
    for i in range(max_answer_length):

        decoder_inputs = {
            'decoder_prev_predict_input': preds,
            'decoder_question_input': encoded_questions
        }

        for j, decoder_input in enumerate(decoder_states):
            decoder_inputs[f'decoder_state_input_{j}'] = decoder_input

        preds, *decoder_states = decoder_model.predict(decoder_inputs)
        preds = np.argmax(preds, axis=-1)
        pred_list.append(preds)

    return np.concatenate(pred_list, axis=-1)
    
    
def get_attention_info(encoder_model, decoder_model, attention_model, 
                       story, question, answer, y_att, id_to_word):

    stories = story[np.newaxis, :]
    questions = question[np.newaxis, :]
    story_sent_strings = data.id_lists_to_texts(story, id_to_word)
    question_str = data.ids_to_text(question, id_to_word)
    answer_str = data.ids_to_text(answer, id_to_word)

    predicted_answer = predict(encoder_model, decoder_model, stories, questions, len(answer))
    predicted_answer_str = data.ids_to_text(predicted_answer[0], id_to_word)
    
    att_inputs = build_encoder_inputs(encoder_model, stories, questions)
    att_preds = attention_model.predict(att_inputs)
    att_preds = np.row_stack(att_preds)
    att_preds = np.swapaxes(att_preds, 0, 1)
    
    rows = []
    for story_sent, att_weights, y_att_for_sent in zip(story_sent_strings, att_preds, y_att):
        row = {'text': story_sent, 'y_att': y_att_for_sent}
        for iter, weight in enumerate(att_weights):
            row[f'iter_{iter + 1}_att'] = weight
        rows.append(row)
    
    columns = ['text'] + [k for k in rows[0] if 'iter' in k] + ['y_att']
    
    df = pd.DataFrame(rows, columns=columns)
    
    qa = pd.DataFrame(
        [question_str, answer_str, predicted_answer_str], 
        index=['question', 'answer', 'pred_answer'], 
        columns=[''])
    
    return df, qa
    

def debug_attention(encoder_model, decoder_model, attention_model, 
                    data_bunch, data_set, question_index, save_results=False):
    
    story = data_bunch[f'X_{data_set}_story_sents'][question_index]
    question = data_bunch[f'X_{data_set}_questions'][question_index]
    answer = data_bunch[f'y_{data_set}'][question_index]
    y_att = data_bunch[f'y_{data_set}_att'][question_index]
    
    att_info, qa_info = get_attention_info(
        encoder_model, decoder_model, attention_model, story, question,
        answer, y_att, data_bunch.id_to_word)
    
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_colwidth', 120)
    pd.set_option("display.max_columns", 101)
    
    print(att_info)
    print(qa_info)
    
    if save_results is not None:
        att_path = os.path.join(data_dir_path, 'attention_info.csv')
        att_info.to_csv(att_path)
        qa_path = os.path.join(data_dir_path, 'question_answer_info.csv')
        qa_info.to_csv(qa_path)


def train_model(model, train_data, epochs):
    checkpoint_path = os.path.join(data_dir_path, f'temp_weights_{socket.gethostname()}.hdf5')    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, 
                                   save_weights_only=True, monitor='val_loss')
    
    X_train = {
        'story_input': train_data.X_train_stories, 
        'question_input': train_data.X_train_questions, 
        'decoder_input': train_data.X_train_decoder
    }
    
    for i in range(train_data.X_train_story_sents.shape[1]):
        X_train[f'story_sentence_{i}_input'] = train_data.X_train_story_sents[:, i]
    
    X_val = {
        'story_input': train_data.X_val_stories, 
        'question_input': train_data.X_val_questions, 
        'decoder_input': train_data.X_val_decoder
    }
    
    for i in range(train_data.X_val_story_sents.shape[1]):
        X_val[f'story_sentence_{i}_input'] = train_data.X_val_story_sents[:, i]
    
    y_train = [train_data.y_train[:, :, np.newaxis]]
    y_val = [train_data.y_val[:, :, np.newaxis]]
        
    if len(model.outputs) > 1:
        if model_is_sent_level(model):
            y_train.append(train_data.y_train_att)
            y_val.append(train_data.y_val_att)
        else:
            y_train.append(train_data.X_train_hints)
            y_val.append(train_data.X_val_hints)
    
    # Train the model.
    model.fit(X_train, y_train, batch_size=64, epochs=epochs,
              validation_data=(X_val, y_val),
              callbacks=[checkpointer])

    # Load the best weights.
    model.load_weights(checkpoint_path)

    return model

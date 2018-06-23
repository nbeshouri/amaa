from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Dense,
    Input,
    Embedding,
    Bidirectional,
    GRU,
    Dropout,
    concatenate,
    RepeatVector,
    Lambda, 
    TimeDistributed,
    LSTM,
    Reshape,
    Flatten,
    Permute
)
from keras import backend as K
import os
import numpy as np
import socket
import keras.layers as layers


data_dir_path = os.path.join(os.path.dirname(__file__), 'data')


import sys
sys.setrecursionlimit(5000)


#
# BASELINE MODEL
#


def build_baseline_model(story_length, question_length, answer_length, 
                         embedding_matrix, recur_size=256, recurrent_layers=1):        
    
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
        encoders.append(Bidirectional(GRU(recur_size // 2, return_sequences=return_seq, 
                        name=f'encoder{i}')))
                        
    decoders = []
    for i in range(recurrent_layers):
        decoders.append(GRU(recur_size, return_state=True, return_sequences=True, name=f'decoder{i}'))
    
    word_predictor = Dense(embedding_matrix.shape[0], activation='softmax', name='word_predictor')
    
    # Encode story.
    x = embedding_lookup(story_input)
    get_last_time_step = Lambda(lambda t: t[:, -1,])
    encoder_states = []
    for encoder in encoders:
        x = encoder(x)  # <--- this is the line that causes seg fault.
        # Here I'm slicing off the last time step to feed as the initial
        # state to the decoder layers. I could get these back in the 
        # correct shape via return_state=True param, but that doesn't 
        # play nice with the bidirectional layer. Also, I'm assuming 
        # that in keras's GRU implementation, output and state are the 
        # same, which they are in the Colah blog post.
        if encoder is not encoders[-1]:
            encoder_state = get_last_time_step(x)
        else:
            encoder_state = x
        encoder_states.append(encoder_state)
    
    # Encode question.
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
    train_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    # Build encoder model.
    inputs = story_input, question_input
    outputs =  encoder_states + [encoded_question]
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
    # TODO: This accepts (?, ?, 256) and returns (?, 1, 10)... I'm sort of surprised
    # that it works... shouldn't I need a time Distributed? maybe because it's just one?
    x = word_predictor(x)
    
    
    inputs = [decoder_prev_predict_input, decoder_question_input] + decoder_state_inputs
    outputs = [x] + decoder_states
    decoder_model = Model(inputs=inputs, outputs=outputs)    
    
    return train_model, encoder_model, decoder_model


#
# DMN MODEL
#

    
def build_dmn_model(story_length, question_length, answer_length, 
                           embedding_matrix, recur_size=256, recurrent_layers=1,
                           iterations=3, gate_dense_size=128, use_mem_gru=False):
    
    story_input = Input(shape=(story_length,), name='story_input')
    question_input = Input(shape=(question_length,), name='question_input')
    decoder_input = Input(shape=(answer_length,), name='decoder_input')
    
    embedding_lookup = Embedding(embedding_matrix.shape[0],
                                 embedding_matrix.shape[1],
                                 weights=[embedding_matrix],
                                 trainable=False,
                                 name='embedding_lookup')
    
    # Declare major layers.
    input_gru = GRU(recur_size, return_sequences=True, return_state=True)
    ep_gru = GRU(recur_size)
    if use_mem_gru:
        mem_gru = GRU(recur_size)
    output_gru = GRU(recur_size, return_sequences=True, return_state=True)
    gate_dense1 = Dense(gate_dense_size, activation='tanh')
    gate_dense2 = Dense(1, activation='sigmoid')  # Sigmoid makes more sense.
    word_predictor = Dense(embedding_matrix.shape[0], activation='softmax', 
                           name='word_predictor')
    flatten = Flatten()
    repeat_stroy_len_times = RepeatVector(story_length)
    repeat_recur_size_times = RepeatVector(recur_size)
    permute = Permute((2, 1))
    repeat_once = RepeatVector(1)
    # sum_along_last = Lambda(lambda t: K.sum(t, axis=-1, keepdims=True))
    
    # Encode story.
    x = embedding_lookup(story_input)
    fact_vectors, state = input_gru(x)
    
    # Set all steps in story but those at the end of a sentence
    # to zeros.
    # x = repeat_recur_size_times(story_mask_input)
    # story_mask_weights = permute(x)
    # fact_vectors = layers.multiply([fact_vectors, story_mask_weights])
    
    # Encode question.
    x = embedding_lookup(question_input)
    _, question_vector = input_gru(x, initial_state=K.zeros_like(state))
    
    # Generate memory vector.
    memory_vector = question_vector    
    question_vectors = repeat_stroy_len_times(question_vector)
    pointwise1 = layers.multiply([fact_vectors, question_vectors])
    delta1 = layers.subtract([fact_vectors, question_vectors])
    delta1 = layers.multiply([delta1, delta1])
    
    for iteration in range(iterations):
        # Calculate the weights for the gate vectors.
        memory_vectors = repeat_stroy_len_times(memory_vector)
        
        pointwise2 = layers.multiply([fact_vectors, memory_vectors])
        delta2 = layers.subtract([fact_vectors, memory_vectors])
        delta2 = layers.multiply([delta2, delta2])
        
        # NOTE: This don't help, or at least isn't necessary.
        # dot1 = sum_along_last(layers.multiply([fact_vectors, sim1(question_vectors)]))
        # dot2 = sum_along_last(layers.multiply([fact_vectors, sim2(memory_vectors)]))
        
        gate_feature_vectors = concatenate([pointwise1, pointwise2, delta1, delta2])

        x = gate_dense1(gate_feature_vectors)
        x = gate_dense2(x)
        x = flatten(x)
        if iteration == 0:
            hint_output = x
        x = repeat_recur_size_times(x)
        gate_weights = permute(x)
    
        # Calculate the episode vector for this iteration.
        weighted_fact_vectors = layers.multiply([fact_vectors, gate_weights])
        episode = ep_gru(weighted_fact_vectors)
        if use_mem_gru:
            episode = repeat_once(episode)
            memory_vector = mem_gru(episode, initial_state=memory_vector)
        else:
            memory_vector = episode
        
    # Decode answer.    
    repeated_question = RepeatVector(answer_length)(question_vector)
    x = embedding_lookup(decoder_input)
    x = concatenate([x, repeated_question])
    x, _ = output_gru(x, initial_state=memory_vector)
    answer_prediction = word_predictor(x)
    
    # Build training model.
    inputs = story_input, question_input, decoder_input
    outputs = answer_prediction, hint_output
    train_model = Model(inputs=inputs, outputs=outputs)
    train_model.compile(loss=['sparse_categorical_crossentropy', 'mse'],
                              optimizer='rmsprop', metrics=['accuracy'])
                      
    # Build encoder model.
    inputs = story_input, question_input
    outputs =  [memory_vector, question_vector]
    encoder_model = Model(inputs=inputs, outputs=outputs)
    
    # Build decoder model.
    decoder_prev_predict_input = Input(shape=(1,), name='decoder_prev_predict_input')
    decoder_question_input = Input(shape=(recur_size,), name='decoder_question_input')
    decoder_state_inputs = [Input(shape=(recur_size,), name=f'decoder_state_input_{i}') 
                            for i in range(recurrent_layers)]    
    
    x = embedding_lookup(decoder_prev_predict_input)
    repeated_question = repeat_once(decoder_question_input)
    x = concatenate([x, repeated_question])
    
    x, decoder_state = output_gru(x, initial_state=decoder_state_inputs[0])
    x = word_predictor(x)
    
    inputs = [decoder_prev_predict_input, decoder_question_input] + decoder_state_inputs
    outputs = [x, decoder_state]
    decoder_model = Model(inputs=inputs, outputs=outputs)
    
    return train_model, encoder_model, decoder_model


#
# DMN MODEL VERSION 2
#

def build_dmn2_model(num_story_sentences, story_sentence_length, question_length, 
                     answer_length, embedding_matrix, recur_size=256, recurrent_layers=1,
                     iterations=3, gate_dense_size=128, use_mem_gru=False, gate_supervision=True):
    
    story_sent_inputs = [Input(shape=(story_sentence_length,), name=f'story_sentence_{i}_input')
                         for i in range(num_story_sentences)]
    question_input = Input(shape=(question_length,), name='question_input')
    decoder_input = Input(shape=(answer_length,), name='decoder_input')
    
    embedding_lookup = Embedding(embedding_matrix.shape[0],
                                 embedding_matrix.shape[1],
                                 weights=[embedding_matrix],
                                 trainable=False,
                                 name='embedding_lookup')
    
    # Declare reused layers.
    # input_gru = GRU(recur_size, return_sequences=False, return_state=False)
    
    encoders = [] 
    for i in range(recurrent_layers):
        return_seq = False if i == recurrent_layers - 1 else True
        encoders.append(GRU(recur_size, return_sequences=return_seq, return_state=True, name=f'encoder_{i}'))
    
    ep_encoders = [] 
    for i in range(recurrent_layers):
        return_seq = False if i == recurrent_layers - 1 else True
        ep_encoders.append(GRU(recur_size, return_sequences=return_seq, return_state=True, name=f'ep_encoder_{i}'))
    
    if use_mem_gru:
        mem_dense1 = Dense(recur_size, activation='relu')
        mem_dense2 = Dense(recur_size, activation='tanh')
    
    # output_gru = GRU(recur_size, return_sequences=True, return_state=True)
    
    decoders = []
    for i in range(recurrent_layers):
        decoders.append(GRU(recur_size, return_state=True, return_sequences=True, name=f'decoder_{i}'))
    
    gate_dense1 = Dense(gate_dense_size, activation='tanh')
    gate_dense2 = Dense(1, activation='sigmoid')  # Sigmoid makes more sense.
    word_predictor = Dense(embedding_matrix.shape[0], activation='softmax', 
                           name='word_predictor')
    flatten = Flatten()
    # repeat_stroy_len_times = RepeatVector(story_length)
    repeat_recur_size_times = RepeatVector(recur_size)
    permute = Permute((2, 1))
    repeat_num_sents_times = RepeatVector(num_story_sentences)
    repeat_once = RepeatVector(1)
    
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
    
    # Generate memory vector.
    # memory_vectors = [question_vector] * recurrent_layers
    per_layer_memory_vectors = [question_vector] * recur_size
    question_vectors = repeat_num_sents_times(question_vector)
    pointwise1 = layers.multiply([merged_sents, question_vectors])
    delta1 = layers.subtract([merged_sents, question_vectors])
    delta1 = layers.multiply([delta1, delta1])
    
    for iteration in range(iterations):
        # Calculate the weights for the gate vectors.
        memory_vectors = repeat_num_sents_times(per_layer_memory_vectors[-1])
        pointwise2 = layers.multiply([merged_sents, memory_vectors])
        delta2 = layers.subtract([merged_sents, memory_vectors])
        delta2 = layers.multiply([delta2, delta2])
        
        # NOTE: This don't help, or at least isn't necessary.
        # dot1 = sum_along_last(layers.multiply([fact_vectors, sim1(question_vectors)]))
        # dot2 = sum_along_last(layers.multiply([fact_vectors, sim2(memory_vectors)]))
        
        gate_feature_vectors = concatenate([pointwise1, pointwise2, delta1, delta2])
        x = gate_dense1(gate_feature_vectors)
        x = gate_dense2(x)
        x = flatten(x)
    
        if iteration == iterations - 1:
            hint_output = x
        
        x = repeat_recur_size_times(x)
        gate_weights = permute(x)
        
        # Calculate the episode vector for this iteration.
        weighted_fact_vectors = layers.multiply([merged_sents, gate_weights])
        
        episodes = []
        x = weighted_fact_vectors
        for ep_encoder in ep_encoders:
            x, episode = ep_encoder(x, initial_state=initial_state)
            episodes.append(episode)
            
        # episodes = []
        # x = weighted_fact_vectors
        # for ep_encoder, prev_episode in zip(ep_encoders, per_layer_memory_vectors):
        #     x, episode = ep_encoder(x, initial_state=prev_episode)
        #     episodes.append(episode)
        
        
        if use_mem_gru:
            new_per_layer_memory_vectors = []
            for memory_vector, episode in zip(per_layer_memory_vectors, episodes):
                x = concatenate([memory_vector, episode])
                x = mem_dense1(x)
                memory_vector = mem_dense2(x)
                new_per_layer_memory_vectors.append(memory_vector)
            per_layer_memory_vectors = new_per_layer_memory_vectors
        else:
            per_layer_memory_vectors = episodes

    # Decode answer.    
    repeated_question = RepeatVector(answer_length)(question_vector)
    x = embedding_lookup(decoder_input)
    x = concatenate([x, repeated_question])
    
    for decoder, memory_vector in zip(decoders, per_layer_memory_vectors):
        x, _ = decoder(x, initial_state=memory_vector)
    answer_prediction = word_predictor(x)
    
    # Build training model.
    inputs = story_sent_inputs + [question_input, decoder_input]
    outputs = [answer_prediction]
    losses = ['sparse_categorical_crossentropy']
    if gate_supervision:
        outputs.append(hint_output)
        losses.append('mse')
    train_model = Model(inputs=inputs, outputs=outputs)
    train_model.compile(loss=losses, optimizer='rmsprop', metrics=['accuracy'])
                      
    # Build encoder model.
    inputs = story_sent_inputs + [question_input]
    outputs =  per_layer_memory_vectors + [question_vector]
    # print('outputs', outputs)
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
    
    return train_model, encoder_model, decoder_model
    

#
# SHARED UTILS
#

def model_is_sent_level(model):
    for input in model.inputs:
        if 'sentence' in input.name:
            return True
    return False

def model_use_att_hints(model):
    pass

def predict(encoder_model, decoder_model, stories, questions, max_answer_length, story_masks=None):
    """Returns the predictions as indicies."""
    
    encoder_inputs = {}
    
    if model_is_sent_level(encoder_model):
        for i in range(stories.shape[1]):
            encoder_inputs[f'story_sentence_{i}_input'] = stories[:, i]
    else:
        encoder_inputs['story_input'] = stories
    
    # print(encoder_inputs)
    encoder_inputs['question_input'] = questions
    
    # print(encoder_inputs)
    
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
        

def train_model(model, data, epochs):
    # Train 
    checkpoint_path = os.path.join(data_dir_path, f'temp_weights_{socket.gethostname()}.hdf5')    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, 
                                   save_weights_only=True, monitor='val_loss')
    
    X_train = {
        'story_input': data.X_train_stories, 
        'question_input': data.X_train_questions, 
        'decoder_input': data.X_train_decoder
    }
    
    for i in range(data.X_train_story_sents.shape[1]):
        X_train[f'story_sentence_{i}_input'] = data.X_train_story_sents[:, i]
    
    X_val = {
        'story_input': data.X_val_stories, 
        'question_input': data.X_val_questions, 
        'decoder_input': data.X_val_decoder
    }
    
    for i in range(data.X_val_story_sents.shape[1]):
        X_val[f'story_sentence_{i}_input'] = data.X_val_story_sents[:, i]
    
    y_train = [data.y_train[:, :, np.newaxis]]
    y_val = [data.y_val[:, :, np.newaxis]]
        
    if len(model.outputs) > 1:
        if model_is_sent_level(model):
            y_train.append(data.y_train_att)
            y_val.append(data.y_val_att)
        else:
            y_train.append(data.X_train_hints)
            y_val.append(data.X_val_hints)
    
    # Train the model.
    model.fit(X_train, y_train, batch_size=128, epochs=epochs,
              validation_data=(X_val, y_val),
              callbacks=[checkpointer])

    # Load the best weights.
    model.load_weights(checkpoint_path)

    return model
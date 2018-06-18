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
    Reshape
)
from keras import backend as K
import os
import numpy as np
import socket


data_dir_path = os.path.join(os.path.dirname(__file__), 'data')


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
    
    # encoded_questions = []
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
    x = word_predictor(x)
    
    inputs = [decoder_prev_predict_input, decoder_question_input] + decoder_state_inputs
    outputs = [x] + decoder_states
    decoder_model = Model(inputs=inputs, outputs=outputs)
    
    
    
    # x = embedding_lookup(decoder_input)
    # repeated_question = RepeatVector(answer_length)(encoded_questions[-1])
    # x = concatenate([x, repeated_question])
    # get_last_time_step = Lambda(lambda t: t[:, -1,])
    # for decoder, encoded_story in zip(decoders, encoded_stories):
    #     if len(encoded_story.shape) > 2:
    #         encoded_story = get_last_time_step(encoded_story)
    #     x, _ = decoder(x, initial_state=encoded_story)
    # outputs = TimeDistributed(word_predictor)(x)
    # 
    
    
    return train_model, encoder_model, decoder_model


#
# DMN MODEL
#

def build_model(embedding_matrix, recur_size=128, gate_dense_size=128, iterations=3):
    input_length = 20
    
    story_input = Input(shape=(input_length,))
    question_input = Input(shape=(4,))
    
    embedding_lookup = Embedding(embedding_matrix.shape[0],
                                 embedding_matrix.shape[1],
                                 weights=[embedding_matrix],
                                 trainable=False)
    
    story_embedded = embedding_lookup(story_input)
    question_embedded = embedding_lookup(question_input)
    
    ep_gru = GRU(recur_size)
    mem_gru = GRU(recur_size)
    # sentence_mask = Masking(0)
    
    # TODO: This should be bidirectional, but you'll need to deal with
    # the state stuff. It also might not make sense to reuse it... though
    # then any direct similarity between the fact vectors and the question
    # vects ceases to make sense. But still, keeping them separate and
    # using a simpler attention mechanism. We'll just have to see.
    
    
    # input_gru = Bidirectional(GRU(recur_size // 2, return_sequences=True))
    input_gru = GRU(recur_size, return_sequences=True, return_state=True)
    gate_dense1 = TimeDistributed(Dense(gate_dense_size, activation='tanh'))
    gate_dense2 = TimeDistributed(Dense(1, activation='sigmoid'))  # Sigmoid makes more sense.
    
    fact_vectors, state = input_gru(story_embedded)
    
    # fact_vectors = sentence_mask(fact_vectors)
    # k.zeros_like
    
    question_vector, _ = input_gru(question_embedded, initial_state=K.zeros_like(state))
    question_vector = Lambda(lambda x: x[:, 0])(question_vector)
    memory_vector = question_vector
    
    # TODO: You can implement the per sentence masking by passing in a
    # mask of 0s for all put '.' characters and using these to weight the 
    # gate weights. In essense, where there's no gate weight, the gru step
    # is a no op. (EDIT: No, you'll have implement that yourself, but it's
    # totally doable.)
    
    # NOTE: These lambdas are going to kill masks unless you pass in 
    # a mask value when you create them. This may or may not matter.
    # The masking of zeros doesn't seem to help much in the standard model
    # where it is at least possible. 
    
    for i in range(iterations):
        
        # TODO: You're not actually managing the state that well
        gate_vectors = get_gate_vectors(input_length, fact_vectors, memory_vector, question_vector, gate_dense1, gate_dense2)
        
        # Compute episode for this iteration.
        state = None
        for i in range(input_length):
            fact_vector = Lambda(lambda x: x[:, i:i + 1])(fact_vectors)  # (?, recur_size)
            x = ep_gru(fact_vector, initial_state=state)
            gate_weight = Lambda(lambda x: x[:, i:i + 1])(gate_vectors)
            if state is None:
                state = Lambda(lambda f: f * gate_weight)(x)  # Such lambdas can't be serialized.
            else:
                state = Lambda(lambda f: f * gate_weight + (1 - gate_weight) * state)(x)
        episode = state
        
        episode = Reshape((1, recur_size))(episode)
        # Use episode to update episodic memory.
        memory_vector = mem_gru(episode, initial_state=memory_vector)
        
        
        
    output_gru = GRU(recur_size)
    output_input = Reshape((1, recur_size))(question_vector)
    x = output_gru(output_input, initial_state=memory_vector)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(embedding_matrix.shape[1], activation='linear')(x)
    model = Model(inputs=[story_input, question_input], outputs=outputs)
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print(model.summary())
    
    return model


# TODO: You'll want to try square instead of abs here.
def get_gate_vectors(input_length, fact_vectors, memory_vector, question_vector, dense1, dense2):
    memory_vectors = RepeatVector(input_length)(memory_vector)
    question_vectors = RepeatVector(input_length)(question_vector)
    # pointwise1 = layers.multiply([fact_vectors, question_vectors])
    # pointwise2 = layers.multiply([fact_vectors, memory_vectors])
    # delta1 = K.square(layers.subtract([fact_vectors, question_vectors]))
    # delta2 = K.square(layers.subtract([fact_vectors, memory_vectors]))
    # These are dot products, but I couldn't get the built in dot products
    # to work with three axis tensors.
    # TODO: Try square instead of abs.
    # dot1 = K.sum(fact_vectors * memory_vectors, axis=-1, keepdims=True)
    # dot2 = K.sum(fact_vectors * question_vectors, axis=-1, keepdims=True)
    feature_vectors = layers.concatenate([fact_vectors, memory_vectors, question_vectors])
    # feature_vectors = layers.concatenate([fact_vectors, memory_vectors, question_vectors, pointwise1, pointwise2, delta1, delta2])

    x = dense1(feature_vectors)
    x = dense2(x)
    x = Reshape((input_length,))(x)
    return x


#
# SHARED UTILS
#


def predict(encoder_model, decoder_model, stories, questions, max_answer_length):
    """Returns the predictions as indicies."""
    *encoded_stories, encoded_questions = encoder_model.predict([stories, questions])
    batch_size = encoded_stories[0].shape[0]
    preds = np.zeros((batch_size, 1)) + 2
    decoder_states = encoded_stories
    pred_list = []
    for i in range(max_answer_length):
        preds, *decoder_states = decoder_model.predict([preds, encoded_questions] + decoder_states)
        preds = np.argmax(preds, axis=-1)
        pred_list.append(preds)
    return np.concatenate(pred_list, axis=-1)
        

def train_model(model, data, epochs):
    # Train 
    checkpoint_path = os.path.join(data_dir_path, f'temp_weights_{socket.gethostname()}.hdf5')    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True, monitor='val_loss')
    
    X_train = [data.X_train_stories, data.X_train_questions, data.X_train_decoder]
    X_val = [data.X_val_stories, data.X_val_questions, data.X_val_decoder]
    
    # Train the model.
    model.fit(X_train, data.y_train[:, :, np.newaxis], batch_size=128, epochs=epochs,
              validation_data=(X_val, data.y_val[:, :, np.newaxis]),
              callbacks=[checkpointer])

    # Load the best weights.
    model.load_weights(checkpoint_path)

    return model
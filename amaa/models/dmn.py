import keras
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Dense,
    Input,
    Embedding,
    Bidirectional,
    GRU,
    Dropout,
    Concatenate,
    Reshape,
    RepeatVector,
    Lambda,
    TimeDistributed,
    Masking
)
import keras.layers as layers

recur_size = 128
gate_dense_size = 128
iterations = 3

# NOTE: Stop messing with this. None of the results are valid until you get all the
# rest of it done.

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
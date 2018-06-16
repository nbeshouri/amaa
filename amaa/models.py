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
    TimeDistributed
)
from keras import backend as K
import os
import numpy as np


data_dir_path = os.path.join(os.path.dirname(__file__), 'data')


def build_baseline_model(story_length, question_length, answer_length, embedding_matrix, recur_size=128, dense_size=128):        
    story_input = Input(shape=(story_length,))
    question_input = Input(shape=(question_length,))
    decoder_input = Input(shape=(answer_length,))
    
    embedding_lookup = Embedding(embedding_matrix.shape[0],
                                 embedding_matrix.shape[1],
                                 weights=[embedding_matrix],
                                 trainable=False)
                                 
    encoder = Bidirectional(GRU(recur_size // 2))
    decoder = GRU(recur_size, return_sequences=True)
    
    # Encode story.
    x = embedding_lookup(story_input)
    encoded_story = encoder(x)
    
    # Encode question.
    x = embedding_lookup(question_input)
    zeros = K.zeros_like(encoded_story)
    state_1 = K.slice(zeros, (0, 0), (-1, recur_size // 2))
    state_2 = K.slice(zeros, (0, recur_size // 2), (-1, -1))
    encoded_question = encoder(x, initial_state=[state_1, state_2])
    
    # Decode answer.
    x = embedding_lookup(decoder_input)
    repeated_question = RepeatVector(answer_length)(encoded_question)
    x = concatenate([x, repeated_question])
    x = decoder(x, initial_state=encoded_story)
    outputs = TimeDistributed(Dense(embedding_matrix.shape[1], activation='softmax'))(x)
    
    # Build model.
    inputs = (story_input, question_input, decoder_input)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
                  
    return model

                  
def train_model(model, data, epochs):
    # Train 
    checkpoint_path = os.path.join(data_dir_path, 'temp_model.hdf5')    
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
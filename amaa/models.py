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


#
# BASELINE MODEL
#


def build_baseline_model(story_length, question_length, answer_length, 
                         embedding_matrix, recur_size=128, dense_size=128):        
    
    story_input = Input(shape=(story_length,), name='story_input')
    question_input = Input(shape=(question_length,), name='question_input')
    
    
    decoder_input = Input(shape=(answer_length,), name='decoder_input')
    
    embedding_lookup = Embedding(embedding_matrix.shape[0],
                                 embedding_matrix.shape[1],
                                 weights=[embedding_matrix],
                                 trainable=False,
                                 name='embedding_lookup')
                                 
    encoder = Bidirectional(GRU(recur_size // 2), name='encoder')
    decoder = GRU(recur_size, return_state=True, return_sequences=True, name='decoder')
    word_predictor = Dense(embedding_matrix.shape[0], activation='softmax', name='word_predictor')
    
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
    x, _ = decoder(x, initial_state=encoded_story)
    outputs = TimeDistributed(word_predictor)(x)
    
    # Build training model.
    inputs = story_input, question_input, decoder_input
    train_model = Model(inputs=inputs, outputs=outputs)
    train_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    # Build encoder model.
    inputs = story_input, question_input
    outputs = encoded_story, encoded_question
    encoder_model = Model(inputs=inputs, outputs=outputs)
    
    # Build decoder model.
    decoder_prev_predict_input = Input(shape=(1,), name='decoder_prev_predict_input')
    decoder_state_input = Input(shape=(recur_size,), name='decoder_state_input')
    decoder_question_input = Input(shape=(recur_size,), name='decoder_question_input')
    
    x = embedding_lookup(decoder_prev_predict_input)
    repeated_question = RepeatVector(1)(decoder_question_input)
    x = concatenate([x, repeated_question])
    x, decoder_state = decoder(x, initial_state=decoder_state_input)
    x = word_predictor(x)
    inputs = decoder_prev_predict_input, decoder_state_input, decoder_question_input
    outputs = x, decoder_state
    decoder_model = Model(inputs=inputs, outputs=outputs)
    
    return train_model, encoder_model, decoder_model


#
# SHARED UTILS
#


def predict(encoder_model, decoder_model, stories, questions, max_answer_length):
    """Returns the predictions as indicies."""
    encoded_stories, encoded_questions = encoder_model.predict([stories, questions])
    batch_size = encoded_stories.shape[0]
    preds = np.zeros((batch_size, 1)) + 2
    decoder_state = encoded_stories
    pred_list = []
    for i in range(max_answer_length):
        preds, decoder_state = decoder_model.predict([preds, decoder_state, encoded_questions])
        preds = np.argmax(preds, axis=-1)
        pred_list.append(preds)
    return np.concatenate(pred_list, axis=-1)
        

def score_on_babi_tasks(model):
    """Return a Series of task specific scores."""
    


def train_model(model, data, epochs):
    # Train 
    checkpoint_path = os.path.join(data_dir_path, 'temp_weights.hdf5')    
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
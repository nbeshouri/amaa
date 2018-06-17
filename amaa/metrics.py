import numpy as np
from sklearn.metrics import accuracy_score
from . import data, models


def get_accuracy(encoder_model, decoder_model, X_stories, X_questions, y, id_to_word):    
    y_pred = models.predict(encoder_model, decoder_model, X_stories, X_questions, y.shape[1])
    y_pred = data.answer_id_lists_to_texts(y_pred, id_to_word)
    y_true = data.answer_id_lists_to_texts(y, id_to_word)
    return accuracy_score(y_true, y_pred)


def score_on_babi_tasks(encoder_model, decoder_model, task_subset=None, use_10k=False):
    """Return a Series of task specific scores."""
    # TODO: Should always return train and val, optional test data.
    # TODO: Check to make sure this actually works on all tasks.
    if task_subset is None:
        task_subet = range(1, 21)
    
    task_scores = {}
    for task_num in task_subset:
        babi_data = data.get_babi_data(task_subset=[task_num], use_10k=use_10k)
        X = [babi_data.X_test_stories, babi_data.X_test_questions, babi_data.X_test_decoder]
        task_score = get_accuracy(encoder_model, 
                                  decoder_model, 
                                  babi_data.X_test_stories,
                                  babi_data.X_test_questions,
                                  babi_data.y_test,
                                  babi_data.id_to_word)
        task_scores[task_num] = task_score
    return task_scores
    
    
    
    
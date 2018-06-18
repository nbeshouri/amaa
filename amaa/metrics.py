import numpy as np
import os
from sklearn.metrics import accuracy_score
from . import data, models, utils
import pandas as pd
import keras.backend as K



def get_accuracy(encoder_model, decoder_model, X_stories, X_questions, y, id_to_word):    
    y_pred = models.predict(encoder_model, decoder_model, X_stories, X_questions, y.shape[1])
    y_pred = data.answer_id_lists_to_texts(y_pred, id_to_word)
    y_true = data.answer_id_lists_to_texts(y, id_to_word)
    return accuracy_score(y_true, y_pred)


def score_on_task(encoder_model, decoder_model, data_bunch):
    row = {}
    for dataset in 'train', 'val', 'test':
        row[dataset] = get_accuracy(encoder_model, 
                                    decoder_model, 
                                    data_bunch[f'X_{dataset}_stories'],
                                    data_bunch[f'X_{dataset}_questions'],
                                    data_bunch[f'y_{dataset}'],
                                    data_bunch.id_to_word)
    
    return pd.Series(row)


def score_on_babi_tasks(model_builder_func, weights_prefix=None, epochs=100, 
                        multi_task_epochs=100, model_kwargs=None, task_subset=None, 
                        use_10k=False, results_file_name=None, mode='single', 
                        multi_weight_save_prefix=None):
    """Return a Series of task specific scores."""
    # TODO: Should always return train and val, optional test data.
    # TODO: Check to make sure this actually works on all tasks.
    
    if task_subset is None:
        task_subset = range(1, 21)

    if model_kwargs is None:
        model_kwargs = {}
    
    rows = []
    if mode in ('single', 'both'):
    
        for task_num in task_subset:
            babi_data = data.get_babi_data(task_subset=[task_num], use_10k=use_10k)
            train_model, encoder_model, decoder_model = model_builder_func(
                babi_data.X_train_stories.shape[1], babi_data.X_train_questions.shape[1], 
                babi_data.y_train.shape[1], babi_data.embedding_matrix, **model_kwargs)
            if weights_prefix is not None:
                utils.load_weights(train_model, weights_prefix)
            else:
                print('single run, training task:', task_num)
                models.train_model(train_model, babi_data, epochs)
            print('single run, testing task:', task_num)
            row = score_on_task(encoder_model, decoder_model, babi_data)
            row.loc['task_number'] = task_num
            row.loc['trained_on'] = 'single'
            row.loc['10k'] = use_10k
            rows.append(row)
            # K.clear_session()
            
        # TODO: Using clear session has side effects. You can get keras
        # to use a tf graph, but you need to figure out how force the
        # removal of just that graph from memory. K.clear_session just
        # seems to call the tf.clear_default_graph.
        # NOTE: Calling this every iteration causes a seg fault
        # in the model builder step.
        # K.clear_session()

    if mode in ('multi', 'both'):
        
        training_babi_data = data.get_babi_data(use_10k=use_10k)
        
        train_model, encoder_model, decoder_model = model_builder_func(
            training_babi_data.X_train_stories.shape[1], training_babi_data.X_train_questions.shape[1], 
            training_babi_data.y_train.shape[1], training_babi_data.embedding_matrix, **model_kwargs)
        
        if weights_prefix is not None:
            utils.load_weights(train_model, weights_prefix)
        else:
            print('multi run, training all tasks')
            models.train_model(train_model, training_babi_data, epochs)
        
        if multi_weight_save_prefix is not None:
            utils.save_weights(train_model, multi_weight_save_prefix)
        
        for task_num in range(1, 21):
            print('multi run, testing task', task_num)
            testing_babi_data = data.get_babi_data(
                task_subset=[task_num], use_10k=use_10k, 
                forced_story_length=training_babi_data.X_train_stories.shape[1], 
                forced_question_length=training_babi_data.X_train_questions.shape[1],
                forced_answer_length=training_babi_data.y_train.shape[1])
    
            row = score_on_task(encoder_model, decoder_model, testing_babi_data)
            row.loc['task_number'] = task_num
            row.loc['trained_on'] = 'multi'
            row.loc['10k'] = use_10k
            rows.append(row)        
    
    # K.clear_session()

    results = pd.DataFrame(rows)
    results['epochs'] = epochs
    results['model_builder_func'] = model_builder_func.__name__
    results['model_kwargs'] = str(model_kwargs)
    if results_file_name is not None:
        path = os.path.join(utils.data_dir_path, results_file_name)
        utils.archive_data(results_file_name)
        results.to_csv(path)
    return results


def clip_data_set(data_bunch, limit):
    for key in data_bunch.keys():
        if 'X_' in key or 'y_' in key:
            data_bunch[key] = data_bunch[key][:limit]
    return data_bunch










# def score_on_babi_tasks(model_builder_func, weights_prefix, model_kwargs=None,  
#                         task_subset=None, use_10k=False):
#     """Return a Series of task specific scores."""
#     # TODO: Should always return train and val, optional test data.
#     # TODO: Check to make sure this actually works on all tasks.
# 
#     if task_subset is None:
#         task_subset = range(1, 21)
# 
#     if model_kwargs is None:
#         model_kwargs = {}
# 
#     datasets = 'train', 'val', 'test'
#     rows = []
#     for task_num in task_subset:
#         babi_data = data.get_babi_data(task_subset=[task_num], use_10k=use_10k)
#         train_model, encoder_model, decoder_model = model_builder_func(
#             babi_data.X_train_stories.shape[1], babi_data.X_train_questions.shape[1], 
#             babi_data.y_train.shape[1], babi_data.embedding_matrix, **model_kwargs)
#         utils.load_weights(train_model, weights_prefix)
#         row = score_on_task(encoder_model, decoder_model, babi_data)
#         # TODO: Using clear session has side effects. You can get keras
#         # to use a tf graph, but you need to figure out how force the
#         # removal of just that graph from memory. K.clear_session just
#         # seems to call the tf.clear_default_graph.
#         K.clear_session()  
#         rows.append(row)
# 
#     return pd.DataFrame(rows, index=task_subset)


# def score_on_babi_tasks(encoder_model, decoder_model, task_subset=None, use_10k=False):
#     """Return a Series of task specific scores."""
#     # TODO: Should always return train and val, optional test data.
#     # TODO: Check to make sure this actually works on all tasks.
# 
#     if task_subset is None:
#         task_subset = range(1, 21)
# 
#     all_task_data = data.get_babi_data(use_10k=use_10k)
# 
#     datasets = 'train', 'val', 'test'
#     rows = []
#     for task_num in task_subset:
#         babi_data = data.get_babi_data(task_subset=[task_num], use_10k=use_10k)
#         row = {}
#         for dataset in datasets:
#             task_score = get_accuracy(encoder_model, 
#                                       decoder_model, 
#                                       babi_data[f'X_{dataset}_stories'],
#                                       babi_data[f'X_{dataset}_questions'],
#                                       babi_data[f'y_{dataset}'],
#                                       babi_data.id_to_word)
#             row[dataset] = task_score
#         rows.append(row)
# 
#     return pd.DataFrame(rows, columns=datasets, index=task_subset)
    
    
    
# def score_on_babi_tasks(model_builder, weights_prefix, model_kwargs=None,  
#                         task_subset=None, use_10k=False):
#     """Return a Series of task specific scores."""
#     # TODO: Should always return train and val, optional test data.
#     # TODO: Check to make sure this actually works on all tasks.
# 
#     if task_subset is None:
#         task_subset = range(1, 21)
# 
#     if model_kwargs is None:
#         model_kwargs = {}
# 
#     datasets = 'train', 'val', 'test'
#     rows = []
#     for task_num in task_subset:
#         babi_data = data.get_babi_data(task_subset=[task_num], use_10k=use_10k)
#         train_model, encoder_model, decoder_model = models.build_baseline_model(
#             babi_data.X_train_stories.shape[1], babi_data.X_train_questions.shape[1], 
#             babi_data.y_train.shape[1], babi_data.embedding_matrix, **model_kwargs)
#         utils.load_weights(train_model, weights_prefix)
#         row = {}
#         for dataset in datasets:
#             task_score = get_accuracy(encoder_model, 
#                                       decoder_model, 
#                                       babi_data[f'X_{dataset}_stories'],
#                                       babi_data[f'X_{dataset}_questions'],
#                                       babi_data[f'y_{dataset}'],
#                                       babi_data.id_to_word)
#             row[dataset] = task_score
#         K.clear_session()
#         rows.append(row)
# 
#     return pd.DataFrame(rows, columns=datasets, index=task_subset)    
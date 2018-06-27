import os
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from glob import glob
from . import data, models, utils


def get_accuracy(encoder_model, decoder_model, X_stories,
                 X_questions, y, id_to_word, story_masks=None):
    y_pred = models.predict(
        encoder_model, decoder_model, X_stories,
        X_questions, y.shape[1], story_masks)
    y_pred = data.answer_id_lists_to_texts(y_pred, id_to_word)
    y_true = data.answer_id_lists_to_texts(y, id_to_word)
    return accuracy_score(y_true, y_pred)


def score_on_task(encoder_model, decoder_model, data_bunch):
    row = {}
    for dataset in 'train', 'val', 'test':
        if models.model_is_sent_level(encoder_model):
            row[dataset] = get_accuracy(
                encoder_model, decoder_model, data_bunch[f'X_{dataset}_story_sents'],
                data_bunch[f'X_{dataset}_questions'], data_bunch[f'y_{dataset}'],
                data_bunch.id_to_word, data_bunch[f'X_{dataset}_hints'])
        else:
            row[dataset] = get_accuracy(
                encoder_model, decoder_model, data_bunch[f'X_{dataset}_stories'],
                data_bunch[f'X_{dataset}_questions'], data_bunch[f'y_{dataset}'],
                data_bunch.id_to_word, data_bunch[f'X_{dataset}_hints'])
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
    
    if results_file_name is not None:
        results_path = os.path.join(utils.data_dir_path, results_file_name)
        utils.archive_data(results_file_name)
    
    def write_out_results(rows):
        results = pd.DataFrame(rows)
        results['epochs'] = epochs
        results['model_builder_func'] = model_builder_func.__name__
        results['model_kwargs'] = str(model_kwargs)
        results.to_csv(results_path)
        return results
    
    print(model_builder_func)
    print(model_kwargs)
    print(results_file_name)
    
    rows = []
    if mode in ('single', 'both'):
    
        for task_num in task_subset:
            babi_data = data.get_babi_data(task_subset=[task_num], use_10k=use_10k)
            if 'dmn' in model_builder_func.__name__:
                train_model, encoder_model, decoder_model = model_builder_func(
                    babi_data.X_train_story_sents.shape[1], 
                    babi_data.X_train_story_sents.shape[2],
                    babi_data.X_train_questions.shape[1], 
                    babi_data.y_train.shape[1], 
                    babi_data.embedding_matrix, 
                    **model_kwargs)
            else:
                train_model, encoder_model, decoder_model = model_builder_func(
                    babi_data.X_train_stories.shape[1], 
                    babi_data.X_train_questions.shape[1], 
                    babi_data.y_train.shape[1], 
                    babi_data.embedding_matrix, 
                    **model_kwargs)
            if weights_prefix is not None:
                utils.load_weights(train_model, weights_prefix)
            if epochs is not None:
                print('single run, training task:', task_num)
                models.train_model(train_model, babi_data, epochs)
            print('single run, testing task:', task_num)
            row = score_on_task(encoder_model, decoder_model, babi_data)
            row.loc['task_number'] = task_num
            row.loc['trained_on'] = 'single'
            row.loc['10k'] = use_10k
            print(row)
            rows.append(row)
            write_out_results(rows)
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
        
        if 'dmn' in model_builder_func.__name__:
            train_model, encoder_model, decoder_model = model_builder_func(
                training_babi_data.X_train_story_sents.shape[1], 
                training_babi_data.X_train_story_sents.shape[2],
                training_babi_data.X_train_questions.shape[1], 
                training_babi_data.y_train.shape[1], 
                training_babi_data.embedding_matrix, 
                **model_kwargs
            )
        else:
            train_model, encoder_model, decoder_model = model_builder_func(
                training_babi_data.X_train_stories.shape[1], training_babi_data.X_train_questions.shape[1], 
                training_babi_data.y_train.shape[1], training_babi_data.embedding_matrix, **model_kwargs)
        
        if weights_prefix is not None:
            utils.load_weights(train_model, weights_prefix)
        
        if epochs is not None:
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
                forced_answer_length=training_babi_data.y_train.shape[1],
                forced_num_sents=training_babi_data.X_train_story_sents.shape[1],
                forced_sent_length=training_babi_data.X_train_story_sents.shape[2])
            row = score_on_task(encoder_model, decoder_model, testing_babi_data)
            row.loc['task_number'] = task_num
            row.loc['trained_on'] = 'multi'
            row.loc['10k'] = use_10k
            print(row)
            rows.append(row)        

    write_out_results(rows)
    
    
def merge_results(folder_path, save_results=True):
    # Read all into a DataFrame.
    glob_path = os.path.join(folder_path, '*.csv')
    csv_paths = glob(glob_path)
    dataframes = [pd.read_csv(path) for path in csv_paths 
                  if 'aggregate' not in path and 'per_task' not in path]
    best_results = []
    for task in range(1, 21):
        task_rows = []
        for df in dataframes:
            task_row = df.loc[df.task_number == task]
            if len(task_row) == 0:
                continue
            assert len(task_row) == 1
            task_row = task_row.iloc[0]  # As Series.
            task_rows.append(task_row)
        val_scores = [task_row.val for task_row in task_rows]
        best_row = task_rows[np.argmax(val_scores)]
        best_results.append(best_row)
    
    columns = ['train', 'val', 'test', 'task_number', 'trained_on', '10k',
               'epochs', 'model_builder_func', 'model_kwargs']

    per_task_df = pd.DataFrame(best_results, columns=columns)
    agg_df = per_task_df[['train', 'val', 'test']].mean()
    
    if save_results:
        per_task_path = os.path.join(folder_path, 'per_task_results.csv')
        per_task_df.to_csv(per_task_path)
        agg_path = os.path.join(folder_path, 'aggregate_results.csv')
        agg_df.to_csv(agg_path)
        
    return per_task_df, agg_df


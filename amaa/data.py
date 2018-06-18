"""
This module contains a collection of functions used to generate different
versions of the data set.

Todo:
    

"""



# from . import utils
import os
from glob import glob
import re
import numpy as np
from munch import Munch
from itertools import chain
from . import transforms
from . import utils
import joblib
import numpy as np


data_dir_path = os.path.join(os.path.dirname(__file__), 'data')
_babi_1K_path = os.path.join(data_dir_path, 'tasks_1-20_v1-2', 'en')
_babi_10K_path = os.path.join(data_dir_path, 'tasks_1-20_v1-2', 'en-10k')
joblib_cache_dir_path = os.path.join(data_dir_path, 'joblib_cache')
memory = joblib.Memory(cachedir=joblib_cache_dir_path)



def _read_babi_lines(path):
    
    story_lines = []

    with open(path) as f:
        for line in f:
            line_num = int(line.split(' ')[0])
            if line_num == 1 and story_lines:
                story_lines = []
            if '?' in line:
                yield story_lines, line
            else:
                story_lines.append(line)


def load_task(task_num, version, folder_path):
    """
    Creates a set of QA tuples for a given task.
    """
    glob_path = os.path.join(folder_path, f'qa{task_num}_*{version}*')
    task_path = glob(glob_path)[0]
    
    output = []
    
    for story_lines, question_line in _read_babi_lines(task_path):
        story = ''.join(story_lines)
        story = story.replace('\n', ' ')
        story = re.sub(r'\d+\s', r'', story)
        question, answer, _ = question_line.split('\t')
        question = re.sub(r'\d+\s', r'', question)
        output.append((story, question, answer))
    
    return output


def load_tasks(version, folder_path, task_subset=None):
    """
    Load the QA tuples for all tasks.
    """
    output = []
    for i in range(1, 21):
        if task_subset is not None and i not in task_subset:
            continue
        output.append(load_task(i, version, folder_path))
    return output


def get_embeddings(texts, min_vocab_size=0):
    """
    Build emeddding mappings based on `texts`.

    Args:
        texts (Iterable[str]): A sequence of text to fit the embeddings
            on.
        min_vocab_size (Optional[int]): Minimum number of words, not
            counting those in `texts` to include in the embedding
            vocabulary.

    Returns:
        (tuple): tuple containing:
            word_to_vec (dict): A map between word tokens and numpy vectors.
            word_to_id (dict): A map between word tokens and embedding ids.
            embedding_matrix (np.ndarry): A numpy matrix that maps between
                embedding ids and embedding vectors.

    """
    data_vocab = set()
    for token_list in transforms.token_strs_to_token_lists(texts):
        for token in token_list:
            data_vocab.add(token)

    word_to_vec = {}
    embeddings_path = os.path.join(data_dir_path, 'glove.6B/glove.6B.200d.txt')
    with open(embeddings_path) as f:
        for line_num, line in enumerate(f):
            values = line.split()
            word = values[0]
            if min_vocab_size < line_num + 1 and word not in data_vocab:
                continue
            vector = np.asarray(values[1:], dtype='float32')
            word_to_vec[word] = vector

    total_vocab = data_vocab | set(word_to_vec.keys())
    rand_state = np.random.RandomState(42)
    word_to_id = {'<PAD>': 0, '<EOS>': 1, '<START>': 2}
    num_meta_tokens = 3
    embedding_matrix = rand_state.rand(len(total_vocab) + num_meta_tokens, 200)
    embedding_matrix[0] = 0
    embedding_matrix[1] = 1
    embedding_matrix[2] = 2
    for i, word in enumerate(total_vocab):
        word_id = i + num_meta_tokens
        if word in word_to_vec:
            embedding_matrix[word_id] = word_to_vec[word]
        word_to_id[word] = word_id

    return word_to_vec, word_to_id, embedding_matrix


def ids_to_text(ids, id_to_word):
    tokens = [id_to_word[id] for id in ids]
    return ' '.join(tokens)

    
def id_lists_to_texts(id_lists, id_to_word):
    return [ids_to_text(ids, id_to_word) for ids in id_lists]


def answer_ids_to_text(ids, id_to_word):
    text = ids_to_text(ids, id_to_word)
    text = text.split('<EOS>')[0]
    return text.strip()

def answer_id_lists_to_texts(id_lists, id_to_word):
    return [answer_ids_to_text(ids, id_to_word) for ids in id_lists]


def get_train_val_test(train_sqas, test_sqas, task_subset=None):
    
    
    if task_subset == None:
        task_subset = range(1, 21)
    
    flat_train_sqas = chain(*train_sqas)
    flat_val_sqas = []
    flat_test_sqas = []
    test_sqas = np.array(test_sqas)
    for task_sqas, task_num in zip(test_sqas, task_subset):        
        _, val_indices, test_indices = get_train_val_test_indices(
            len(task_sqas), val_ratio=0.50, test_ratio=0.50, seed=42 + task_num)        
        flat_val_sqas.extend(task_sqas[val_indices])
        flat_test_sqas.extend(task_sqas[test_indices])
    
    def process_sqas(sqas):
        token_sqas = map(transforms.tokenize_texts, zip(*sqas))
        token_sqas = tuple(map(np.array, token_sqas))
        return token_sqas
    
    train_stories, train_questions, train_answers = process_sqas(flat_train_sqas)
    val_stories, val_questions, val_answers = process_sqas(flat_val_sqas)
    test_stories, test_questions, test_answers = process_sqas(flat_test_sqas)
    
    data = Munch(
        X_train_stories=train_stories,
        X_train_questions=train_questions,
        y_train=train_answers,
        
        X_val_stories=val_stories,
        X_val_questions=val_questions,
        y_val=val_answers,
        
        X_test_stories=test_stories,
        X_test_questions=test_questions,
        y_test=test_answers
    )
    
    return data


def texts_to_ids(texts, word_to_id, max_sequence_length=None):
    """
    Args:
        texts (Iterable[str]): A sequence of texts to fit the embeddings
            on.
        word_to_id (dict): A map between word tokens and embedding ids.
        max_sequence_length (Optional[int]): The maximum number of words
            to include in each line of dialogue. Shorter sequences will
            be padded with the <PAD> vector.

    Returns:
        X (np.ndarray): An array with shape `(len(texts), max_sequence_length)`
            containing the correct embedding ids.

    """
    
    token_lists = transforms.token_strs_to_token_lists(texts)
    texts_ids = []
    for token_list in token_lists:
        word_ids = []
        for token in token_list:
            if token in word_to_id:
                word_ids.append(word_to_id[token])
        # Add <EOS> token.
        word_ids.append(1)  
        texts_ids.append(word_ids)
    
    if max_sequence_length is None:
        max_sequence_length = max(len(ids) for ids in texts_ids)
    
    X = np.zeros((len(texts), max_sequence_length), dtype=int)
    for i, text_ids in enumerate(texts_ids):
        text_ids = text_ids[:max_sequence_length]
        X[i, :len(text_ids)] = text_ids

    return X


def align_padding(*id_matrices, forced_length=None):
    if forced_length is None:
        max_row_len = max(matrix.shape[1] for matrix in id_matrices)
    else:
        max_row_len = forced_length
    output = []
    for matrix in id_matrices:
        padding_needed = max_row_len - matrix.shape[1]
        padded_matrix = np.pad(matrix, ((0, 0), (0, padding_needed)), 
                               'constant', constant_values=(0, 0))
        output.append(padded_matrix)
    return output


def add_nn_X_y(data, word_to_id, X_padding):
    # This should be smart enough to add the correct amount of
    # padding on its own. 
    keys = [key for key in data.keys if 'X_' in key]
    for key in keys:
        data[key] = to_ids(data[key])
    return data
    
    
    # data = Munch(
    #     X_train_stories=train_stories,
    #     X_train_questions=train_questions,
    #     y_train=train_answers,
    # 
    #     X_val_stories=test_stories[val_indices],
    #     X_val_questions=test_questions[val_indices],
    #     y_val=test_answers[val_indices],
    # 
    #     X_test_stories=test_stories[test_indices],
    #     X_test_questions=test_questions[test_indices],
    #     y_test=test_answers[test_indices]
    # )

def get_time_shifted(sequences):
    # TODO: Start, pad, etc. should be constants.
    start_tokens = np.zeros((sequences.shape[0], 1)) + 2  # <START> == 2
    return np.concatenate([start_tokens, sequences[:, :-1]], axis=1)


def get_train_val_test_indices(num_rows, val_ratio=0.25, test_ratio=0.25, seed=42):
    """Return indices of the train, test, and validation sets."""
    rand = np.random.RandomState(seed)
    indices = rand.permutation(range(num_rows))
    train_ratio = 1 - val_ratio - test_ratio
    train_indices = indices[:int(len(indices) * train_ratio)]
    val_indices = indices[len(train_indices):len(train_indices) + int(len(indices) * val_ratio)]
    test_indices = indices[len(train_indices) + len(val_indices):]
    return train_indices, val_indices, test_indices
    

@memory.cache
def get_babi_embeddings(use_10k=False, min_vocab_size=0):
    # Calculate embedding matrix with suitable vocab for all
    # tasks. I don't calculate a task specific vocab because
    # I want the word indices to be the same in all task data
    # sets.
    babi_path = _babi_10K_path if use_10k else _babi_1K_path
    train_tasks = load_tasks('train', babi_path)
    test_tasks = load_tasks('test', babi_path)
    tokenized_texts = map(transforms.to_tokens, chain(*chain(*chain(train_tasks, test_tasks))))
    return get_embeddings(tokenized_texts, min_vocab_size=min_vocab_size)


@memory.cache
def get_babi_data(task_subset=None, use_10k=False, forced_story_length=None, 
                  forced_question_length=None, forced_answer_length=None):
    
    babi_path = _babi_10K_path if use_10k else _babi_1K_path
    train_tasks = load_tasks('train', babi_path, task_subset)
    test_tasks = load_tasks('test', babi_path, task_subset)
    data = get_train_val_test(train_tasks, test_tasks, task_subset)
        
    data.word_to_vec, data.word_to_id, data.embedding_matrix = get_babi_embeddings(use_10k)
    data.id_to_word = {id: word for word, id in data.word_to_id.items()}
    
    # Convert lists of texts to lists of lists of word ids.
    keys = [key for key in data.keys() if 'X_' in key or 'y_' in key]
    for key in keys:
        data[key] = texts_to_ids(data[key], data.word_to_id)
    
    data.X_train_stories, data.X_val_stories, data.X_test_stories = align_padding(
        data.X_train_stories, data.X_val_stories, data.X_test_stories, 
        forced_length=forced_story_length)
    data.X_train_questions, data.X_val_questions, data.X_test_questions = align_padding(
        data.X_train_questions, data.X_val_questions, data.X_test_questions, 
        forced_length=forced_question_length)
    data.y_train, data.y_val, data.y_test = align_padding(
        data.y_train, data.y_val, data.y_test, forced_length=forced_answer_length)
        
    data.X_train_decoder = get_time_shifted(data.y_train)
    data.X_val_decoder = get_time_shifted(data.y_val)
    data.X_test_decoder = get_time_shifted(data.y_test)
    
    return data
    






# data = get_babi_data(task_subset=[1, 20])


# print(len(data.X_train_questions))
# 
# print(id_lists_to_texts(data.X_train_questions, data.id_to_word))

# small_babi = get_babi_data()
# utils.save_data(small_babi, 'babi_1K_data.pickle')
# 
# print('Done small.')
# 
# big_babi = get_babi_data(use_10k=True)
# utils.save_data(big_babi, 'babi_10k_data.pickle')

# TODO: Decorator that takes a cache name and will cache the output for a given 
# set of args. It should also take an optional flag to recompute data. Or it could
# hash the source code file to see if anything has changed. 
    

# get_babi_data()
    
# train_tasks = load_tasks('train', _babi_1K_path)[:100]
# test_tasks = load_tasks('test', _babi_1K_path)[:100]
# 
# # train_tasks = load_task(8, 'train', _babi_1K_path)
# # test_tasks = load_task(8, 'test', _babi_1K_path)
# data = get_train_val_test(train_tasks, test_tasks)
# 
# print(data.X_val_stories[3])
# print(data.X_val_questions[3])
# print(data.y_val[3])
# 
# print(data.X_train_stories[54])
# print(data.X_train_questions[54])
# print(data.y_train[54])



# print(len(train_tasks))
# print(train_tasks[:100])
# get_train_val_test(train_tasks, test_tasks)


# print(len(output))

    



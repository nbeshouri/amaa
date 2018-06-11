"""
This module contains utilities used by other modules or notebooks.

"""

import os
from glob import glob
import re
import joblib


data_dir_path = os.path.join(os.path.dirname(__file__), 'data')
archived_data_dir_path = os.path.join(data_dir_path, 'archived')


def matches_patterns(patterns, text, flags=re.IGNORECASE):
    """Return true if the matches any of the given patterns."""
    # TODO: This would probably be more efficient if you
    # joined all these patterns together and did a single searc.
    for pattern in patterns:
        match = re.search(pattern, text, flags=flags)
        if match is not None:
            return True
    return False


def load_values(file_name):
    """Import a file contain a list of comma separated strs as a list."""
    path = os.path.join(data_dir_path, file_name)
    with open(path) as f:
        data_str = f.read()
    data_str = data_str.replace('\n', ',')
    values = data_str.split(',')
    values = [v.strip() for v in values if v.strip()]
    output = []
    pattern = r'["\'](.*)["\']'
    for value in values:
        match = re.match(pattern, value)
        if match is not None:
            output.append(match.group(1))
        else:
            output.append(value)
    return output

         
def load_data(file_name):
    """Load a pickled file from the local data directory."""
    load_path = os.path.join(data_dir_path, file_name)
    data = joblib.load(load_path)
    return data


def save_data(data, file_name, compress=3):
    """Pickle data and save it in the local data directory."""
    path = os.path.join(data_dir_path, file_name)
    archive_data(file_name)
    joblib.dump(data, path, compress=compress)


def archive_data(file_name):
    """Move a file in the data folder to the archive folder if the file exists."""
    if '/' in file_name:
        raise ValueError('`file_name` should be a file name, not a path.')
    
    old_path = os.path.join(data_dir_path, file_name)
    
    # If the file doesn't exist, then there's nothing to do and we don't
    # need to worry about overwriting it.
    if not os.path.isfile(old_path):
        return
    
    # If there's already a file with this name (sans version number)
    # in the folder, then use that file's name to extract the current
    # version number. Otherwise go with the default of 0.
    last_version_num = 0
    just_name, extension = os.path.splitext(file_name)
    glob_path = os.path.join(archived_data_dir_path, just_name + '*' + extension)
    globbed = sorted(glob(glob_path))
    if globbed:
        last_version = sorted(glob(glob_path))[-1]
        pattern = f'\d+(?=\\{extension})'
        last_version_str = re.search(pattern, last_version).group(0)
        last_version_num = int(last_version_str)
    
    # Create a new file name with the incremented version number
    # and move it to the archive.
    new_version_num = last_version_num + 1
    new_version_str = f'{new_version_num:0>3g}'
    new_file_name = just_name + '_' + new_version_str + extension
    new_path = os.path.join(archived_data_dir_path, new_file_name)
    os.rename(old_path, new_path)

"""
data preparation tools
@vbar
"""
import os
import requests
import zipfile
import re
import shutil
from tqdm import tqdm


def download(url: str, path: str = None, file_name: str = None,
             replace_file=False, verify_url=True):
    """
    downloads url

    :param url: url to download
    :param path: destination path. None points at current directory
    :param file_name: name saved. None keeps original name
    :param replace_file: If True replaces files with the same name
    :param verify_url: Ignores secure communication certificate
    :return: local path of the file
    """
    if not file_name:
        file_name = url.split('/')[-1]

    if not path:
        path = os.getcwd().replace('\\', '/')
        full_path = f"{path}/{file_name}"
    else:
        path = path.replace('\\', '/')
        if path[-1] == '/':
            full_path = path + file_name
        else:
            full_path = f"{path}/{file_name}"

    if replace_file or not os.path.exists(full_path):
        if not os.path.exists(path):
            os.mkdir(path)
        print("Download starts...")
        file = requests.get(url, verify=verify_url)
        with open(full_path, 'wb') as f:
            f.write(file.content)
    else:
        print(f"Warning : File <{file_name}> has already been downloaded.")

    return full_path


def unzip_data(file_path: str, replace=False):
    """
    unzips a file

    :param file_path: file path to unzip
    :param replace: deletes directory with the same name
    :return: unzips file and creates directory with the same name
    """
    assert os.path.isfile(file_path), "Path does not contain a file"
    path_list = file_path.split('.')

    if replace:
        os.rmdir(file_path)
    elif os.path.exists(str().join(path_list[:-1])):
        pass
    elif path_list[-1] == "zip":
        with zipfile.ZipFile(file_path, 'r') as f:
            f.extractall(file_path.split('.')[0])
    else:
        raise Exception('File path points to a file')

    file_path = str().join(path_list[:-1])
    return file_path
    

def label_to_path(path, regex: str, labels: (list, tuple) = None, to_path: str = None):
    """
    labels a file list with regex and move them to labeled directories
    
    :param path: files' path
    :param regex: regex rule to get the labels
    :param labels: acceptable labels (should be lowered)
    :param to_path: destination path
    :return: unlabeled data
    """
    os.chdir(path)
    unlabeled = list()

    if not to_path:
        to_path = ".."

    def move_to_label(f, l_path):
        if not os.path.exists(l_path):
            os.mkdir(l_path)
        shutil.move(f, l_path)

    for file in tqdm(os.listdir()):
        label = re.findall(regex, file)[0].lower()
        if labels:
            if label in labels:
                move_to_label(file, os.path.join(to_path, label))
            else:
                unlabeled.append(file)
        else:
            move_to_label(file, os.path.join(to_path, label))

    if unlabeled:
        print('Warning : returns unlabeled files:', unlabeled, sep='\n')
    else:
        print('Everything is labeled')
        os.chdir("..")
        os.rmdir(path.split('/')[-1])

    return unlabeled

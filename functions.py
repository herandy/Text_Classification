import numpy as np
import os
import sys
from sys import platform
if platform == "win32":
    cp = 'copy '
else:
    cp = 'cp '


def augment(data, labels, worddict):
    # Augmentation in text can be tricky, one way would be to exchange similar words.
    # You need to make sure, as well as you can, that the meaning stays exactly the same.
    # For the ones done here I focused mostly on negative cues
    # but there is an endless amount of substitutes one can come up with
    aug_data = data.copy()
    aug_labels = labels.copy()
    for idx in range(len(data)):
        if worddict['stopped'] in aug_data[idx]:
            augmented = aug_data[idx].copy()
            augmented_2 = aug_data[idx].copy()
            augmented[np.where(augmented == worddict['stopped'])] = worddict['quit']
            augmented_2[np.where(augmented_2 == worddict['stopped'])] = worddict['ceased']
            while len(augmented) > len(aug_data[idx]):
                augmented = np.delete(augmented, np.where(augmented == 0)[0][0])
            while len(augmented) < len(aug_data[idx]):
                augmented = np.insert(augmented, np.where(augmented == 0)[0][0], 0)

            while len(augmented_2) > len(aug_data[idx]):
                augmented_2 = np.delete(augmented_2, np.where(augmented_2 == 0)[0][0])
            while len(augmented_2) < len(aug_data[idx]):
                augmented_2 = np.insert(augmented_2, np.where(augmented_2 == 0)[0][0], 0)
            aug_data = np.append(aug_data, np.expand_dims(augmented, axis=0), axis=0)
            aug_labels = np.append(aug_labels, labels[idx])
            aug_data = np.append(aug_data, np.expand_dims(augmented_2, axis=0), axis=0)
            aug_labels = np.append(aug_labels, labels[idx])
        if worddict['quit'] in aug_data[idx]:
            augmented = aug_data[idx].copy()
            augmented_2 = aug_data[idx].copy()
            augmented[np.where(augmented == worddict['quit'])] = worddict['stopped']
            augmented_2[np.where(augmented_2 == worddict['quit'])] = worddict['ceased']
            while len(augmented) > len(aug_data[idx]):
                augmented = np.delete(augmented, np.where(augmented == 0)[0][0])
            while len(augmented) < len(aug_data[idx]):
                augmented = np.insert(augmented, np.where(augmented == 0)[0][0], 0)

            while len(augmented_2) > len(aug_data[idx]):
                augmented_2 = np.delete(augmented_2, np.where(augmented_2 == 0)[0][0])
            while len(augmented_2) < len(aug_data[idx]):
                augmented_2 = np.insert(augmented_2, np.where(augmented_2 == 0)[0][0], 0)
            aug_data = np.append(aug_data, np.expand_dims(augmented, axis=0), axis=0)
            aug_labels = np.append(aug_labels, labels[idx])
            aug_data = np.append(aug_data, np.expand_dims(augmented_2, axis=0), axis=0)
            aug_labels = np.append(aug_labels, labels[idx])
        if (worddict['never'] in aug_data[idx]) & (worddict['stopped'] in aug_data[idx]):
            augmented = aug_data[idx].copy()
            augmented = np.insert(augmented, np.where(augmented == worddict['never'])[0] + 1, worddict['not'])
            augmented[np.where(augmented == worddict['never'])] = worddict['did']
            augmented[np.where(augmented == worddict['stopped'])] = worddict['stop']
            while len(augmented) > len(aug_data[idx]):
                augmented = np.delete(augmented, np.where(augmented == 0)[0][0])
            while len(augmented) < len(aug_data[idx]):
                augmented = np.insert(augmented, np.where(augmented == 0)[0][0], 0)
            aug_data = np.append(aug_data, np.expand_dims(augmented, axis=0), axis=0)
            aug_labels = np.append(aug_labels, labels[idx])
        if (worddict['did'] in aug_data[idx]) & (worddict['not'] in aug_data[idx]) & (worddict['stop'] in aug_data[idx]):
            augmented = aug_data[idx].copy()
            augmented = np.delete(augmented, np.where(augmented == worddict['not']))
            augmented[np.where(augmented == worddict['did'])] = worddict['never']
            augmented[np.where(augmented == worddict['stop'])] = worddict['stopped']
            while len(augmented) > len(aug_data[idx]):
                augmented = np.delete(augmented, np.where(augmented == 0)[0][0])
            while len(augmented) < len(aug_data[idx]):
                augmented = np.insert(augmented, np.where(augmented == 0)[0][0], 0)

            aug_data = np.append(aug_data, np.expand_dims(augmented, axis=0), axis=0)
            aug_labels = np.append(aug_labels, labels[idx])

    return aug_data, aug_labels


class Logger(object):
    def __init__(self, output_path):
        self.terminal = sys.stdout
        self.log = open(output_path + "log.txt", "w+", encoding='utf8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def create_result_dirs(output_path, file_name):
    # This function creates a results folder and copies
    # the file you run as well as this functions file in there.
    if not os.path.exists(output_path):
        print('creating log folder in: ', output_path)
        os.makedirs(output_path)
        try:
            os.makedirs(os.path.join(output_path, 'params'))
        except:
            pass
        func_file_name = os.path.basename(__file__)
        if func_file_name.split('.')[1] == 'pyc':
            func_file_name = func_file_name[:-1]
        functions_full_path = os.path.join(output_path, func_file_name)
        cmd = cp + func_file_name + ' "' + functions_full_path + '"'
        os.popen(cmd)
        run_file_full_path = os.path.join(output_path, file_name)
        cmd = cp + file_name + ' "' + run_file_full_path + '"'
        os.popen(cmd)

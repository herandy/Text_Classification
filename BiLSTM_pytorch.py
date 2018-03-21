# -*- coding: utf-8 -*-
import os, sys
import torch
import torch.utils.data
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

torch.manual_seed(1)
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score, confusion_matrix
from functions import Logger, create_result_dirs
import re
import socket
import time


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class dataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, data, labels):
        self.data = data
        # enc = OneHotEncoder(sparse=False)
        self.labels = labels  # enc.fit_transform(labels.reshape(-1, 1))

    def __getitem__(self, index):
        datum = torch.LongTensor(self.data[index])
        label = torch.LongTensor([self.labels[index].tolist()])
        return datum, label

    def __len__(self):
        return len(self.data)


def getIds(filename, removepunc=False):
    if os.path.isfile(filename):
        with open(filename, 'r', encoding="utf8") as f:
            ptext = f.read().lower()
    else:
        ptext = filename.lower()
    ptext = re.sub("([0-9])+", lambda m: " " + m.group(0) + " ", ptext)
    ptext = re.sub(r"[^A-Za-z0-9\n]", lambda m: " " + m.group(0) + " ", ptext)
    if removepunc:
        ptext = re.sub(r"[^A-Za-z0-9 \.\n]", " ", ptext)
    ptext = re.sub(" +", " ", ptext)
    data = word_tokenize(ptext.replace('-', ' '))
    data_idx = [worddict.get(word, -1) for word in data]
    data_idx = [idx for idx in data_idx if idx > -1]
    data_idx = data_idx[-WINDOW_SIZE:]
    if len(data_idx) < WINDOW_SIZE:
        data_idx = np.pad(data_idx, (WINDOW_SIZE - len(data_idx), 0), 'constant', constant_values=(0, 0))
    return data_idx


def over_sample(x, y):
    freqs = np.bincount(y)
    vals = np.unique(y)
    maxfreq = np.max(freqs)
    for idx in np.unique(vals):
        multiplier = ((maxfreq - freqs[idx]) / freqs[idx])
        x = np.vstack((x, np.matlib.repmat(x[y == idx], int(np.floor(multiplier)), 1)))
        y = np.hstack((y, np.matlib.repmat(y[y == idx], int(np.floor(multiplier)), 1).flatten()))
        residual = int(np.round((multiplier - int(np.floor(multiplier))) * freqs[idx]))
        x = np.vstack((x, x[y == idx][:residual]))
        y = np.hstack((y, y[y == idx][:residual]))
    idx = np.random.permutation(len(y))
    x = x[idx, :]
    y = y[idx]
    return x, y


N_HIDDEN = 48
cuda_device = 0
# loss_coefficient = 0.75
LEARNING_RATE = 5e-4
WINDOW_SIZE = 72
NCLASS = 4
DROPOUT = 0.2
EPOCHS = 250
REG_VAL = 2e-6
# Lambda = 0.75
TRAIN_EMBEDDING = False
test = False
BATCH_SIZE = 256
test_batch_size = 200
test_path = '22-09-2017_14-07-11_MD1N289C'

print('Loading train_data...')
path = "./"

# Load word vectors. Depending on the format you might need to use another function in gensim to load your word vectors
# Convert glove to word2vec format
# import gensim
# gensim.scripts.glove2word2vec.glove2word2vec('glove.840B.300d.txt', 'glove')
# loadVectors = True
print('Loading Word Vectors...')
loadVectors = 'word2vec' not in locals()
if loadVectors:
    filename = (path + 'glove')
    # Glove format
    word2vec = KeyedVectors.load_word2vec_format(filename)
    # Normal word2vec format
    # word2vec = KeyedVectors.load(filename)
MAXNWORDS = word2vec.syn0.shape[0]

if 'voc' not in locals():
    voc = np.array(word2vec.index2word)
    worddict = {}
    i = 0
    for word in word2vec.index2word:
        worddict[word] = i
        i = i + 1

path = './'
if 'traindata' not in locals():
    # if os.path.exists(path+'ag_news_csv/processed.pickle'):
    #     with open(path+'ag_news_csv/processed.pickle', 'rb') as pf:
    #         [data, validdata, testdata, labels, validlabels, testlabels] = pickle.load(pf)
    #         data, labels = overSample2(data, labels)
    #         data, validdata, testdata = np.int32(data), np.int32(validdata), np.int32(testdata)
    #         labels, validlabels, testlabels = np.int32(labels), np.int32(validlabels), np.int32(testlabels)
    # else:
    import pandas as pd
    print('Loading data...')
    train_data = pd.read_csv(path + 'ag_news_csv/train.csv', header=None)
    test_data = pd.read_csv(path + 'ag_news_csv/test.csv', header=None)

    train_labels = train_data[0].as_matrix() - 1
    testlabels = test_data[0].as_matrix() - 1
    # NCLASS = np.max(train_labels) + 1
    NCLASS = len(np.unique(train_labels))
    train_data = train_data[1] + " " + train_data[2]
    test_data = test_data[1] + " " + test_data[2]

    train_data = pd.Series.apply(train_data, getIds)
    test_data = pd.Series.apply(test_data, getIds)
    train_data = np.array(train_data.tolist())
    testdata = np.array(test_data.tolist())

    from sklearn.model_selection import train_test_split

    train_data, train_labels = over_sample(train_data, train_labels)

    data, validdata, labels, validlabels = train_test_split(train_data, train_labels, test_size=0.05, random_state=42)
    data, validdata, testdata = np.int32(data), np.int32(validdata), np.int32(testdata)
    labels, validlabels, testlabels = np.int32(labels), np.int32(validlabels), np.int32(testlabels)

longrun = []
trainlongrun = []
if not test:
    output_path = './results/' + os.path.basename(__file__).split('.')[0] + '/' + time.strftime("%d-%m-%Y_") + \
                  time.strftime("%H-%M-%S") + '_' + socket.gethostname()
    pyscript_name = os.path.basename(__file__)
    create_result_dirs(output_path, pyscript_name)
    sys.stdout = Logger(output_path)
else:
    output_path = './results/' + os.path.basename(__file__).split('.')[0] + '/' + test_path

Note = 'Note : ---'

print('Creating network...')


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.word_embeddings.weight.data.copy_(torch.from_numpy(word2vec.syn0))
        self.word_embeddings.weight.requires_grad = False

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=2,
                            batch_first=True,
                            bidirectional=True, bias=True, dropout=DROPOUT)

        # The linear layer that maps from hidden state space to tag space
        self.predict_lstm = nn.Linear(2 * hidden_dim, tagset_size)

    def init_hidden_lstm(self, size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self.lstm.num_layers * 2, size, self.hidden_dim)).cuda(cuda_device),
                autograd.Variable(torch.zeros(self.lstm.num_layers * 2, size, self.hidden_dim)).cuda(cuda_device))

    def forward(self, sentence):
        hidden_lstm = self.init_hidden_lstm(sentence.size(0))
        embeds = self.word_embeddings(sentence)

        lstm_input = F.alpha_dropout(embeds, p=DROPOUT, training=self.training)
        lstm_out, self.hidden_lstm = self.lstm(lstm_input, hidden_lstm)
        concatenated_output = torch.cat([lstm_out.select(1, WINDOW_SIZE - 1).contiguous()], dim=1)
        concatenated_output = F.dropout(concatenated_output, p=DROPOUT, training=self.training)
        output_lstm = self.predict_lstm(concatenated_output)

        return output_lstm


model = BiLSTM(word2vec.syn0.shape[1], N_HIDDEN, MAXNWORDS, NCLASS)

# tensor = autograd.Variable(torch.IntTensor(torch.from_numpy(data)))

train_set = dataset(data, labels)
valid_set = dataset(validdata, validlabels)
test_set = dataset(testdata, testlabels)

data_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
validation_data_loader = torch.utils.data.DataLoader(valid_set, batch_size=test_batch_size, shuffle=False,
                                                     num_workers=0)
test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=0)

loss = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE,
                             weight_decay=REG_VAL)
model.cuda(cuda_device)
loss.cuda(cuda_device)

best_f1 = 0

for step in range(EPOCHS):
    train_preds = []
    train_shuffled_labels = []

    model.training = True
    batch_loss = 0
    for i, batch_data in enumerate(data_loader, 0):
        # lstm_hidden = model.init_hidden_lstm()
        # gru_hidden = model.init_hidden_gru()
        model.zero_grad()
        gpu_data, batch_labels = batch_data
        batch_size = gpu_data.size(0)
        gpu_data = gpu_data.cuda(cuda_device)
        batch_labels = batch_labels.cuda(cuda_device)
        gpu_data = Variable(gpu_data)
        batch_labels = Variable(batch_labels)
        optimizer.zero_grad()
        prediction = model.forward(gpu_data)
        output = loss.forward(prediction, batch_labels.squeeze())
        _, tmp_preds = torch.max(prediction, 1)
        train_preds.extend(tmp_preds.cpu().data.numpy())
        train_shuffled_labels.extend(batch_labels.cpu().data.numpy())
        output.backward()
        optimizer.step()
        batch_loss += output.data[0]

    train_f1 = f1_score(train_shuffled_labels, train_preds, average='weighted')

    valid_preds = []

    model.training = False
    for i, batch_data in enumerate(validation_data_loader, 0):
        gpu_data, batch_labels = batch_data
        batch_size = gpu_data.size(0)
        gpu_data = gpu_data.cuda(cuda_device)
        batch_labels = batch_labels.cuda(cuda_device)
        gpu_data = Variable(gpu_data)
        batch_labels = Variable(batch_labels)
        prediction = model.forward(gpu_data)
        valid_preds.extend(prediction.cpu().data.numpy())

    valid_f1 = f1_score(validlabels, np.argmax(valid_preds, axis=1), average='weighted')

    test_preds = []

    for i, batch_data in enumerate(test_data_loader, 0):
        gpu_data, batch_labels = batch_data
        batch_size = gpu_data.size(0)
        gpu_data = gpu_data.cuda(cuda_device)
        batch_labels = batch_labels.cuda(cuda_device)
        gpu_data = Variable(gpu_data)
        batch_labels = Variable(batch_labels)
        prediction = model.forward(gpu_data)
        test_preds.extend(prediction.cpu().data.numpy())

    test_f1 = f1_score(testlabels, np.argmax(test_preds, axis=1), average='weighted')

    print('Epoch %d\t\t\tTraining F1: %.3f\t\tValidation F1: %.3f\t\tTest F1: %.3f\t\tloss: %.7f' % (
        step, train_f1, valid_f1, test_f1, batch_loss / (len(labels) // BATCH_SIZE)))

    if valid_f1 > best_f1:
        print('new best f1: %.3f' % valid_f1)
        best_f1 = valid_f1
        save_checkpoint({
            'epoch': step + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_f1,
            'optimizer': optimizer.state_dict(),
        }, filename=os.path.join(output_path, 'checkpoint.pth.tar'))


final_test_preds = []

model.training = False
for i, batch_data in enumerate(test_data_loader, 0):
    gpu_data, batch_labels = batch_data
    batch_size = gpu_data.size(0)
    gpu_data = gpu_data.cuda(cuda_device)
    batch_labels = batch_labels.cuda(cuda_device)
    gpu_data = Variable(gpu_data)
    batch_labels = Variable(batch_labels)
    prediction = model.forward(gpu_data)
    final_test_preds.extend(prediction.cpu().data.numpy())

final_test_preds_ensemble = []

for i, batch_data in enumerate(test_data_loader, 0):
    gpu_data, batch_labels = batch_data
    batch_size = gpu_data.size(0)
    gpu_data = gpu_data.cuda(cuda_device)
    batch_labels = batch_labels.cuda(cuda_device)
    gpu_data = Variable(gpu_data)
    batch_labels = Variable(batch_labels)
    model.training = False
    prediction = model.forward(gpu_data)
    model.training = True
    for idx in range(20):
        prediction += model.forward(gpu_data)
    final_test_preds_ensemble.extend(prediction.cpu().data.numpy())

test_f1 = f1_score(testlabels, np.argmax(final_test_preds_ensemble, axis=1), average='weighted')

print('\nFinal\t\t\tTest ensemble F1: %.3f' % test_f1)

# Load best Model

checkpoint = torch.load(os.path.join(output_path, 'checkpoint.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])

# best_valid_preds = []
#
# model.training = False
# for i, batch_data in enumerate(validation_data_loader, 0):
#     gpu_data, batch_labels = batch_data
#     batch_size = gpu_data.size(0)
#     gpu_data = gpu_data.cuda(cuda_device)
#     batch_labels = batch_labels.cuda(cuda_device)
#     gpu_data = Variable(gpu_data)
#     batch_labels = Variable(batch_labels)
#     prediction = model.forward(gpu_data)
#     best_valid_preds.extend(prediction.cpu().data.numpy())
#
# valid_f1 = f1_score(validlabels, np.argmax(best_valid_preds, axis=1), average='weighted')

best_test_preds = []

model.training = False
for i, batch_data in enumerate(test_data_loader, 0):
    gpu_data, batch_labels = batch_data
    batch_size = gpu_data.size(0)
    gpu_data = gpu_data.cuda(cuda_device)
    batch_labels = batch_labels.cuda(cuda_device)
    gpu_data = Variable(gpu_data)
    batch_labels = Variable(batch_labels)
    prediction = model.forward(gpu_data)
    best_test_preds.extend(prediction.cpu().data.numpy())

test_f1 = f1_score(testlabels, np.argmax(best_test_preds, axis=1), average='weighted')

print('\nBest\t\t\tTest F1: %.3f' % test_f1)

test_f1 = f1_score(testlabels, np.argmax(np.array(final_test_preds)+np.array(best_test_preds), axis=1), average='weighted')

print('\nTest final and best F1: %.3f' % test_f1)

best_test_preds_ensemble = []

for i, batch_data in enumerate(test_data_loader, 0):
    gpu_data, batch_labels = batch_data
    batch_size = gpu_data.size(0)
    gpu_data = gpu_data.cuda(cuda_device)
    batch_labels = batch_labels.cuda(cuda_device)
    gpu_data = Variable(gpu_data)
    batch_labels = Variable(batch_labels)
    model.training = False
    prediction = model.forward(gpu_data)
    model.training = True
    for idx in range(20):
        prediction += model.forward(gpu_data)
    best_test_preds_ensemble.extend(prediction.cpu().data.numpy())

test_f1 = f1_score(testlabels, np.argmax(best_test_preds_ensemble, axis=1), average='weighted')

print('\nBest\t\t\tTest ensemble F1: %.3f' % test_f1)

test_f1 = f1_score(testlabels, np.argmax(np.array(best_test_preds_ensemble)+np.array(final_test_preds_ensemble), axis=1), average='weighted')

print('\nEnsemble final and best\t\tTest F1: %.3f' % test_f1)

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:19:45 2017

@author: z003herx & z003tduh
"""

from keras.layers import *
from keras.layers.merge import average
from keras.models import Model
from keras.optimizers import Adam
from gensim.models import KeyedVectors
import os
from keras.utils import np_utils
import numpy as np
from keras import regularizers
import keras
import keras.backend as K
from sklearn import metrics
from nltk.tokenize import word_tokenize
import re
from functions import augment
from keras_functions import f1, f1_metric
import keras.backend as K

best_f1 = [0.0]
best_model = []

# datafolder = "../../data/"
# folder = "Corpora_2/Training/"
# validfolder = "Corpora_2/Validation/"
# featurefolder = "reports_smoking_annotated\\Corpora\\featureData\\"
# testfolder = "Educational/"

# labeldict = {"nonsmoker": 0, "current": 1, "previous": 2, "unknown": 3, "notrelated": -1}
# labelinvdict = {0: "nonsmoker", 1: "current", 2: "previous", 3: "unknown", -1: "notrelated"}
# labelvoc = np.array(list(labeldict.keys()))


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


NCLASS = 4
N_HIDDEN = 48
WINDOW_SIZE = 72
BATCH_SIZE = 256
EPOCHS = 250
DROP_OUT = 0.2
REGVAL = 1e-6
LEARNING_RATE = 5e-4
SUPERVISED_FEATURES = False
augment_flag = False


# def getIds(filename, removepunc=False):
#     with open(filename, 'r', encoding="utf8") as f:
#         data = normalizeText(f.read(), removepunc)
#         data_idx = [worddict.get(word, -1) for word in data]
#         data_idx = [idx for idx in data_idx if idx > -1]
#         data_idx = data_idx[-MAXLEN:]
#         if len(data_idx) < MAXLEN:
#             data_idx = np.pad(data_idx, (MAXLEN - len(data_idx), 0), 'constant', constant_values=(0, 0))
#         return np.array(data_idx).astype('int32')

def getBatch(idx):
    x = data[idx:(idx + BATCH_SIZE)]
    y = labels[idx:(idx + BATCH_SIZE)]
    # x2 = datax[idx:(idx + BATCH_SIZE)]
    return x, y


def overSample2(x, y):
    freqs = np.bincount(y)
    vals = np.unique(y)
    maxfreq = np.max(freqs)
    for idx in np.unique(vals):
        multiplier = (maxfreq - freqs[idx]) / freqs[idx]
        x = np.vstack((x, np.matlib.repmat(x[y == idx], int(np.floor(multiplier)), 1)))
        y = np.hstack((y, np.matlib.repmat(y[y == idx], int(np.floor(multiplier)), 1).flatten()))
        residual = int(np.round((multiplier - int(np.floor(multiplier))) * freqs[idx]))
        x = np.vstack((x, x[y == idx][:residual]))
        y = np.hstack((y, y[y == idx][:residual]))
    idx = np.random.permutation(len(y))
    x = x[idx, :]
    y = y[idx]
    return x, y


optimizer = Adam(lr=LEARNING_RATE)


def CNN_model(N_HIDDEN=24):
    lin = Input(shape=(WINDOW_SIZE,), dtype='int32', name='input')
    emb = Embedding(NWORDS, DIM, input_length=WINDOW_SIZE, weights=[word2vec.syn0], trainable=False)(lin)

    dense = Dense(N_HIDDEN, activation='selu', kernel_regularizer=regularizers.l2(REGVAL),
                  kernel_initializer='lecun_normal')(emb)

    drop = AlphaDropout(rate=DROP_OUT)(dense)
    lstm = Bidirectional(CuDNNLSTM(N_HIDDEN, return_sequences=False, kernel_regularizer=regularizers.l2(REGVAL)))(drop)
    lstm_dense = Dense(NCLASS, activation='softmax', kernel_regularizer=regularizers.l2(REGVAL), name='lstm_dense')(lstm)

    model = Model(inputs=[lin], outputs=lstm_dense)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[f1])
    return model


# epsilon = 1.0e-9
#
#
# def get_ensemble_loss(y_pred_lstm, y_pred_cnn):
#     def ensemble_loss(y_true, y_pred_lstm, y_pred_cnn):
#         """Just another crossentropy"""
#         y_pred_lstm = K.clip(y_pred_lstm, epsilon, 1.0 - epsilon)
#         y_pred_lstm /= y_pred_lstm.sum(axis=-1, keepdims=True)
#         y_pred_cnn = K.clip(y_pred_cnn, epsilon, 1.0 - epsilon)
#         y_pred_cnn /= y_pred_cnn.sum(axis=-1, keepdims=True)
#         cce = K.categorical_crossentropy(y_true, y_pred_lstm) + K.categorical_crossentropy(y_true, y_pred_cnn)
#         return cce
#     return ensemble_loss

from keras.layers import Input, Embedding,concatenate,Conv1D,GlobalMaxPooling1D,Dropout,Dense,Bidirectional,LSTM,GRU
from keras.models import Model
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, log_loss, confusion_matrix
from keras import regularizers
from keras.callbacks import ModelCheckpoint

path = './'
print('Loading Word Vectors...')
loadVectors = 'word2vec' not in locals()
if loadVectors:
    filename = (path + 'glove')
    # Glove format
    word2vec = KeyedVectors.load_word2vec_format(filename)
    # Normal word2vec format
    # word2vec = KeyedVectors.load(filename)
MAXNWORDS = word2vec.syn0.shape[0]

DIM = word2vec.syn0.shape[1]
NWORDS = word2vec.syn0.shape[0]

if 'voc' not in locals():
    voc = np.array(word2vec.index2word)
    worddict = {}
    i = 0
    for word in word2vec.index2word:
        worddict[word] = i
        i = i + 1


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

    train_data, train_labels = overSample2(train_data, train_labels)

    data, validdata, labels, validlabels = train_test_split(train_data, train_labels, test_size=0.05, random_state=42)
    data, validdata, testdata = np.int32(data), np.int32(validdata), np.int32(testdata)
    labels, validlabels, testlabels = np.int32(labels), np.int32(validlabels), np.int32(testlabels)
    

# model = CNN_model()
# model.fit(data, np_utils.to_categorical(labels), epochs=EPOCHS, batch_size=BATCH_SIZE,
#           validation_data=(validdata, np_utils.to_categorical(validlabels)), callbacks=[metric])

model = CNN_model()
# model = CNN_model()
# small = small_CNN_model()

best_model.append(model.get_weights())

metric = f1_metric(testdata=testdata, testlabels=testlabels, best_f1=best_f1, best_model=best_model)

noisy_prediction = K.function(inputs=[model.layers[0].input, K.learning_phase()],
                              outputs=[model.get_layer('lstm_dense').output])

model.fit(data, y=np_utils.to_categorical(labels), epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_data=(validdata, np_utils.to_categorical(validlabels)),
          callbacks=[metric, keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)])

model.set_weights(metric.best_model[0])

test_prob = model.predict(testdata)
pred = np.argmax(test_prob, axis=1)
cm = metrics.confusion_matrix(testlabels, pred)
print("Test F1: %.3f" % (metrics.f1_score(testlabels, pred, average='weighted')))
print("Test Accuracy : %.3f" % np.mean(testlabels == pred))
print('Confusion Matrix: ')
print(cm)
p, r, f1, s = metrics.precision_recall_fscore_support(testlabels, pred)
# print('f1: ', np.average(f1, weights=s))
testensemble = test_prob

for i in range(0, 19):
    prob = noisy_prediction([testdata, 1])[0]
    testensemble = testensemble + prob

prob_normalized = testensemble / 20
pred = np.argmax(prob_normalized, axis=1)
print("Test ensemble F1: %.3f" % (metrics.f1_score(testlabels, pred, average='weighted')))
print("Test ensemble Accuracy : %.3f" % np.mean(testlabels == pred))
cm = metrics.confusion_matrix(testlabels, pred, labels=[0, 1, 2, 3])
print('Confusion Matrix: ')
print(cm)

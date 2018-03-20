import keras.backend as K
import keras
from sklearn import metrics
import numpy as np


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


class f1_metric(keras.callbacks.Callback):
    def __init__(self, testdata=[], testlabels=[], best_f1=[], best_model=[]):
        super().__init__()
        self.testdata = testdata
        self.testlabels = testlabels
        self.best_f1 = best_f1
        self.best_model = best_model

    def on_epoch_end(self, batch, logs={}):
        prob = np.asarray(self.model.predict(self.validation_data[0]))
        prediction= np.argmax(prob, axis=1)
        targ = np.argmax(self.validation_data[1], axis=1)
        print()

        print(metrics.classification_report(targ, prediction))
        cm = metrics.confusion_matrix(targ, prediction)
        print('Confusion Matrix for ensemble: ')
        print(cm)
        p, r, f1, s = metrics.precision_recall_fscore_support(targ, prediction)
        self.precisions = np.average(p, weights=s)
        self.recalls = np.average(r, weights=s)
        self.f1s = np.average(f1, weights=s)

        if self.f1s > self.best_f1[0]:
            self.best_f1[0] = self.f1s
            print("Best f1 is: ", self.best_f1[0])
            test_prob = self.model.predict(self.testdata)

            print('best model, test F1:')
            # print(metrics.classification_report(test_labels, np.argmax(test_pred_lstm, axis=1)))
            cm = metrics.confusion_matrix(self.testlabels, np.argmax(test_prob, axis=1))
            print('Confusion Matrix: ')
            print(cm)
            p, r, f1, s = metrics.precision_recall_fscore_support(self.testlabels, np.argmax(test_prob, axis=1))
            print('test f1: ', np.average(f1, weights=s))
            self.best_model[0] = self.model.get_weights()
        return


class logloss_metric(keras.callbacks.Callback):
    def __init__(self, testdata=[], testlabels=[], best_logloss=[], best_model=[]):
        super().__init__()
        self.testdata = testdata
        self.testlabels = testlabels
        self.best_logloss = best_logloss
        self.best_model = best_model

    def on_epoch_end(self, batch, logs={}):
        prob = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        print()

        self.logloss = metrics.log_loss(targ, prob)

        print("logloss: ", self.logloss)

        if self.logloss < self.best_logloss[0]:
            self.best_logloss[0] = self.logloss
            print("Best logloss is: ", self.best_logloss[0])
            self.best_model = self.model.get_weights()
        return


class f1_metric_cnn_lstm(keras.callbacks.Callback):
    def __init__(self, testdata=[], testlabels=[], best_f1=[], best_model=[]):
        super().__init__()
        self.testdata = testdata
        self.testlabels = testlabels
        self.best_f1 = best_f1
        self.best_model = best_model

    def on_epoch_end(self, batch, logs={}):
        prob = np.asarray(self.model.predict(self.validation_data[0]))
        prob_lstm = prob[0]
        prob_cnn = prob[1]
        prediction_lstm = np.argmax(prob_lstm, axis=1)
        prediction_cnn = np.argmax(prob_cnn, axis=1)
        prediction = np.argmax(prob_lstm+prob_cnn, axis=1)
        targ = np.argmax(self.validation_data[1], axis=1)
        print()

        print('Ensemble: ')
        print(metrics.classification_report(targ, prediction))
        print('LSTM and GRU: ')
        print(metrics.classification_report(targ, prediction_lstm))
        print('CNN: ')
        print(metrics.classification_report(targ, prediction_cnn))
        cm = metrics.confusion_matrix(targ, prediction)
        print('Confusion Matrix for ensemble: ')
        print(cm)
        p, r, f1, s = metrics.precision_recall_fscore_support(targ, prediction)
        self.precisions = np.average(p, weights=s)
        self.recalls = np.average(r, weights=s)
        self.f1s = np.average(f1, weights=s)

        if self.f1s > self.best_f1[0]:
            self.best_f1[0] = self.f1s
            print("Best f1 is: ", self.best_f1[0])
            test_prob_lstm, test_prob_cnn = self.model.predict(self.testdata)

            print('best model, test F1:')
            # print(metrics.classification_report(test_labels, np.argmax(test_pred_lstm, axis=1)))
            cm = metrics.confusion_matrix(self.testlabels, np.argmax(test_prob_lstm + test_prob_cnn, axis=1))
            print('Confusion Matrix: ')
            print(cm)
            p, r, f1, s = metrics.precision_recall_fscore_support(self.testlabels, np.argmax(test_prob_lstm + test_prob_cnn, axis=1))
            print('test f1: ', np.average(f1, weights=s))
            self.best_model[0] = self.model.get_weights()
        return


class logloss_metric(keras.callbacks.Callback):
    def __init__(self, testdata=[], testlabels=[], best_logloss=[], best_model=[]):
        super().__init__()
        self.testdata = testdata
        self.testlabels = testlabels
        self.best_logloss = best_logloss
        self.best_model = best_model

    def on_epoch_end(self, batch, logs={}):
        prob = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        print()

        self.logloss = metrics.log_loss(targ, prob)

        print("logloss: ", self.logloss)

        if self.logloss < self.best_logloss[0]:
            self.best_logloss[0] = self.logloss
            print("Best logloss is: ", self.best_logloss[0])
            self.best_model = self.model.get_weights()
        return

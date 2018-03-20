import os
import sys
import numpy as np
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T


def padding(l,max_len,pad_idx,x=True):
    if len(l[0]) == 1:
        if x: pad = [pad_idx]*(max_len-len(l))
        else: pad = [[0,1]]*(max_len-len(l))
        return np.concatenate((l,pad),axis=0)
    elif len(l[0]) > 1:
        L = np.zeros((len(l), max_len), dtype=np.int32)
        for idx in range(len(l)):
            if x: pad = [pad_idx]*(max_len-len(l[idx]))
            else: pad = [[0,1]]*(max_len-len(l[idx]))
            L[idx] = np.concatenate((l[idx],pad),axis=0)
        return L


__all__ = [
    "DropoutLayer",
    "AlphaDropoutLayer"
    ]


class DropoutLayer(lasagne.layers.Layer):
    """Dropout layer
    Sets values to zero with probability p. See notes for disabling dropout
    during testing.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If ``True`` (the default), scale the input by ``1 / (1 - p)`` when
        dropout is enabled, to keep the expected output mean the same.
    shared_axes : tuple of int
        Axes to share the dropout mask over. By default, each value can be
        dropped individually. ``shared_axes=(0,)`` uses the same mask across
        the batch. ``shared_axes=(2, 3)`` uses the same mask across the
        spatial dimensions of 2D feature maps.
    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.
    The behaviour of the layer depends on the ``deterministic`` keyword
    argument passed to :func:`lasagne.layers.get_output`. If ``True``, the
    layer behaves deterministically, and passes on the input unchanged. If
    ``False`` or not specified, dropout (and possibly scaling) is enabled.
    Usually, you would use ``deterministic=False`` at train time and
    ``deterministic=True`` at test time.
    See also
    --------
    dropout_channels : Drops full channels of feature maps
    spatial_dropout : Alias for :func:`dropout_channels`
    dropout_locations : Drops full pixels or voxels of feature maps
    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.
    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """
    def __init__(self, incoming, p=0.5, rescale=True, shared_axes=(),
                 **kwargs):
        super(DropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale
        self.shared_axes = tuple(shared_axes)

    @property
    def q(self):
        return T.constant(1) - self.p

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            return self.apply_dropout(input, const=0)

    def apply_dropout(self, input, const=0):
        # Using theano constant to prevent upcasting
        one = T.constant(1)

        if self.rescale:
            input /= self.q

        # use nonsymbolic shape for dropout mask if possible
        mask_shape = self.input_shape
        if any(s is None for s in mask_shape):
            mask_shape = input.shape

        # apply dropout, respecting shared axes
        if self.shared_axes:
            shared_axes = tuple(a if a >= 0 else a + input.ndim
                                for a in self.shared_axes)
            mask_shape = tuple(1 if a in shared_axes else s
                               for a, s in enumerate(mask_shape))
        mask = self._srng.binomial(mask_shape, p=self.q,
                                   dtype=input.dtype)
        if self.shared_axes:
            bcast = tuple(bool(s == 1) for s in mask_shape)
            mask = T.patternbroadcast(mask, bcast)

        if const != 0:
            return (input * mask) + (const * (T.constant(1) - mask))
        else:
            return input * mask


class AlphaDropoutLayer(DropoutLayer):
    """Dropout layer.
    Sets values to alpha if the dropout mask doesn't filter out inputs already.
    This keeps the zero mean and unit variance self-normalizing property true.
    See notes for disabling dropout during testing.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to alpha
    rescale : bool
        If ``True`` (the default), scale the input by ``1 / (1 - p)`` when
        dropout is enabled, to keep the expected output mean the same.
    shared_axes : tuple of int
        Axes to share the dropout mask over. By default, each value can be
        dropped individually. ``shared_axes=(0,)`` uses the same mask across
        the batch. ``shared_axes=(2, 3)`` uses the same mask across the
        spatial dimensions of 2D feature maps.
    alpha : float or SELU instance
        Is responsible for keeping the mean and variance consistent to what
        they were before AlphaDropout. This maintains the self-normalizing
        property. The default values are fixed point solutions to equations
        (4) and (5) in [1]_ for zero mean and unit variance input. The
        analytic expressions for them are given in equation (14) also in [1]_.
    Notes
    -----
    The alpha dropout layer is a regularizer that randomly sets input values to
    zero and also applies a function to bring the remaining neurons to a
    mean of 0 and the variance to 1 unit; see [1]_ for why this might improve
    generalization. The behaviour of the layer depends on the ``deterministic``
    keyword argument passed to :func:`lasagne.layers.get_output`. If ``True``,
    the layer behaves deterministically, and passes on the input unchanged. If
    ``False`` or not specified, dropout (and possibly scaling) is enabled.
    Usually, you would use ``deterministic=False`` at train time and
    ``deterministic=True`` at test time.
    References
    ----------
    .. [1] Klambauer, G., Unterthiner, T., Mayr, A., Hochreiter, S. (2017):
           Self-Normalizing Neural Networks. arXiv preprint: 1706.02515
    """

    def __init__(self, incoming, p=0.1, rescale=True, shared_axes=(),
                 alpha=None, **kwargs):
        """Class initialization."""
        super(AlphaDropoutLayer, self).__init__(incoming,
                                                p=p,
                                                rescale=rescale,
                                                shared_axes=shared_axes,
                                                **kwargs)
        from lasagne.nonlinearities import SELU
        if alpha is None:
            self.alpha = -1.0507009873554804934193349852946 * \
                         1.6732632423543772848170429916717
        elif isinstance(alpha, SELU):
            self.alpha = - alpha.scale * alpha.scale_neg
        else:
            self.alpha = alpha

    def get_output_for(self, input, deterministic=False, **kwargs):
        """Apply alpha dropout."""
        if deterministic or self.p == 0:
            return input
        else:
            mask = self.apply_dropout(input, const=self.alpha)
            a = T.pow(self.q + T.square(self.alpha) * self.q * self.p, -0.5)
            b = -a * self.p * self.alpha

            return a * mask + b


from keras import backend as K


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

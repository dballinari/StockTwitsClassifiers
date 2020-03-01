"""
Help function for making predictions
"""
import numpy as np


def get_chunks(iterable, chunk_size):
    """
    Generator that divides an iterable into chunks
    Args:
        iterable: iterable
        chunk_size: size of each chunk as an integer

    Returns: iterable in chunks

    """
    size = iterable.shape[0]
    if size < chunk_size:
        yield iterable
    chunks_nb = int(size / chunk_size)
    iter_ints = range(0, chunks_nb)
    for i in iter_ints:
        j = i * chunk_size
        if i + 1 < chunks_nb:
            k = j + chunk_size
            yield iterable[j:k]
        else:
            yield iterable[j:]


def predict_in_chunks(model, x, chunk_size):
    """
    Function that makes predictions with a model by feeding the data in chunks
    Args:
        model: a model of the module sklearn with the method 'predict'
        x: the features for which predictions are made
        chunk_size: size of the chunks as an integer

    Returns: a numpy-array of predictions

    """
    y = np.empty(0)
    for x_i in get_chunks(x, chunk_size):
        y_i = model.predict(x_i.toarray())
        y = np.concatenate((y, y_i))
    return y


def bull(x):
    """
    Compute the bullishness index based on a sentiment array x
    Args:
        x: array of sentiment scores with 0 being the neutral sentiment

    Returns: bullishness index as a float

    """
    return np.log((1 + np.sum(x > 0)) / (1 + np.sum(x < 0)))

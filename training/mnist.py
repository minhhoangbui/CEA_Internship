import urllib
import gzip
import numpy
import collections
import os
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

# Copy from 
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/learn/python/learn/datasets

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

def retry(initial_delay,
                    max_delay,
                    factor=2.0,
                    jitter=0.25,
                    is_retriable=None):
    """Simple decorator for wrapping retriable functions.
    Args:
        initial_delay: the initial delay.
        factor: each subsequent retry, the delay is multiplied by this value.
                (must be >= 1).
        jitter: to avoid lockstep, the returned delay is multiplied by a random
                number between (1-jitter) and (1+jitter). To add a 20% jitter, set
                jitter = 0.2. Must be < 1.
        max_delay: the maximum delay allowed (actual max is
                max_delay * (1 + jitter).
        is_retriable: (optional) a function that takes an Exception as an argument
                and returns true if retry should be applied.
    """
    if factor < 1:
        raise ValueError('factor must be >= 1; was %f' % (factor,))

    if jitter >= 1:
        raise ValueError('jitter must be < 1; was %f' % (jitter,))

    # Generator to compute the individual delays
    def delays():
        delay = initial_delay
        while delay <= max_delay:
            yield delay * random.uniform(1 - jitter,  1 + jitter)
            delay *= factor

    def wrap(fn):
        """Wrapper function factory invoked by decorator magic."""

        def wrapped_fn(*args, **kwargs):
            """The actual wrapper function that applies the retry logic."""
            for delay in delays():
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # pylint: disable=broad-except)
                    if is_retriable is None:
                        continue

                    if is_retriable(e):
                        time.sleep(delay)
                    else:
                        raise
            return fn(*args, **kwargs)
        return wrapped_fn
    return wrap

_RETRIABLE_ERRNOS = {
    110,  # Connection timed out [socket.py]
}


def _is_retriable(e):
    return isinstance(e, IOError) and e.errno in _RETRIABLE_ERRNOS


@retry(initial_delay=1.0, max_delay=16.0, is_retriable=_is_retriable)
def urlretrieve_with_retry(url, filename=None):
    return urllib.urlretrieve(url, filename)


def maybe_download(filename, work_directory, source_url):
    """Download the data from source url, unless it's already here.
    Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist.
    Returns:
      Path to resulting file.
    """
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)

    if not os.path.exists(filepath):
        temp_file_name, _ = urlretrieve_with_retry(source_url)
        shutil.copy(temp_file_name, filepath)
        with open(filepath) as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
    f: A file object that can be passed into a gzip reader.
    Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
    Raises:
    ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                           (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].
    Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
    Returns:
    labels: a 1D uint8 numpy array.
    Raises:
    ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                           (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)

        return labels


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self,
               images,
               labels,
               one_hot=False,
               dtype=numpy.float32,
               reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """

        assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        if dtype == numpy.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def read_data_sets(train_dir,
                   one_hot=False,
                   dtype=numpy.float32,
                   reshape=True,
                   validation_size=5000):

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    local_file = maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
    with open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    local_file = maybe_download(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
    with open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = maybe_download(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)

    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images,
                    validation_labels,
                    dtype=dtype,
                    reshape=reshape)
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

    return Datasets(train=train, validation=validation, test=test)


def load_mnist(train_dir='MNIST-data'):
    return read_data_sets(train_dir)


def reshape_mnist(raw_byte):
    image = numpy.reshape(raw_byte, (28, 28))
    return image


def show(number_to_show, images):
    fig = plt.figure()
    for i in range(1, number_to_show+1):
        fig.add_subplot(10, 20, i)
        plt.imshow(reshape_mnist(images[i-1]), cmap=mpl.cm.Greys)
        plt.axis('off')
    
    plt.show()


if __name__ == '__main__':
    datasets = load_mnist(train_dir='/tmp/mnist/')
    train_data = datasets.train
    print(train_data.images.shape)
    show(100, train_data.images)



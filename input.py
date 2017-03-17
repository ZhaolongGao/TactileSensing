"""Routine for decoding the haptic files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

# Global Parameters
NUM_CLASSES = 3

# Dimension of a single example
# SAMP_FREQ = 5000
# MILI_SEC_PER_WINDOW = 100
NUM_REC_PER_WINDOW = 500  # SAMP_FREQ * MILI_SEC_PER_WINDOW / 1000
WIDTH_MULTIPLIER = 1
CHANNELS = 1


def read_hapdata(input_queue):
    """Reads and parses examples from tactile data files.

    Args:
        input_queue: A queue of strings with the filenames
            and labels to read from.

    Returns:
        An object representing a single example. With the shape of
        [NUM_REC_PER_WINDOW, WIDTH_MULTIPLIER, CHANNELS]

    """

    class HAPRecord(object):
        pass

    result = HAPRecord()

    result.length = NUM_REC_PER_WINDOW
    result.height = NUM_REC_PER_WINDOW
    result.width = WIDTH_MULTIPLIER
    result.depth = CHANNELS

    # Read a record from filenames in input_queue
    reader = tf.TextLineReader()
    result.key, value = reader.read(
        tf.train.string_input_producer([input_queue[0]]))

    # set number of items in record_defaults as channels with tf.float32
    record_defaults = [[1.0] for _ in range(CHANNELS)]

    # read the records and batch them as window size
    record = tf.decode_csv(value, record_defaults=record_defaults)
    record_batch = tf.train.batch(record, batch_size=NUM_REC_PER_WINDOW)
    result.example = tf.reshape(record_batch,
                                [NUM_REC_PER_WINDOW, WIDTH_MULTIPLIER, CHANNELS])

    result.label = input_queue[1]

    return result


def _generate_example_and_label_batch(example, label,
                                      min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of examples and labels.

  Args:
    example: 3-D Tensor of [NUM_REC_PER_WINDOW, 1, CHANNELS] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    examples: Examples. 4D tensor of [batch_size, NUM_REC_PER_WINDOW, 1, CHANNELS] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    # num_preprocess_threads = 16
    if shuffle:
        examples, label_batch = tf.train.shuffle_batch(
            [example, label],
            batch_size=batch_size,
            # num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        examples, label_batch = tf.train.batch(
            [example, label],
            batch_size=batch_size,
            # num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    labels = tf.reshape(label_batch, [batch_size])
    return examples, labels


def inputs(batch_size, data_dir=""):
    """Construct inputs for training.
    Args:
        data_dir:
        batch_size: Number of pieces per batch.

    Returns:
        images:
        labels:
    """
    # Create filename list and parse the labels from the names
    filename_list = [os.path.join(data_dir, 'type%d.csv' % i)
                     for i in range(NUM_CLASSES)]
    label_list = [int(filename_list[i][4]) for i in range(NUM_CLASSES)]

    # Check if there are enough files compare to the number of classes
    for f in filename_list:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Convert to tensors
    filenames = tf.convert_to_tensor(filename_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

    # Create an input queue
    input_queue = tf.train.slice_input_producer([filenames, labels])

    # Read examples from files in the filename queue.
    read_input = read_hapdata(input_queue)  # **HERE**
    # reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    #
    # height = IMAGE_SIZE
    # width = IMAGE_SIZE
    #
    # # Image processing for evaluation.
    # # Crop the central [height, width] of the image.
    # resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
    #                                                        height, width)
    #
    # # Subtract off the mean and divide by the variance of the pixels.
    # float_image = tf.image.per_image_standardization(resized_image)
    #
    # # Set the shapes of tensors.
    # float_image.set_shape([height, width, 3])
    # read_input.label.set_shape([1])
    #
    # Ensure that the random shuffling has good mixing properties.
    num_examples_per_epoch = 1000
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_example_and_label_batch(read_input.example, read_input.label,
                                             min_queue_examples, batch_size,
                                             shuffle=False)

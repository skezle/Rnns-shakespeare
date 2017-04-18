import numpy as np

## https://gist.github.com/spitis/2dd1720850154b25d2cec58d4b75c4a0

###############################################################
##################### Doesnt work!!! ##########################
###############################################################
def ptb_iterator(raw_data, batch_size, num_steps, steps_ahead=1):
  """Iterate on the raw PTB data.
  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)

  batch_len = data_len // batch_size

  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  offset = 0
  if data_len % batch_size:
    offset = np.random.randint(0, data_len % batch_size)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i + offset:batch_len * (i + 1) + offset]

  epoch_size = (batch_len - steps_ahead) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+steps_ahead]
    yield (x, y)

  if epoch_size * num_steps < batch_len - steps_ahead:

      yield (data[:, epoch_size*num_steps : batch_len - steps_ahead], data[:, epoch_size*num_steps + 1:])

def shuffled_ptb_iterator(raw_data, batch_size, num_steps):
    """
    Takes the raw data and arranges in such a way that one can feed it into
    our RNN with the appropriate batch_size and num_steps
    """
    print("batch_size: {}".format(batch_size))
    print("num_steps: {}".format(num_steps))
    raw_data = np.array(raw_data, dtype=np.int32)
    ##r = len(raw_data) % num_steps
    print("shape of raw data: {}".format(raw_data.shape))
    r = len(raw_data) % (num_steps + 1) ## used to be +1 to make this work
    print("r: {}".format(r))
    ## randomly take a section of the data and delete it
    if r:
        n = np.random.randint(0, r)
        raw_data = raw_data[n:n + len(raw_data) - r] ## one long vector

    ##raw_data = np.reshape(raw_data, [-1, num_steps]) ## (x, num_steps)
    raw_data = np.reshape(raw_data, [-1, num_steps + 1]) ## +1 added to make this work
    print("shape of raw data: {}".format(raw_data.shape))
    np.random.shuffle(raw_data)

    num_batches = int(np.ceil(len(raw_data) / int(batch_size)))
    print("num_batches: {}".format(num_batches))

    for i in range(num_batches):
        data = raw_data[i*batch_size:min(len(raw_data), (i+1)*batch_size),:]
        print("yield 1: {}".format(data[:,:-1].shape))
        print("yield 2: {}".format(data[:,1:].shape))
        yield (data[:,:-1], data[:,1:])

<h3>Short description</h3>
Autoencoder implementation in TensorFlow. Visualizes the (up to 75x75 first) reconstructed values every 1000 training epochs. The background color corresponds to the absolute difference between the reconstructed value and the actual (target) value. These are typically in [0,1] by design (scaling is applied by default, but can be turned off) even though tanh is used as an activation function (so values in [-1,1] are possible in the reconstruction). We use bins corresponding to different errors, with bin 0 (dark green) being the best (small error) and red being the worst. A colored digit in the foreground indicates that the entry in question has moved to another bin (improved or worsened) during the last epoch. The digit indicates which bin it is in now.

Todo:
- choose layerwise training or not
- add weight decay
- changing the weight initialization
- specify random seed

<h3>Dependencies</h3>
- tensorflow
- pandas
- numpy

<h3>Usage</h3>
```
usage: ascii_autoencoder.py [-h] [-T] [-n N_HIDDEN_NODES [N_HIDDEN_NODES ...]]
                            [-l LEARNING_RATE] [-d DROPOUT_PROB] [-c N_CYCLES]
                            [-b BATCH_SIZE] [--no_scaling] [--no_output]
                            [--decoded_file DECODED_FILE]
                            [--encoded_file ENCODED_FILE]
                            [--scaled_input_file SCALED_INPUT_FILE]
                            fpath

Autoencoder implemented in tensorflow

positional arguments:
  fpath                 Path to file containing matrix with input examples for
                        the autoencoder

optional arguments:
  -h, --help            show this help message and exit
  -T, --transpose       Transpose input
  -n N_HIDDEN_NODES [N_HIDDEN_NODES ...], --n_hidden_nodes N_HIDDEN_NODES [N_HIDDEN_NODES ...]
                        Number of hidden nodes in each hidden layer
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate
  -d DROPOUT_PROB, --dropout_prob DROPOUT_PROB
                        Dropout probability
  -c N_CYCLES, --n_cycles N_CYCLES
                        Number of training cycles
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of training examples per cycle
  --no_scaling          Do not scale input (use this if you have already
                        scaled it)
  --no_output           Skip writing encoded and decoded values to file?
  --decoded_file DECODED_FILE
                        Name of file to write decoded (reconstructed) values
                        to
  --encoded_file ENCODED_FILE
                        Name of file to write encoded values to
  --scaled_input_file SCALED_INPUT_FILE
                        Name of file to write scaled input values to
```

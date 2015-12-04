import tensorflow as tf
import numpy as np
import math
import pandas as pd
import sys
from col import print_row, print_row2
import argparse

def scale01(vec):
	"""
	Scale the input values a vector to the interval [0, 1]
	"""
	# Check if all the values are the same - then uninformative so do not scale, to avoid division by zero
	if vec.min() == vec.max(): return vec
	return( (vec-vec.min()) / float(vec.max()-vec.min()) )

def corrupt(array, corr):
	"""
	Set a fraction ('corr') of the entries in a matrix to zero. May or may not be equivalent to a dropout function in tensorflow.
	"""
	array = np.array(array)
	for row in array:
		row[np.random.choice(len(row), int(corr*len(row)))] = 0.0
	return array

def dump_file(tens, fname):
	"""
	Write the contents of a tensor (for instance, the encoded or decoded inputs) to file.
	"""
	outfile = open(fname, 'w')
	for row in tens:
		for i in range(len(row)):
			outfile.write(str(row[i]))
			if i == len(row)-1: outfile.write('\n')
			else: outfile.write('\t')

def create(x, layer_sizes):
	"""
	Borrowed from Salik Syed https://gist.github.com/saliksyed/593c950ba1a3b9dd08d5
	"""
        # Build the encoding layers
        next_layer_input = x
        encoding_matrices = []
        for dim in layer_sizes:
                input_dim = int(next_layer_input.get_shape()[1])
                # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
                W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))
                # Initialize b to zero
                b = tf.Variable(tf.zeros([dim]))
                # We are going to use tied-weights so store the W matrix for later reference.
                encoding_matrices.append(W)
                output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)
                # the input into the next layer is the output of this layer
                next_layer_input = output
        # The fully encoded x value is now stored in the next_layer_input
        encoded_x = next_layer_input
        # build the reconstruction layers by reversing the reductions
        layer_sizes.reverse()
        encoding_matrices.reverse()
        for i, dim in enumerate(layer_sizes[1:] + [ int(x.get_shape()[1])]) :
                # we are using tied weights, so just lookup the encoding matrix for this step and transpose it
                W = tf.transpose(encoding_matrices[i])
                b = tf.Variable(tf.zeros([dim]))
                output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)
                next_layer_input = output
        # the fully encoded and reconstructed value of x is here:
        reconstructed_x = next_layer_input
        return {
                'encoded': encoded_x,
                'decoded': reconstructed_x,
                'cost' : tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))
        }

parser = argparse.ArgumentParser(description='Autoencoder implemented in tensorflow')
parser.add_argument('fpath', help='Path to file containing matrix with input examples for the autoencoder')
parser.add_argument('-T', '--transpose', action='store_true', help='Transpose input')
parser.add_argument('-n', '--n_hidden_nodes', default=[2], nargs='+', help='Number of hidden nodes in each hidden layer',type=int)
parser.add_argument('-l', '--learning_rate', default=0.1, help='Learning rate',type=float)
parser.add_argument('-d', '--dropout_prob', default=0.2, help='Dropout probability',type=float)
parser.add_argument('-c', '--n_cycles', default=50000, help='Number of training cycles',type=int)
parser.add_argument('-b', '--batch_size', default=500, help='Number of training examples per cycle',type=int)
parser.add_argument('--no_scaling', action='store_true', help='Do not scale input (use this if you have already scaled it)')
parser.add_argument('--no_output', action='store_true', help='Skip writing encoded and decoded values to file?')
parser.add_argument('--decoded_file', default='decoded.tsv', help='Name of file to write decoded (reconstructed) values to')
parser.add_argument('--encoded_file', default='encoded.tsv', help='Name of file to write encoded values to')
parser.add_argument('--scaled_input_file', default='scaled_input.tsv', help='Name of file to write scaled input values to')

# Parse input arguments
args = parser.parse_args()
data = pd.read_csv(args.fpath, sep="\t", index_col=0)
n_hidden = args.n_hidden_nodes
learning_rate = args.learning_rate 
corrupt_level = args.dropout_prob 

# We want to scale the data per feature, so the scaling will be different depending on the input - examples x features or features x examples?
if args.transpose: # in the input file, examples are in the columns and features are in the rows
	if args.no_scaling:
		input_data = np.array(data.T)
	else:
		input_data = np.array(map(scale01, np.array(data))).T

else: # examples are in rows, features are in columns
	if args.no_scaling:
		input_data = np.array(data)
	# Scale columnwise, ie transpose, scale, transpose back
	else:
		input_data = np.array(map(scale01,np.array(data.T))).T
		#input_data = np.array(map(scale01,np.array(data.T)))

print(input_data.shape)
print(input_data[0])

n_samp, n_input = input_data.shape
print("No of samples: " + str(n_samp)) 
print("No of inputs: " + str(n_input)) 
n_rounds = args.n_cycles
batch_size = min(args.batch_size, n_samp)

x = tf.placeholder("float", [None, n_input])
ae = create(x, n_hidden)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

bins = np.array([0.01,0.05,0.1,0.15,0.20,0.30,0.50,1.0])
vis_input = min(n_input, 75)
vis_samples = min(n_samp, 75)
prev_diff = None

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(ae['cost'])

for i in range(n_rounds):    
    sample = np.random.randint(n_samp, size=batch_size)
    batch_xs = input_data[sample][:]    
    batch_xs = corrupt(batch_xs, corrupt_level)
    sess.run(train_step, feed_dict={x: np.array(batch_xs)})
    
    if i % 1000 == 0:
         diff = abs(sess.run(ae['decoded'], feed_dict={x:np.array(input_data)})[0:vis_samples,0:vis_input] - input_data[0:vis_samples,0:vis_input])
         for j in range(0,len(diff)):
             discr = np.digitize(diff[j], bins)
             if prev_diff is not None:
                 discr_prev = np.digitize(prev_diff[j], bins)
             else:
                 discr_prev = None
             print_row2(discr, discr_prev)
         print ''
         prev_diff = diff
         print i, sess.run(ae['cost'], feed_dict={x: np.array(input_data)})

reconstr = sess.run(ae['decoded'], feed_dict={x: input_data})
encoded = sess.run(ae['encoded'], feed_dict={x: input_data})

if not args.no_output:
	dump_file(reconstr, args.decoded_file)
	dump_file(encoded, args.encoded_file)
	dump_file(input_data, args.scaled_input_file)


import warnings
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from distutils.version import LooseVersion

def tf_checks():
	# Check TensorFlow Version
	print()
	assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
	print('TensorFlow Version: {}'.format(tf.__version__))

	print()

	# Check for a GPU
	if not tf.test.gpu_device_name():
	    print('Warning: No GPU found. Please use a GPU to train your neural network.')
	else:
	    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

	print('\n\n')


def get_song_matrixes(path, num_songs):
	# Songs matrix configurations
	files=glob.glob('{}/*.*mid*'.format(path))
	input_songs=[]
	target_songs=[]

	# Converting songs from midi to matrix
	print('[*] Converting songs to matrix')
	for i, f in enumerate(tqdm(files)):
	    song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
	    if np.array(song).shape[0] > 50:    
	        input_songs.append(song)
	        target_songs.append(song)
	    if i == num_songs:
	        break
	return (input_songs, target_songs)


def get_data_insights(input_songs, target_songs):
	# Finding the longest song in the dataset
	# Finding the number of tokens in the songs (usually 156)
	max_encoder_seq_length = max([len(song) for song in input_songs])
	num_encoder_tokens = max([song.shape[1] for song in input_songs])
	max_decoder_seq_length = max([len(song) for song in target_songs])
	num_decoder_tokens = max([song.shape[1] for song in target_songs])

	print('[*] Converted {} songs to matrix'.format(i))
	print('\n\n')

	return (max_encoder_seq_length,num_encoder_tokens,max_decoder_seq_length,num_decoder_tokens)


def get_input_data(input_songs, target_songs, max_encoder_seq_length, num_encoder_tokens, max_decoder_seq_length, num_decoder_tokens):
	# Creating the input placeholder for the encoder
	encoder_input_data = np.zeros(
	                        (len(input_songs), 
	                        max_encoder_seq_length, 
	                        num_encoder_tokens),
	                        dtype='float32')

	# Creating the input placeholders for the decoder
	decoder_input_data = np.zeros(
	                        (len(input_songs), 
	                        max_decoder_seq_length, 
	                        num_decoder_tokens),
	                        dtype='float32')
	decoder_target_data = np.zeros(
	                        (len(input_songs), 
	                        max_decoder_seq_length, 
	                        num_decoder_tokens),
	                        dtype='float32')

	# converting the song data into a shape that the encoder
	# and the decoder can understand: (num_samples, max_seq_length, num_tokens)
	for i, (input_song, target_song) in enumerate(zip(input_songs, target_songs)):
	    
	    encoder_input_data[i] = np.concatenate((input_song, 
	            np.zeros((max_encoder_seq_length-input_song.shape[0], num_encoder_tokens))))
	    encoder_input_data[i, -1, -1] = 1

	    # decoder_target_data is ahead of decoder_input_data by one timestep
	    decoder_input_data[i] = np.concatenate((target_song, 
	            np.zeros((max_decoder_seq_length-target_song.shape[0], num_decoder_tokens))))
	    decoder_input_data[i, 0, 0] = 1

	    decoder_target_data[i] = np.concatenate((target_song, 
	            np.zeros((max_decoder_seq_length-target_song.shape[0], num_decoder_tokens))))
	    decoder_target_data[i, -1, -1] = 1

	print()
	print('Encoder input data shape:',encoder_input_data.shape)
	print('Decoder input data shape:',decoder_input_data.shape)
	print('Decoder target data shape:',decoder_target_data.shape)
	print()


	return (encoder_input_data, decoder_input_data, decoder_target_data)

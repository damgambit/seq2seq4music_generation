import warnings
import tensorflow as tf
import glob
from tqdm import tqdm
import midi_manipulation
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


def get_song_matrixes(path, num_songs, seq_length):
    # Songs matrix configurations
    files=glob.glob('{}/*.*mid*'.format(path))
    input_songs=[]
    target_songs=[]

    # Converting songs from midi to matrix
    print('[*] Converting songs to matrix')
    for i, f in enumerate(tqdm(files)):
        song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
        if np.array(song).shape[0] > 50:   
            length = np.array(song).shape[0]
            for j in range(length // seq_length):
                input_songs.append(song[seq_length*j:seq_length*(j+1)])
                target_songs.append(song[seq_length*j:seq_length*(j+1)])
        if i == num_songs:
            break
    print('[*] Converted {} songs to matrix'.format(i))
    print('\n\n')
    
    return (input_songs, target_songs)


def token_to_state(idx, tokens):
	return tokens[idx]

def state_to_token(tokens, state):
	idx = np.argwhere((tokens[:]==state).all(1) == True)[0][0]
	return idx

def embed_to_state(embed, tokens):
	idx = np.argmax(embed)
	return tokens[idx]

def embed_song_to_song(embed_song, tokens):
	song = []
	for embed in embed_song:
		state = embed_to_state(embed, tokens)
		song.append(state)
	return song

def song_to_embed_song(song, tokens):
	embed_song = []
	for i, state in enumerate(song):
		idx = state_to_token(state, tokens)
		embed = np.zeros(num_encoder_tokens)
		embed[idx] = 1
		embed_song.append(embed)
	return embed_song


def get_tokens(input_songs):
	tokens = []
	print(input_songs[0].shape)

	tokens.append(np.zeros((156)))


	for i, song in enumerate(input_songs):
		if(i%50==0):
			print('Processing song: {}/{}'.format(i,np.array(input_songs).shape[0]))
		embed_song = []
		for i, state in enumerate(song):
			if not any((tokens[:]==state).all(1)):
				tokens.append(state)

	return tokens

def get_embeded_songs(input_songs, tokens, num_encoder_tokens):
	embeded_songs = []
	for i, song in enumerate(input_songs):
		if(i%50==0):
			print('Processing embed: {}/{}'.format(i,np.array(input_songs).shape[0]))
		embed_song = []
		for i, state in enumerate(song):
			idx = state_to_token(state, tokens)
			embed = np.zeros(num_encoder_tokens)
			embed[idx] = 1
			embed_song.append(embed)
		embeded_songs.append(embed_song)

	return embeded_songs


def get_data_insights(input_songs, target_songs):


	# Finding the longest song in the dataset
	# Finding the number of tokens in the songs (usually 156)
	max_encoder_seq_length = max([len(song) for song in input_songs])
	num_encoder_tokens = max([song.shape[1] for song in input_songs])
	max_decoder_seq_length = max([len(song) for song in target_songs])
	num_decoder_tokens = max([song.shape[1] for song in target_songs])

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

	print(encoder_input_data.shape, np.array(input_songs).shape)
	# converting the song data into a shape that the encoder
	# and the decoder can understand: (num_samples, max_seq_length, num_tokens)
	for i, (input_song) in enumerate(input_songs):
	    

	    # decoder_target_data is ahead of decoder_input_data by one timestep
	    for t, data in enumerate(input_song):
	    	encoder_input_data[i, t] = data
	    
	    	decoder_target_data[i, t] = data

	    	if t > 0:
	    		decoder_input_data[i, t-1] = data

	    encoder_input_data[i, -1, -1] = 1
	    decoder_input_data[i, 0, 0] = 1
	    decoder_target_data[i, -1, -1] = 1

	    

	print()
	print('Encoder input data shape:',encoder_input_data.shape)
	print('Decoder input data shape:',decoder_input_data.shape)
	print('Decoder target data shape:',decoder_target_data.shape)
	print()


	return (encoder_input_data, decoder_input_data, decoder_target_data)

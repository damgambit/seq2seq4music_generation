import warnings
import tensorflow as tf
import glob
from tqdm import tqdm
import midi_manipulation
import numpy as np
from tensorflow.python.ops import control_flow_ops
from distutils.version import LooseVersion
from utils import *



input_songs, target_songs = get_song_matrixes('./blues', 30, 300)

input_songs = np.array(input_songs)
target_songs = np.array(target_songs)


tokens = get_tokens(input_songs)
num_encoder_tokens = np.array(tokens).shape[0]
num_decoder_tokens = np.array(tokens).shape[0]


print('[*] Embedding Songs')
embeded_input_songs = get_embeded_songs(input_songs, tokens, num_encoder_tokens)
embeded_target_songs = get_embeded_songs(input_songs, tokens, num_encoder_tokens)

print(np.array(embeded_input_songs[0]).shape)


print(np.array(embed_song_to_song(embeded_input_songs[0], tokens)).shape)

# Finding the longest song in the dataset
max_encoder_seq_length = max([len(song) for song in embeded_input_songs])
max_decoder_seq_length = max([len(song) for song in embeded_target_songs])


print('Number of samples:', len(input_songs))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# Get input data in shape (num_sample, max_seq_length, num_tokens)
encoder_input_data, decoder_input_data, decoder_target_data = get_input_data(
                                                                        embeded_input_songs, 
                                                                        embeded_target_songs,
                                                                        max_encoder_seq_length, 
                                                                        num_encoder_tokens, 
                                                                        max_decoder_seq_length, 
                                                                        num_decoder_tokens)
from keras.models import Model
from keras.layers import Input, LSTM, Dense

# Hyperparameters
batch_size  =   16      # Batch size for training.
epochs      =   15     # Number of epochs to train for.
latent_dim  =   256     # Latent dimensionality of the encoding space.
#num_samples =   10000   # Number of samples to train on.


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)



# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print()
print(model.summary())
print()


# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracj'])

print('[*] Starting Training')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# model.load_weights('s2s.h5')
print('[*] Ready to be used \n\n')
# Save model
model.save('s2s.h5')



# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 0] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    song_matrix = np.zeros(
                        (max_decoder_seq_length, 
                        num_decoder_tokens),
                        dtype='float32')
    i = 0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        idx = np.argmax(output_tokens[-1,-1,:])
        #print(output_tokens[-1,-1,:])
        song_matrix[i, idx] = 1
        target_seq[0, 0, idx] = 1.
        #print(np.array(h).shape)
        # Exit condition: either hit max length
        # or find stop character.
        if (i+2 > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        
        

        if i%250 == 0:
            print('[iter:{}] [max_decoder_seq_length: {}]'.format(i, max_decoder_seq_length))
        i+=1

        # Update states
        states_value = [h, c]

    return song_matrix


seq_length = max_encoder_seq_length

song = np.array(midi_manipulation.midiToNoteStateMatrix('./blues/BB_King_-_Sweet_Sixteen._mid_'))

encoder_input_data = []
if np.array(song).shape[0] > 50:   
    length = np.array(song).shape[0]
    for j in range(length // seq_length):
        encoder_input_data.append(song[seq_length*j:seq_length*(j+1)])
encoder_input_data = get_embeded_songs(encoder_input_data, tokens, num_encoder_tokens)                                                       


# Take one sequence (part of the training test)
# for trying out decoding.
print('[*] Encoding-Decoding')
input_seq = encoder_input_data[0:-1]
decoded_song = decode_sequence(input_seq)

decoded_song = embed_song_to_song(decoded_song, tokens)

# Converting Song to midi from matrix
print('[*] Converting and saving song')
midi_manipulation.noteStateMatrixToMidi(decoded_song)
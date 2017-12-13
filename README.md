# seq2seq4music_generation
In development...

## Videos
### Music+AI - Blues Music played By an Artificial Intelligence - #1 Generation 
https://www.youtube.com/watch?v=SMJ2pqAllyQ&t=32s

Hi I'm Damien and this is going to be a great series about how an Artificial Intelligence can learn to generate various styles of Music. We will start from blues music!

In this #1 Generation the AI has been trained only on 40 songs, with a relative small network. I've used the state of the art of Deep Learning for sequence modelling, the Encoder/Decoder architecture seq2seq.
The songs are midi files vectorized with the following shape: (seq_length, 156). Every "word" of the song has 156*156 possible combination, and for that every word needs to tokenized in order to properly train the network. We end up with a shape of (seq_length, max_token_number) for each song/seq_length passed to the model.

For further information about the model and how it does work, feel free to contact me or wait for more in dept explaination video/article. 

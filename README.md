# Recurrent Neural Network for Language Translation

I built a language translator algorithm using a Sequence to Sequence model Recurrent Neural Network.

The RNN is built on TensorFlow, written in Python 3 and is presented via Jupyter Notebook. Trained via cloud on FloydHub's gpus. 

The following are some of the steps I took to build this RNN:

Preprocessing

- Convert source and target text to proper word ids

Build the Neural Network: Here I build the components necessary for a Sequence-to-Sequence model with the functions below:

model_inputs:  Create TF Placeholders for input, targets, and learning rate.
process_decoding_input: remove last word id from each target_data batch, concat GO ID to beginning. Use TF tf.strided_slice and tf.concat. 
encoding_layer: create an Encoder RNN layer using tf.nn.dynamic_rnn()
decoding_layer_train: training logits w/ tf.contrib.seq2seq.simple_decoder_fn_train() and tf.contrib.seq2seq.dynamic_rnn_decoder(). 
decoding_layer_infer: inference logits w/ tf.contrib.seq2seq.simple_decoder_fn_inference() and tf.contrib.seq2seq.dynamic_rnn_decoder().
decoding_layer: implement the decoding_layer() function to create a Decoder RNN layer.
seq2seq_model: this implements all of the above functions to build the neural network model


Training the network

Hyperparameters: epochs, batch size, rnn size, num_layers, encoding_embedding_size, decoding_embedding_size, learning rate, keep_prob
Training: Trained the neural network and achieved a training and validation accuracy rate of 94% after just 10 epochs.

Translation

To feed a sentence into the model for translation, we first need to preprocess it. I implement the function sentence_to_seq() to preprocess new sentences.

Finally at this point the neural network can translate english phrases entered in the variable translate_sentence from English to French. Note: Since this is a very small corpus you'll need to only pick words from the dataset used in order to get an accurate translation.

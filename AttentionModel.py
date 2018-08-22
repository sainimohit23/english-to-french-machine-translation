import tensorflow as tf
import numpy as np
import pickle
import datetime

def load_preprocess():
    with open('preprocess_en_fr.p', mode='rb') as in_file:
        return pickle.load(in_file)


(source_int_text, target_int_text), (source_int_to_vocab, target_int_to_vocab) , (source_vocab_to_int, target_vocab_to_int) = load_preprocess()

# =============================================================================
#       STEPS INVOLVED:

#     (1) define input parameters to the encoder model
#         enc_dec_model_inputs
#     (2) build encoder model
#         encoding_layer
#     (3) define input parameters to the decoder model
#         enc_dec_model_inputs, process_decoder_input, decoding_layer
#     (4) build decoder model for training
#         decoding_layer_train
#     (5) build decoder model for inference
#         decoding_layer_infer
#     (6) put (4) and (5) together
#         decoding_layer
#     (7) connect encoder and decoder models
#         seq2seq_model
#     (8) train and estimate loss and accuracy
# 
# =============================================================================

def enc_dec_model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets') 
    
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    source_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)    
    
    return inputs, targets, target_sequence_length, max_target_len, source_sequence_length

def hyperparam_inputs():
    lr_rate = tf.placeholder(tf.float32, name='lr_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return lr_rate, keep_prob

def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :return: Preprocessed target data
    """
    # get '<GO>' id
    go_id = target_vocab_to_int['<GO>']
    
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)
    
    return after_concat


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_vocab_size, 
                   encoding_embedding_size,
                   source_sequence_length):
    """
    :return: tuple (RNN output, RNN state)
    """
    embed = tf.contrib.layers.embed_sequence(rnn_inputs, 
                                             vocab_size=source_vocab_size, 
                                             embed_dim=encoding_embedding_size)
    
    stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])
    
    outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_cells, 
                                                             cell_bw=stacked_cells, 
                                                             inputs=embed, 
                                                             sequence_length=source_sequence_length, 
                                                             dtype=tf.float32)
    
    concat_outputs = tf.concat(outputs, 2)
    return concat_outputs, state

def decoding_layer_train(encoder_outputs, encoder_state, dec_cell, dec_embed_input, 
                         target_sequence_length, max_summary_length, 
                         output_layer, keep_prob):
    """
    Create a training process in decoding layer 
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    
    train_helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)
    
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, encoder_outputs,
                                                               memory_sequence_length=target_sequence_length)
    
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism,
                                                         attention_layer_size=rnn_size/2)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_cell, helper=train_helper, 
                                              initial_state=attention_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                                              output_layer=output_layer) 
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_summary_length)
    
    return outputs



def decoding_layer_infer(encoder_outputs, encoder_state, dec_cell,
                         dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob,
                         target_sequence_length):
    """
    Create a inference process in decoding layer 
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
                                                      tf.fill([batch_size], start_of_sequence_id), 
                                                      end_of_sequence_id)
    
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, encoder_outputs,
                                                               memory_sequence_length=target_sequence_length)
    
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism,
                                                         attention_layer_size=rnn_size/2)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_cell, helper=infer_helper, 
                                              initial_state=attention_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                                              output_layer=output_layer)
   
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_target_sequence_length)
    
    return outputs

def decoding_layer(encoder_outputs, dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    target_vocab_size = len(target_vocab_to_int) + 1
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_outputs,
                                            encoder_state, 
                                            cells, 
                                            dec_embed_input, 
                                            target_sequence_length, 
                                            max_target_sequence_length, 
                                            output_layer, 
                                            keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_outputs,
                                            encoder_state, 
                                            cells, 
                                            dec_embeddings, 
                                            target_vocab_to_int['<GO>'], 
                                            target_vocab_to_int['<EOS>'], 
                                            max_target_sequence_length, 
                                            target_vocab_size, 
                                            output_layer,
                                            batch_size,
                                            keep_prob,
                                            target_sequence_length)

    return (train_output, infer_output)

def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int,
                  source_sequence_length):
    """
    Build the Sequence-to-Sequence model
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data, 
                                             rnn_size, 
                                             num_layers, 
                                             keep_prob, 
                                             source_vocab_size, 
                                             enc_embedding_size,
                                             source_sequence_length)
    
    dec_input = process_decoder_input(target_data, 
                                      target_vocab_to_int, 
                                      batch_size)
    
    train_output, infer_output = decoding_layer(enc_outputs,
                                                dec_input,
                                               enc_states, 
                                               target_sequence_length, 
                                               max_target_sentence_length,
                                               rnn_size,
                                              num_layers,
                                              target_vocab_to_int,
                                              target_vocab_size,
                                              batch_size,
                                              keep_prob,
                                              dec_embedding_size)
    
    return train_output, infer_output


display_step = 30

epochs = 20
batch_size = 32

rnn_size = 128
num_layers = 3

encoding_embedding_size = 200
decoding_embedding_size = 200

learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

save_path = 'checkpoints/dev'
(source_int_text, target_int_text), _, (source_vocab_to_int, target_vocab_to_int) = load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, target_sequence_length, max_target_sequence_length, source_sequence_length = enc_dec_model_inputs()
    lr, keep_prob = hyperparam_inputs()
    
    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int,
                                                   source_sequence_length)
    
    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
    # - Returns a mask tensor representing the first N positions of each cell.
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    
    

    with tf.name_scope("optimization"):
        # Loss function - weighted softmax cross entropy
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        
        tf.summary.scalar('loss', cost)
        merged = tf.summary.merge_all()
        logdir = 'tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/" 
        
        
def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths
        
        
def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))



def sentence_to_seq(sentence, vocab_to_int):
    results = []
    for word in sentence.split(" "):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])
            
    return results

sentances = ["how are you",
             "what are you doing",
             "get up",
             "go there",
             "want some food"]


# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,
                                                                                                             valid_target,
                                                                                                             batch_size,
                                                                                                             source_vocab_to_int['<PAD>'],
                                                                                                             target_vocab_to_int['<PAD>']))



with tf.Session(graph=train_graph) as sess:
  writer = tf.summary.FileWriter(logdir, train_graph)
  saver = tf.train.Saver()
  try:
      saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
      print("Saved model found")
  except ValueError:
      print("No saved models found, initializing new variables")
      sess.run(tf.global_variables_initializer())

  for epoch_i in range(10,epochs):
      for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
              get_batches(train_source, train_target, batch_size,
                          source_vocab_to_int['<PAD>'],
                          target_vocab_to_int['<PAD>'])):

          _, loss = sess.run(
              [train_op, cost],
              {input_data: source_batch,
               targets: target_batch,
               lr: learning_rate,
               target_sequence_length: targets_lengths,
               keep_prob: keep_probability,
               source_sequence_length: sources_lengths})


          if batch_i % display_step == 0 and batch_i > 0:
              batch_train_logits = sess.run(
                  inference_logits,
                  {input_data: source_batch,
                   target_sequence_length: targets_lengths,
                   keep_prob: 1.0,
                   source_sequence_length: sources_lengths})

              batch_valid_logits = sess.run(
                  inference_logits,
                  {input_data: valid_sources_batch,
                   target_sequence_length: valid_targets_lengths,
                   keep_prob: 1.0,
                   source_sequence_length: valid_sources_lengths})

              train_acc = get_accuracy(target_batch, batch_train_logits)
              valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)       


              print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                    .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))


          if batch_i%30 == 0 and batch_i > 0:
              idx = np.random.randint(0,5)
              st = sentence_to_seq(sentances[idx], source_vocab_to_int)

              trans_logits = sess.run(inference_logits, feed_dict={input_data: [st]*batch_size,
                                       target_sequence_length: [len(st)*2]*batch_size,
                                       source_sequence_length: [len(st)*2]*batch_size,
                                       keep_prob: 1.0})[0]
              cst = sess.run(merged,{input_data: source_batch,
                                   targets: target_batch,
                                   lr: learning_rate,
                                   target_sequence_length: targets_lengths,
                                   source_sequence_length: sources_lengths,
                                   keep_prob: keep_probability})
              writer.add_summary(cst, batch_i)                

              print('Input')
              print('  Word Ids:      {}'.format([i for i in st]))
              print('  input : {}'.format([source_int_to_vocab[i] for i in st]))

              print('\nPrediction')
              print('  Word Ids:      {}'.format([i for i in trans_logits]))
              print('  reply: {}'.format(" ".join([target_int_to_vocab[i] for i in trans_logits])))

      learning_rate *= learning_rate_decay
      if learning_rate < min_learning_rate:
          learning_rate = min_learning_rate
      # Save Model
      saver.save(sess, 'checkpoints/dev',epoch_i)
      print('Model Trained and Saved')

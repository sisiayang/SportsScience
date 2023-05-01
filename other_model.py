from typing import Tuple, Dict, Any
import tensorflow as tf
from other_models.prosenet.model import ProSeNet
from keras.layers import Dense
from keras_ordered_neurons import ONLSTM
from keras_pos_embd import TrigPosEmbedding
from other_models.keras_transformer import get_encoders
from other_models.keras_transformer.gelu import gelu
from model import CNN_with_mask
import keras_self_attention
from keras.layers import GRU

def bilstm(shot_sequence_shape: Tuple[int, int, int],
        rally_info_shape: int = None,
        rnn_structure: str = 'lstm',
        bidirectional_rnn: bool = True,
        rnn_kwargs: Dict[str, Any] = {'units': 32},
        dense_kwargs: Dict[str, Any] = {}
        ) -> tf.keras.Model:
    """Create a RNN-based rally classifier model."""
    rnns = {'gru': tf.keras.layers.GRU,
            'lstm': tf.keras.layers.LSTM}
    # Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape,
                                 name='Shots_input')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')

    embed_shot_layer = CNN_with_mask(kernel_size=3, filters=shot_sequence_shape[2], strides=3, name='Shot_Embedding')

    rnn = rnns.get(rnn_structure, list(rnns.values())[0])
    layer_rnn = rnn(name='Recurrent_layer', **rnn_kwargs, return_sequences=True)
    if bidirectional_rnn:
        layer_rnn = tf.keras.layers.Bidirectional(
            layer_rnn, name='Bidirectional_recurrent_layer')
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=rally_info_shape,
                                     name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(
            name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None
    layer_dense = tf.keras.layers.Dense(units=3, activation='sigmoid', 
                                        **dense_kwargs)
    # Forward pass
    inputs = [input_shots]
    masked_sequence = layer_masking(input_shots)

    embed_shot = tf.squeeze(embed_shot_layer(masked_sequence), axis=2)

    rally_represent = layer_rnn(embed_shot)
    if rally_info_shape is not None:
        inputs.append(input_rally)
        rally_represent = layer_concat_rally([rally_represent, input_rally])
    output_win_prob = layer_dense(rally_represent)
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    return model_predict


def deepmoji(shot_sequence_shape: Tuple[int, int, int],
             rally_info_shape: int = None,
             rnn_kwargs: Dict[str, Any] = {'units': 32},
             attention_kwargs: Dict[str, Any] = {},
             dense_kwargs: Dict[str, Any] = {}
             ) -> tf.keras.Model:
    """Create DeepMoji rally classifier model."""
    # Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape,
                                 name='Shots_input')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')

    embed_shot_layer = CNN_with_mask(kernel_size=3, filters=shot_sequence_shape[2], strides=3, name='Shot_Embedding')

    layer_rnn1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(return_sequences=True, **rnn_kwargs),
        name='Bidirectional_recurrent_layer1')
    layer_rnn2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(return_sequences=True, **rnn_kwargs),
        name='Bidirectional_recurrent_layer2')
    layer_concat_input_rnn = tf.keras.layers.Concatenate(
        name='Input_rnn_merging')
    # layer_attention = keras_self_attention.SeqWeightedAttention(
        # return_attention=True, **attention_kwargs)
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=rally_info_shape,
                                     name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(
            name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None

    gru_layer = GRU(units=16, return_sequences=True, name='GRU_Layer')

    layer_dense = tf.keras.layers.Dense(units=3, activation='sigmoid',
                                        **dense_kwargs)
    # Forward pass
    inputs = [input_shots]
    masked_sequence = layer_masking(input_shots)
    embed_shot = tf.squeeze(embed_shot_layer(masked_sequence), axis=2)

    hidden_states = layer_rnn2(layer_rnn1(embed_shot))
    input_states = layer_concat_input_rnn([embed_shot,
                                           hidden_states])
    # rally_represent, contributions = layer_attention(input_states)
    # if rally_info_shape is not None:
    #     inputs.append(input_rally)
    #     rally_represent = layer_concat_rally([rally_represent, input_rally])

    gru_output = gru_layer(input_states)

    output_win_prob = layer_dense(gru_output)
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    # model_attention = tf.keras.Model(inputs=inputs, outputs=contributions)
    return model_predict


def prosenet_model(shot_sequence_shape: Tuple[int, int, int],
             prosenet_kwargs: Dict[str, Any] = {'k': 10},
             rnn_kwargs: Dict[str, Any] = {'layer_type' : 'lstm',
                                           'layer_args' : {},
                                           'layers' : [32, 32],
                                           'bidirectional' : True}, 
             nclasses=2, 
             ) -> tf.keras.Model:
    """Create a ProSeNet rally classifier model."""
    
    # add a CNN layer first
    input_shots = tf.keras.Input(shape=shot_sequence_shape, name='Shots_input')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')
    embed_shot_layer = CNN_with_mask(kernel_size=3, filters=shot_sequence_shape[2], strides=3, name='Shot_Embedding')
    model_predict = ProSeNet(input_shape=(shot_sequence_shape[0], shot_sequence_shape[2]), nclasses=nclasses, rnn_args=rnn_kwargs, **prosenet_kwargs)

    # classifier = Dense(
    #         nclasses,
    #         activation='softmax',
    #         use_bias=False,
    #         kernel_constraint=lambda w: tf.nn.relu(w),
    #         name='classifier'
    #     )
    
    inputs = input_shots
    masked_sequence = layer_masking(inputs)
    embed_shot = tf.squeeze(embed_shot_layer(masked_sequence), axis=2)
    output = model_predict(embed_shot)

    output = tf.clip_by_value(output, clip_value_min=tf.float32.min, clip_value_max=tf.float32.max)

    model = tf.keras.Model(inputs=inputs, outputs=output, name='Classification')
    return model


def onlstm(shot_sequence_shape: Tuple[int, int, int],
           rally_info_shape: int = None,
           onlstm_kwargs: Dict[str, Any] = {'units': 32, 'chunk_size': 4, 'return_sequences': True},
           dense_kwargs: Dict[str, Any] = {}
           ) -> tf.keras.Model:
    """Create an ON-LSTM rally classifier model."""
    rnns = {'gru': tf.keras.layers.GRU,
            'lstm': tf.keras.layers.LSTM}
    # Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape,
                                  name='Shots_input')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')
    embed_shot_layer = CNN_with_mask(kernel_size=3, filters=shot_sequence_shape[2], strides=3, name='Shot_Embedding')
    layer_onlstm = ONLSTM(name='ONLSTM', **onlstm_kwargs)
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=rally_info_shape,
                                     name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(
            name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None
    layer_dense = tf.keras.layers.Dense(units=3, activation='sigmoid',
                                        **dense_kwargs)
    # Forward pass
    inputs = [input_shots]
    masked_sequence = layer_masking(input_shots)
    embed_shot = tf.squeeze(embed_shot_layer(masked_sequence), axis=2)
    rally_represent = layer_onlstm(embed_shot)
    if rally_info_shape is not None:
        inputs.append(input_rally)
        rally_represent = layer_concat_rally([rally_represent, input_rally])
    output_win_prob = layer_dense(rally_represent)
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    return model_predict


def transformer(shot_sequence_shape: Tuple[int, int, int],
                rally_info_shape: int = None,
                transformer_kwargs: Dict[str, Any] = {'encoder_num': 2,
                                                      'head_num': 2,
                                                      'hidden_dim': 32,
                                                      'feed_forward_activation': gelu},
                dense_kwargs: Dict[str, Any] = {}
           ) -> tf.keras.Model:
    """Create an ON-LSTM rally classifier model."""
    rnns = {'gru': tf.keras.layers.GRU,
            'lstm': tf.keras.layers.LSTM}
    # Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape,
                                 name='Shots_input')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')
    embed_shot_layer = CNN_with_mask(kernel_size=3, filters=shot_sequence_shape[2], strides=3, name='Shot_Embedding')

    layer_pos_embed = TrigPosEmbedding(mode=TrigPosEmbedding.MODE_ADD)
    # layer_pooling = tf.keras.layers.GlobalMaxPooling1D()
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=rally_info_shape,
                                     name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(
            name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None
    layer_dense = tf.keras.layers.Dense(units=3, activation='sigmoid',
                                        **dense_kwargs)
    # Forward pass
    inputs = [input_shots]
    masked_sequence = layer_masking(input_shots)

    embed_shot = tf.squeeze(embed_shot_layer(masked_sequence), axis=2)

    pos_embed_seq = layer_pos_embed(embed_shot)
    encoder_result = get_encoders(input_layer=pos_embed_seq, **transformer_kwargs)
    # rally_represent = layer_pooling(encoder_result)
    if rally_info_shape is not None:
        inputs.append(input_rally)
        rally_represent = layer_concat_rally([rally_represent, input_rally])
    output_win_prob = layer_dense(encoder_result)
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    return model_predict

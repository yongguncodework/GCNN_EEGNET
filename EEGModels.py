from keras.models import Model
from keras.layers.core import Dense, Activation, Permute, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Flatten, merge, Reshape
from keras.layers import DepthwiseConv2D
import keras.backend as K
from keras.layers.merge import Concatenate

def square(x):
    return K.square(x)

def log(x):
    return K.log(x)    
    
def EEGNet(nb_classes, Chans = 64, Samples = 128, regRate = 0.0001,
           dropoutRate = 0.25, kernels = [(2, 32), (8, 4)], strides = (2, 4)):
    """ Keras implementation of EEGNet (arXiv 1611.08024)
    This model is only defined for 128Hz signals. For any other sampling rate
    you'll need to scale the length of the kernels at layer conv2 and conv3
    appropriately (double the length for 256Hz, half the length for 64Hz, etc.)
    
    This also implements a slight variant of the original EEGNet article, where
    we use striding instead of maxpooling. The performance of the network 
    is about the same, although execution time is a bit faster. 
    
    @params 
    nb_classes: total number of final categories
    Chans: number of EEG channels
    Samples: number of EEG sample points per trial
    regRate: regularization rate for L1 and L2 regularizations
    dropoutRate: dropout fraction
    kernels: the 2nd and 3rd layer kernel dimensions (default is the
    [2, 32] x [8, 4]).
    
    """

    # start the model
    input_main   = Input((1, Chans, Samples), name = 'input')
    layer1       = Conv2D(16, (1, Chans), input_shape=(1, Chans, Samples),
                                 kernel_regularizer = l1_l2(l1=regRate, l2=regRate),
                                 name = 'conv_1')(input_main)
    layer1       = BatchNormalization(axis=2, name = 'bn_1')(layer1)
    layer1       = ELU(name = 'elu_1')(layer1)
    layer1       = Dropout(dropoutRate, name = 'drop_1')(layer1)
    
    permute_dims = 2, 1, 3
    permute1     = Permute(permute_dims, name = 'permute_1')(layer1)
    
    layer2       = Conv2D(4, kernels[0], padding = 'same', 
                            kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                            strides = strides, name = 'conv_2')(permute1)
    layer2       = BatchNormalization(axis=2, name = 'bn_2')(layer2)
    layer2       = ELU(name = 'elu_2')(layer2)
    layer2       = Dropout(dropoutRate, name = 'drop_2')(layer2)
    
    layer3       = Conv2D(4, kernels[1], padding = 'same',
                            kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                            strides = strides, name = 'conv_3')(layer2)
    layer3       = BatchNormalization(axis=2, name = 'bn_3')(layer3)
    layer3       = ELU(name = 'elu_3')(layer3)
    layer3       = Dropout(dropoutRate, name = 'drop_3')(layer3)
    
    flatten      = Flatten(name = 'flatten')(layer3)
    
    dense        = Dense(nb_classes, name = 'dense')(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    # sigmoid = Activation('sigmoid', name='sigmoid')(dense)
    
    return Model(inputs=input_main, outputs=softmax)



def EEGNet2(nb_classes, Chans = 64, Samples = 128, regRate = 0.001,
           dropoutRate = 0.25, kernLength = 64, numFilters = 8):
    """ EEGNet variant that does band-pass filtering first, implemented here
    as a temporal convolution, prior to learning spatial filters. Here, we 
    use a Depthwise Convolution to learn the spatial filters as opposed to 
    regular Convolution as depthwise allows us to learn a spatial filter per 
    temporal filter, without it being fully-connected to all feature maps 
    from the previous layer. This helps primarily to reduce the number of 
    parameters to fit... it also more closely represents standard BCI 
    algorithms such as filter-bank CSP.
    
    """

    input1   = Input(shape = (1, Chans, Samples))

    ##################################################################
    layer1       = Conv2D(numFilters, (1, kernLength), padding = 'same',
                          kernel_regularizer = l1_l2(l1=0.0, l2=0.0),
                          input_shape = (1, Chans, Samples),
                          use_bias = False)(input1)
    layer1       = BatchNormalization(axis = 1)(layer1)
    layer1       = DepthwiseConv2D((Chans, 1),
                              depthwise_regularizer = l1_l2(l1=regRate, l2=regRate),
                              use_bias = False)(layer1)
    layer1       = BatchNormalization(axis = 1)(layer1)
    layer1       = Activation('elu')(layer1)
    layer1       = SpatialDropout2D(dropoutRate)(layer1)
    
    layer2       = SeparableConv2D(numFilters, (1, 8), 
                              depthwise_regularizer=l1_l2(l1=0.0, l2=regRate),
                              use_bias = False, padding = 'same')(layer1)
    layer2       = BatchNormalization(axis=1)(layer2)
    layer2       = Activation('elu', name = 'elu_2')(layer2)
    layer2       = AveragePooling2D((1, 4))(layer2)
    layer2       = SpatialDropout2D(dropoutRate, name = 'drop_2')(layer2)
    
    layer3       = SeparableConv2D(numFilters*2, (1, 8), depth_multiplier = 2,
                              depthwise_regularizer=l1_l2(l1=0.0, l2=regRate), 
                              use_bias = False, padding = 'same')(layer2)
    layer3       = BatchNormalization(axis=1)(layer3)
    layer3       = Activation('elu', name = 'elu_3')(layer3)
    layer3       = AveragePooling2D((1, 4))(layer3)
    layer3       = SpatialDropout2D(dropoutRate, name = 'drop_3')(layer3)
    
    
    flatten      = Flatten(name = 'flatten')(layer3)
    
    dense        = Dense(nb_classes, name = 'dense')(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)



def SimpleRNN(nb_classes, Chans = 64, Samples = 128, 
           regRate = 0.01, dropoutRate = 0.25, rnn_units = 3,
           kernels = [(1, 8)]):
    """ 
    A model that just does a spatial filter followed by a temporal filter, then
    passes the extracted features off to an RNN. 
    """

    # start the model
    input_main   = Input((1, Chans, Samples), name = 'input')
    layer1       = Conv2D(4, (Chans, 1), input_shape=(1, Chans, Samples),
                                 kernel_regularizer=l1_l2(l1=regRate, l2=regRate),
                                 name = 'conv_1')(input_main)
    layer1       = BatchNormalization(axis=1, name = 'bn_1')(layer1)
    layer1       = ELU(name = 'elu_1')(layer1)
    layer1       = Dropout(dropoutRate, name = 'drop_1')(layer1)
    
    layer2        = Conv2D(8, kernels[0], padding = 'same', strides = (1, 8),
                                 kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                                 name = 'conv_2')(layer1)
    layer2        = BatchNormalization(axis=1, name = 'bn_2')(layer2)
    layer2        = ELU(name = 'elu_2')(layer2)
    layer2        = Dropout(dropoutRate, name = 'drop_2')(layer2)
    
    layer3        = Conv2D(8, kernels[0], padding = 'same', strides = (1, 8),
                                 kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                                 name = 'conv_3')(layer2)
    layer3        = BatchNormalization(axis=1, name = 'bn_3')(layer3)
    layer3        = ELU(name = 'elu_3')(layer3)
    layer3        = Dropout(dropoutRate, name = 'drop_3')(layer3)
    
    # RNNs in Keras need to be (batchsize, timesteps, features), so we need to do
    # some reshaping and permuting
    permute_dims2  = 3, 1, 2
    permute4       = Permute(permute_dims2, name = 'permute_1')(layer3)
    reshape_dims   = Samples/64, 8
    reshape4       = Reshape(reshape_dims, name = 'reshape_1')(permute4)
    
    # LSTM layer, then pass of to dense for classification
    rnn1       = LSTM(rnn_units, dropout = 0.5, recurrent_dropout = 0.0,
                      return_sequences = True, name = 'rnn_1')(reshape4)
    rnn2       = LSTM(rnn_units, dropout = 0.0, recurrent_dropout = 0.0,
                      return_sequences = False, name = 'rnn_2')(rnn1)
    dense1     = Dense(nb_classes, name = 'dense')(rnn2)
    softmax    = Activation('softmax', name = 'softmax')(dense1)
    
    return Model(inputs=input_main, outputs=softmax)  


def EEGRNN(nb_classes, Chans = 64, Samples = 128, 
           regRate = 0.01, dropoutRate = 0.25, rnn_units = 3,
           kernels = [(1, 32)]):
    """ 
    Just spatially filter the EEG then pass off to RNN.    
    
    """

    # start the model
    input_main   = Input((1, Chans, Samples))
    layer1       = Conv2D(4, (Chans, 1), input_shape=(1, Chans, Samples),
                                 kernel_regularizer=l1_l2(l1=regRate, l2=regRate))(input_main)
    layer1       = BatchNormalization(axis=1)(layer1)
    layer1       = ELU()(layer1)
    layer1       = Dropout(dropoutRate)(layer1)
    
    # RNNs in Keras need to be (batchsize, timesteps, features), so we need to do
    # some reshaping and permuting
    permute_dims2  = 3, 1, 2
    permute4       = Permute(permute_dims2)(layer1)
    reshape_dims   = Samples, 4
    reshape4       = Reshape(reshape_dims)(permute4)
    
    # LSTM layer, then pass of to dense for classification
    rnn1       = LSTM(rnn_units, dropout = 0.0, recurrent_dropout = 0.25,
                      return_sequences = True)(reshape4)
    rnn2       = LSTM(rnn_units, dropout = 0.0, recurrent_dropout = 0.25,
                      return_sequences = False)(rnn1)
    dense1     = Dense(nb_classes)(rnn2)
    softmax    = Activation('softmax')(dense1)
    
    return Model(inputs=input_main, outputs=softmax)  
    
def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):
    """ Keras implementation of the Dense Convolutional Network as described in
    Schirrmeister et. al. (2017), arXiv 1703.0505
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 10 for EEG 
    signals sampled at 250Hz. This explains why we're using length 5 
    convolutions for 128Hz sampled data (approximately half). We keep the 
    maxpool at (1, 3) with (1, 3) strides. 
    """

    # start the model
    input_main   = Input((1, Chans, Samples))
    block1       = Conv2D(25, (1, 5), 
                                 input_shape=(1, Chans, Samples))(input_main)
    block1       = Conv2D(25, (Chans, 1))(block1)
    block1       = BatchNormalization(axis=1)(block1)
    block1       = ELU()(block1)
    block1       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, (1, 5))(block1)
    block2       = BatchNormalization(axis=1)(block2)
    block2       = ELU()(block2)
    block2       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, (1, 5))(block2)
    block3       = BatchNormalization(axis=1)(block3)
    block3       = ELU()(block3)
    block3       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, (1, 5))(block3)
    block4       = BatchNormalization(axis=1)(block4)
    block4       = ELU()(block4)
    block4       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes)(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)

    
def ShallowConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), arXiv 1703.0505
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is 
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.
    
                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25
    """

    # start the model
    input_main   = Input((1, Chans, Samples))
    block1       = Conv2D(40, (1, 13), 
                                 input_shape=(1, Chans, Samples))(input_main)
    block1       = Conv2D(40, (Chans, 1))(block1)
    block1       = BatchNormalization(axis=1)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes)(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)



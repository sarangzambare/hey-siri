

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from audio_data import graph_spectrogram
import numpy as np
x = graph_spectrogram("train.wav")

x.shape

Y = np.load('train_dir/Y.npy')
Y_test = np.load('train_dir/Y_test.npy')

train_dir = "train_dir"

Y = Y.reshape(1000,1375,4)

Y = Y[:200,:,:]
Y.shape
Y_test = Y_test.reshape(100,1375,4)

def load_training_data(train_dir,num_train=1000,num_test=100):
    X = np.zeros((num_train,101,1998))
    X_test = np.zeros((num_test,101,1998))

    for i in range(num_train):
        X[i,:,:] = graph_spectrogram(train_dir + "/train" + str(i) + ".wav")

    for i in range(num_test):
        X_test[i,:,:] = graph_spectrogram(train_dir + "/traintest" + str(i) + ".wav")


    return X, X_test


X, X_test = load_training_data(train_dir,200,50)

X = X.reshape(200,1998,101)

Tx = 1998 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
Ty = 1375 # The number of time steps in the output of our model

def model(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape = input_shape)

    ### START CODE HERE ###

    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(196, kernel_size=624, strides=1)(X_input)                                 # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X)                                 # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization

    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X)                                 # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(4, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    ### END CODE HERE ###

    model = Model(inputs = X_input, outputs = X)

    return model




model = model(input_shape = (Tx, n_freq))
model.summary()

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

model.fit(X, Y, batch_size = 5, epochs=10)

loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)


def detect_triggerword(filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions


# chime_file = "audio_examples/chime.wav"
# def chime_on_activate(filename, predictions, threshold):
#     audio_clip = AudioSegment.from_wav(filename)
#     chime = AudioSegment.from_wav(chime_file)
#     Ty = predictions.shape[1]
#     # Step 1: Initialize the number of consecutive output steps to 0
#     consecutive_timesteps = 0
#     # Step 2: Loop over the output steps in the y
#     for i in range(Ty):
#         # Step 3: Increment consecutive output steps
#         consecutive_timesteps += 1
#         # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
#         if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
#             # Step 5: Superpose audio and background using pydub
#             audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
#             # Step 6: Reset consecutive output steps to 0
#             consecutive_timesteps = 0
#
#     audio_clip.export("chime_output.wav", format='wav')
#
#
#
# IPython.display.Audio("./raw_data/dev/1.wav")
#
#
#
# IPython.display.Audio("./raw_data/dev/2.wav")
#
# filename = "./raw_data/dev/1.wav"
# prediction = detect_triggerword(filename)
# chime_on_activate(filename, prediction, 0.5)
# IPython.display.Audio("./chime_output.wav")
#
#
#
# filename  = "./raw_data/dev/2.wav"
# prediction = detect_triggerword(filename)
# chime_on_activate(filename, prediction, 0.5)
# IPython.display.Audio("./chime_output.wav")
#
#
# def preprocess_audio(filename):
#     # Trim or pad audio segment to 10000ms
#     padding = AudioSegment.silent(duration=10000)
#     segment = AudioSegment.from_wav(filename)[:10000]
#     segment = padding.overlay(segment)
#     # Set frame rate to 44100
#     segment = segment.set_frame_rate(44100)
#     # Export as wav
#     segment.export(filename, format='wav')
#
#
#
# your_filename = "audio_examples/my_audio.wav"
#
#
# preprocess_audio(your_filename)
# IPython.display.Audio(your_filename)
#
#
#
# chime_threshold = 0.5

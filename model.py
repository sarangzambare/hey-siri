

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam


Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram


# Note that even with 10 seconds being our default training example length, 10 seconds of time can be discretized to different numbers of value. You've seen 441000 (raw audio) and 5511 (spectrogram). In the former case, each step represents $10/441000 \approx 0.000023$ seconds. In the second case, each step represents $10/5511 \approx 0.0018$ seconds.
#
# For the 10sec of audio, the key values you will see in this assignment are:
#
# - $441000$ (raw audio)
# - $5511 = T_x$ (spectrogram output, and dimension of input to the neural network).
# - $10000$ (used by the `pydub` module to synthesize audio)
# - $1375 = T_y$ (the number of steps in the output of the GRU you'll build).
#
# Note that each of these representations correspond to exactly 10 seconds of time. It's just that they are discretizing them to different degrees. All of these are hyperparameters and can be changed (except the 441000, which is a function of the microphone). We have chosen values that are within the standard ranges uses for speech systems.
#
# Consider the $T_y = 1375$ number above. This means that for the output of the model, we discretize the 10s into 1375 time-intervals (each one of length $10/1375 \approx 0.0072$s) and try to predict for each of these intervals whether someone recently finished saying "activate."
#
# Consider also the 10000 number above. This corresponds to discretizing the 10sec clip into 10/10000 = 0.001 second itervals. 0.001 seconds is also called 1 millisecond, or 1ms. So when we say we are discretizing according to 1ms intervals, it means we are using 10,000 steps.
#

# In[16]:

Ty = 1375 # The number of time steps in the output of our model
# ## 2.1 - Build the model
#
# Here is the architecture we will use. Take some time to look over the model and see if it makes sense.
#
# <img src="images/model.png" style="width:600px;height:600px;">
# <center> **Figure 3** </center>
#
# One key step of this model is the 1D convolutional step (near the bottom of Figure 3). It inputs the 5511 step spectrogram, and outputs a 1375 step output, which is then further processed by multiple layers to get the final $T_y = 1375$ step output. This layer plays a role similar to the 2D convolutions you saw in Course 4, of extracting low-level features and then possibly generating an output of a smaller dimension.
#
# Computationally, the 1-D conv layer also helps speed up the model because now the GRU  has to process only 1375 timesteps rather than 5511 timesteps. The two GRU layers read the sequence of inputs from left to right, then ultimately uses a dense+sigmoid layer to make a prediction for $y^{\langle t \rangle}$. Because $y$ is binary valued (0 or 1), we use a sigmoid output at the last layer to estimate the chance of the output being 1, corresponding to the user having just said "activate."
#
# Note that we use a uni-directional RNN rather than a bi-directional RNN. This is really important for trigger word detection, since we want to be able to detect the trigger word almost immediately after it is said. If we used a bi-directional RNN, we would have to wait for the whole 10sec of audio to be recorded before we could tell if "activate" was said in the first second of the audio clip.
#

# Implementing the model can be done in four steps:
#
# **Step 1**: CONV layer. Use `Conv1D()` to implement this, with 196 filters,
# a filter size of 15 (`kernel_size=15`), and stride of 4. [[See documentation.](https://keras.io/layers/convolutional/#conv1d)]
#
# **Step 2**: First GRU layer. To generate the GRU layer, use:
# ```
# X = GRU(units = 128, return_sequences = True)(X)
# ```
# Setting `return_sequences=True` ensures that all the GRU's hidden states are fed to the next layer. Remember to follow this with Dropout and BatchNorm layers.
#
# **Step 3**: Second GRU layer. This is similar to the previous GRU layer (remember to use `return_sequences=True`), but has an extra dropout layer.
#
# **Step 4**: Create a time-distributed dense layer as follows:
# ```
# X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)
# ```
# This creates a dense layer followed by a sigmoid, so that the parameters used for the dense layer are the same for every time step. [[See documentation](https://keras.io/layers/wrappers/).]
#
# **Exercise**: Implement `model()`, the architecture is presented in Figure 3.

# In[42]:

# GRADED FUNCTION: model

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
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)                                 # CONV1D
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
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    ### END CODE HERE ###

    model = Model(inputs = X_input, outputs = X)

    return model


# In[43]:

model = model(input_shape = (Tx, n_freq))


# Let's print the model summary to keep track of the shapes.

# In[44]:

model.summary()


# **Expected Output**:
#
# <table>
#     <tr>
#         <td>
#             **Total params**
#         </td>
#         <td>
#            522,561
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **Trainable params**
#         </td>
#         <td>
#            521,657
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **Non-trainable params**
#         </td>
#         <td>
#            904
#         </td>
#     </tr>
# </table>

# The output of the network is of shape (None, 1375, 1) while the input is (None, 5511, 101). The Conv1D has reduced the number of steps from 5511 at spectrogram to 1375.

# ## 2.2 - Fit the model

# Trigger word detection takes a long time to train. To save time, we've already trained a model for about 3 hours on a GPU using the architecture you built above, and a large training set of about 4000 examples. Let's load the model.

# In[45]:

model = load_model('./models/tr_model.h5')


# You can train the model further, using the Adam optimizer and binary cross entropy loss, as follows. This will run quickly because we are training just for one epoch and with a small training set of 26 examples.

# In[46]:

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


# In[47]:

model.fit(X, Y, batch_size = 5, epochs=1)


# ## 2.3 - Test the model
#
# Finally, let's see how your model performs on the dev set.

# In[48]:

loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)


# This looks pretty good! However, accuracy isn't a great metric for this task, since the labels are heavily skewed to 0's, so a neural network that just outputs 0's would get slightly over 90% accuracy. We could define more useful metrics such as F1 score or Precision/Recall. But let's not bother with that here, and instead just empirically see how the model does.

# # 3 - Making Predictions
#
# Now that you have built a working model for trigger word detection, let's use it to make predictions. This code snippet runs audio (saved in a wav file) through the network.
#
# <!--
# can use your model to make predictions on new audio clips.
#
# You will first need to compute the predictions for an input audio clip.
#
# **Exercise**: Implement predict_activates(). You will need to do the following:
#
# 1. Compute the spectrogram for the audio file
# 2. Use `np.swap` and `np.expand_dims` to reshape your input to size (1, Tx, n_freqs)
# 5. Use forward propagation on your model to compute the prediction at each output step
# !-->

# In[49]:

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


# Once you've estimated the probability of having detected the word "activate" at each output step, you can trigger a "chiming" sound to play when the probability is above a certain threshold. Further, $y^{\langle t \rangle}$ might be near 1 for many values in a row after "activate" is said, yet we want to chime only once. So we will insert a chime sound at most once every 75 output steps. This will help prevent us from inserting two chimes for a single instance of "activate". (This plays a role similar to non-max suppression from computer vision.)
#
# <!--
# **Exercise**: Implement chime_on_activate(). You will need to do the following:
#
# 1. Loop over the predicted probabilities at each output step
# 2. When the prediction is larger than the threshold and more than 75 consecutive time steps have passed, insert a "chime" sound onto the original audio clip
#
# Use this code to convert from the 1,375 step discretization to the 10,000 step discretization and insert a "chime" using pydub:
#
# ` audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio.duration_seconds)*1000)
# `
# !-->

# In[50]:

chime_file = "audio_examples/chime.wav"
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0

    audio_clip.export("chime_output.wav", format='wav')


# ## 3.3 - Test on dev examples

# Let's explore how our model performs on two unseen audio clips from the development set. Lets first listen to the two dev set clips.

# In[51]:

IPython.display.Audio("./raw_data/dev/1.wav")


# In[52]:

IPython.display.Audio("./raw_data/dev/2.wav")


# Now lets run the model on these audio clips and see if it adds a chime after "activate"!

# In[53]:

filename = "./raw_data/dev/1.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)
IPython.display.Audio("./chime_output.wav")


# In[54]:

filename  = "./raw_data/dev/2.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)
IPython.display.Audio("./chime_output.wav")


# # Congratulations
#
# You've come to the end of this assignment!
#
# Here's what you should remember:
# - Data synthesis is an effective way to create a large training set for speech problems, specifically trigger word detection.
# - Using a spectrogram and optionally a 1D conv layer is a common pre-processing step prior to passing audio data to an RNN, GRU or LSTM.
# - An end-to-end deep learning approach can be used to built a very effective trigger word detection system.
#
# *Congratulations* on finishing the fimal assignment!
#
# Thank you for sticking with us through the end and for all the hard work you've put into learning deep learning. We hope you have enjoyed the course!
#

# # 4 - Try your own example! (OPTIONAL/UNGRADED)
#
# In this optional and ungraded portion of this notebook, you can try your model on your own audio clips!
#
# Record a 10 second audio clip of you saying the word "activate" and other random words, and upload it to the Coursera hub as `myaudio.wav`. Be sure to upload the audio as a wav file. If your audio is recorded in a different format (such as mp3) there is free software that you can find online for converting it to wav. If your audio recording is not 10 seconds, the code below will either trim or pad it as needed to make it 10 seconds.
#

# In[ ]:

# Preprocess the audio to the correct format
def preprocess_audio(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')


# Once you've uploaded your audio file to Coursera, put the path to your file in the variable below.

# In[ ]:

your_filename = "audio_examples/my_audio.wav"


# In[ ]:

preprocess_audio(your_filename)
IPython.display.Audio(your_filename) # listen to the audio you uploaded


# Finally, use the model to predict when you say activate in the 10 second audio clip, and trigger a chime. If beeps are not being added appropriately, try to adjust the chime_threshold.

# In[ ]:

chime_threshold = 0.5

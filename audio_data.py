
# coding: utf-8

# ## Trigger Word Detection
#
# Welcome to the final programming assignment of this specialization!
#
# In this week's videos, you learned about applying deep learning to speech recognition. In this assignment, you will construct a speech dataset and implement an algorithm for trigger word detection (sometimes also called keyword detection, or wakeword detection). Trigger word detection is the technology that allows devices like Amazon Alexa, Google Home, Apple Siri, and Baidu DuerOS to wake up upon hearing a certain word.
#
# For this exercise, our trigger word will be "Activate." Every time it hears you say "activate," it will make a "chiming" sound. By the end of this assignment, you will be able to record a clip of yourself talking, and have the algorithm trigger a chime when it detects you saying "activate."
#
# After completing this assignment, perhaps you can also extend it to run on your laptop so that every time you say "activate" it starts up your favorite app, or turns on a network connected lamp in your house, or triggers some other event?
#
# <img src="images/sound.png" style="width:1000px;height:150px;">
#
# In this assignment you will learn to:
# - Structure a speech recognition project
# - Synthesize and process audio recordings to create train/dev datasets
# - Train a trigger word detection model and make predictions
#
# Lets get started! Run the following cell to load the package you are going to use.
#

# In[1]:

import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
import matplotlib.pyplot as plt
from scipy.io import wavfile

get_ipython().magic('matplotlib inline')

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def load_raw_audio(train_dir):
    ones = []
    twos = []
    threes = []
    backgrounds = []
    negatives = []

    for filename in os.listdir(train_dir+"/one"):
        if filename.endswith("wav"):
            one = AudioSegment.from_wav(train_dir+"/one/"+filename)
            ones.append(one)
    for filename in os.listdir(train_dir+"/two"):
        if filename.endswith("wav"):
            two = AudioSegment.from_wav(train_dir+"/two/"+filename)
            twos.append(two)
    for filename in os.listdir(train_dir+"/three"):
        if filename.endswith("wav"):
            three = AudioSegment.from_wav(train_dir+"/three/"+filename)
            threes.append(three)

    for filename in os.listdir(train_dir+"/background"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(train_dir+"/background/"+filename)
            backgrounds.append(background[:10000])
    for filename in os.listdir(train_dir+"/negative"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav(train_dir+"/negative/"+filename)
            negatives.append(negative)
    return ones, twos, threes, negatives, backgrounds



ones, twos, threes, negatives, backgrounds = load_raw_audio('train_dir')



def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """

    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """

    segment_start, segment_end = segment_time

    ### START CODE HERE ### (≈ 4 line)
    # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
    overlap = False

    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
    for previous_start, previous_end in previous_segments:
        if ((segment_end >= previous_start) and (segment_start <= previous_end)):
            overlap = True
    ### END CODE HERE ###

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)

    ### START CODE HERE ###
    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert
    # the new audio clip. (≈ 1 line)
    segment_time = get_random_time_segment(len(audio_clip))

    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep
    # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
    while is_overlapping(segment_time,previous_segments):
        segment_time = get_random_time_segment(len(audio_clip))

    # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(segment_time)
    ### END CODE HERE ###

    # Step 4: Superpose audio segment and background
    new_background = background.overlay(audio_clip, position = segment_time[0])

    return new_background, segment_time


def insert_ones(y, segment_end_ms, label):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.


    Arguments:
    y -- numpy array of shape (4, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    label -- class of sound, i.e one/two/three/negative

    Returns:
    y -- updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    # Add 1 to the correct index in the background label (y)
    ### START CODE HERE ### (≈ 3 lines)
    for i in range(segment_end_y+1, segment_end_y + 51):
        if i < Ty:
            if(label == 'one'):
                y[0,i] = 1
            elif(label == 'two'):
                y[1,i] = 1
            elif(label == 'three'):
                y[2,i] = 1
            elif(label == 'negative'):
                y[3,i] = 1
#    y[0][segment_end_y:segment_end_y+50] = 1
    ### END CODE HERE ###

    return y

def create_single_training_example(background, ones, twos, threes, negatives, suffix=None):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    ones -- a list of audio segments of the word "one"
    twos -- a list of audio segments of the word "two"
    threes -- a list of audio segments of the word "three"
    negatives -- a list of audio segments of random words that are not "one/two/three"
    suffix -- a string to add at the end of the filename

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    # Set the random seed


    # Make background quieter
    background = background - 20

    ### START CODE HERE ###
    # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
    y = np.zeros((4, Ty))

    # Step 2: Initialize segment times as empty list (≈ 1 line)
    previous_segments = []
    ### END CODE HERE ###

    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_ones = np.random.randint(0, 2)
    number_of_twos = np.random.randint(0, 2)
    number_of_threes = np.random.randint(0, 2)
    random_indices_one = np.random.randint(2350, size=number_of_ones) #min(len(ones),len(twos),len(threes)) = 2350
    random_indices_two = np.random.randint(2350, size=number_of_twos)
    random_indices_three = np.random.randint(2350, size=number_of_threes)
    random_ones = [ones[i] for i in random_indices_one]
    random_twos = [twos[i] for i in random_indices_two]
    random_threes = [threes[i] for i in random_indices_three]

    ### START CODE HERE ### (≈ 3 lines)
    # Step 3: Loop over randomly selected "activate" clips and insert in background
    for random_one in random_ones:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_one, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms=segment_end,label='one')
    for random_two in random_twos:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_two, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms=segment_end,label='two')
    for random_three in random_threes:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_three, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms=segment_end,label='three')
    ### END CODE HERE ###

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 2)
    random_indices_negative = np.random.randint(1561, size=number_of_negatives) #len(negatives) = 1561
    random_negatives = [negatives[i] for i in random_indices_negative]

    ### START CODE HERE ### (≈ 2 lines)
    # Step 4: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_negative, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end_ms=segment_end,label='negative')
    ### END CODE HERE ###

    # Standardize the volume of the audio clip
    background = match_target_amplitude(background, -20.0)

    # Export new training example
    file_handle = background.export("train_dir/train" + suffix + ".wav", format="wav")
    #print("File (train.wav) was saved in your directory.")

    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    #x = graph_spectrogram("train.wav")

    return y

Ty = 1375 # The number of time steps in the output of our model


def generate_training_set(backgrounds,ones,twos,threes,negatives,num_train=1000,num_test=100):

    Y = np.zeros((num_train,4,Ty))
    Y_test = np.zeros((num_test,4,Ty))

    for i in range(num_train):
        j = np.random.randint(0,6) #number of background tracks = 3
        y = create_single_training_example(backgrounds[j], ones, twos, threes, negatives, str(i))
        Y[i] = y

    for i in range(num_test):
        j=np.random.randint(0,6)
        y = create_single_training_example(backgrounds[j], ones, twos, threes, negatives, "test"+str(i))
        Y_test[i] = y

    np.save("train_dir/Y.npy",Y)
    np.save("train_dir/Y_test.npy",Y_test)

    return 0


generate_training_set(backgrounds,ones,twos,threes,negatives)


######################################################################




Tx = 5511 # The number of time steps input to the model from the spectrogra

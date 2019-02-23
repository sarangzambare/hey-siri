# audio-trigger
This repository is for trigger-word detection/ digit recognition in speech using recurrent neural networks


Trigger word/ wake word detection is one of the most common applications of sequence models. The goal is to detect instances when a certain word is uttered, and more often than not, recognise the said word from amongst a small pool of "wake words"

In this repository, I demonstrate a GRU network, in combination with 1D Convolution, to detect three words:

**One, Two and Three**

The entire process has the following steps :

1. Generate training dataset with labels.
2. Convert audio files into **spectrograms**
3. Train a combination of 1D Convolution and Recurrent layers to detect utterances.


## Generating training dataset:

All sounds in this project are taken from the [Google speech commands dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html).

The dataset has 1 sec audio samples of a host of different words, but we will only be using audio clips of the words "One", "Two" and "Three"


<intro>





<spectrogram>

<backward looking vs forward looking/ why uni-directional LSTMs>

<convolution for >

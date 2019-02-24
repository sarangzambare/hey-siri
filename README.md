# audio-trigger

### Under Construction


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

The dataset has 1 sec audio samples of a host of different words, but I will only be audio clips of the words "One", "Two" and "Three" and a combination of some other random words like **"Bird", "Bed", "Happy", "House" which would be labelled as "negative"**

The dataset also has various background noises, like **"Someone doing the dishes", "Cat Meowing", "Running tap"**

All these sounds will be combined in various proportions to generate **10 sec** audio clips of words "One, Two and Three"
superimposed over the background noises, along with the "negatives"


Concurrently, an array of labels will be generated, with each label being a one-hot encoded vector, like so:

```
[1.0, 0.0, 0.0, 0.0] => "One"
[0.0, 1.0, 0.0, 0.0] => "Two"
[0.0, 0.0, 1.0, 0.0] => "Three"
[0.0, 0.0, 0.0, 1.0] => "negative"
```

The 10 sec audio clips generated will be in ".wav" format. The WAV is a [**bitstream format**](https://en.wikipedia.org/wiki/Bitstream_format). Loosely speaking, it contains time indexed chunks of silent and noisy patches to make up the audio file. Although the WAV file can be fed into a neural network as is, the network learns much better representations if spectrograms are used.

## Spectrograms:

Any time varying signal can be broken down into its frequency components, by **fourier decomposition**, which
is the idea that any signal can be expressed as the sum of sinusoids with different frequencies.

![alt text](https://raw.githubusercontent.com/sarangzambare/audio-trigger/master/png/fourier.png)


For any sound, spectral decomposition gives us the amplitudes of each of its frequency components. For example, a section of a violin being played can be broken down like so:

![alt text](https://raw.githubusercontent.com/sarangzambare/audio-trigger/master/png/violin.png)

If such a decomposition is done at every time step of the audio signal and plotted, what results is a spectrogram.

For example, a harp clip taken from [Chrome Music Lab: Spectrograms](https://musiclab.chromeexperiments.com/Spectrogram/), looks like so:

![alt text](https://raw.githubusercontent.com/sarangzambare/audio-trigger/master/png/harp.png)

where the y-axis is the frequency and x-axis is time, and the colour represents the amplitude.


In our case, each audio clip is 10 sec long. In this program, I used **pydub** to input audio files, with a sampling rate of **200Hz**, which means there will be 1 sample for every 5ms, making it about **2000 samples**. If we plot the spectrogram of a sample training file ("train.wav" file in root directory), it looks like :

![alt text](https://raw.githubusercontent.com/sarangzambare/audio-trigger/master/png/train_2.png)


## Labelling :

To train the model, we need to label the training data according to the utterances of the three words in the audio clips. Also, we want to detect time instances immediately after the words are uttered. To achieve this, the labelling process has the following steps :

1. Get time segments where the target words (including negatives) are placed.
2. Insert label vectors which have a length of 4 right after the segment ends
3. Repeat the label for a fixed amount of time steps (in this program: 50). This is done to give the network some time to process the input before giving out the labels. A label which is only a single time step wide might not allow enough time for this.

For example, if you ignore the repeating of labels for illustration purposes, the training sample audio clip will be labelled as follows :

![alt text](https://raw.githubusercontent.com/sarangzambare/audio-trigger/master/png/label_coding.png)


## Model

For the purpose of detecting the utterances of the three words in the audio signal, we have

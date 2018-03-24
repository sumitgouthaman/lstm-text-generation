
# Text content generation using LSTMs

LSTM (Long Short Term Memory networks) are one of the building blocks of RNNs (Recurrent Neural Networks). Unlike traditional neural networks that have fixed input sizes, LSTMs can be fed a varying lenght sequence of inputs. This makes them perfect for dealing with many text related applications.

One application where LSTMs are ["unreasonably effective"](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) is text generation. Basically, we start with a corpus of text (like a set of blog posts, letters, tweets, or even source code) and train a neural network over it. Once trained, we can feed some starter text to the network, and have it predict the next character. Repeat this process N times with a sliding window, and you can generate N characters of text.

This notebook is an example of applying a very simple LSTM (built with Keras & Tensorflow) over a corpus of transcribed podcast content.

### Topics covered in this notebook
1. Pre-processing text data input for LSTM.  
1. Using trained model to generate text while varying temperature parameter.  
1. Keras sequence generators.  
1. Keras callbacks.  

## Pre-requisites

### Software
This tutorial assumes that you have a workstation with Tensorflow, Keras and some other common data science utilities installed. Since LSTMs are quite compute-intensive to train, running this example without a GPU is mostly impractical.

I like to install these tools in a separate python [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/). This keeps things neat and clean.

I configure my virtual environment as follows:

```shell
mlpy3venv() {
  if [ -d "~/Venv" ] ; then
    mkdir ~/Venv
  fi
  virtualenv --system-site-packages -p python3 ~/Venv/mlpy3venv
  source ~/Venv/mlpy3venv/bin/activate
  easy_install -U pip
  pip3 --no-cache-dir install \
    Pillow \
    h5py \
    ipykernel \
    jupyter \
    matplotlib \
    numpy \
    pandas \
    scipy \
    sklearn \
    keras \
    tqdm \
    && \
    python -m ipykernel.kernelspec
  pip3 install --upgrade tensorflow-gpu
  deactivate

  cat >> ~/.bashrc <<EOF
mlpy3 () {
  cd ~/Project
  source ~/Venv/mlpy3venv/bin/activate
}
EOF
}

mlpy3venv
```

This creates a python 3 virtual env in `~/Venv/mlpy3venv`, installs necessary libraries in it, and creates a bash alias **mlpy3** to quickly activate and use it.

For more details, you can refer my complete workstation setup script [here](https://github.com/sumitgouthaman/workstation-setup/blob/master/workstation_setup.sh).

Run the cell below to import necessary libraries to run rest of the notebook.

**Note about python 2/3:** I try to be both python 2 and 3 compatible for rest of the code in this notebook.


```python
# Python 2 / 3 compatibility
from __future__ import print_function
from __future__ import division

from datetime import datetime
import io
import math
import os
import pickle
import random
import re
import string
import sys

from tqdm import tqdm  # To draw progress bars

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  

# Keras
from keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional                                                
from keras.models import Sequential                 
from keras.optimizers import RMSprop, Adam                
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import Sequence
from keras.utils.data_utils import get_file 
from keras import backend as K

# Scikit learn
from sklearn.model_selection import train_test_split
```

    /home/sumit/Venv/mlpy3venv/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


### Choosing a corpus

I chose to use TWiT's [Security Now](https://twit.tv/shows/security-now) podcast content as the corpus for this experiment. The reasons are pretty straight-forward:  

- Security Now has over 600 episodes spanning more than a decade of stories.  
- Steve Gibson hosts meticulously transcribed content of each episode in plain text format on [GRC](https://www.grc.com/securitynow.htm).  

I am not aware of any other Podcast that meets the same criteria. I considered using off-the-shelf speech-to-text APIs to transcribe other podcasts, but the time and money involved in doing so seemed excessive for a small experiment.

We use Keras' built-in `get_file()` function to download and cache episode content as text files locally.


```python
latest_episode_number = 646  # From: https://www.grc.com/securitynow.htm

# Attempt to download as much of Security Now transcript if possible
downloaded_episode_files = []
for episode in range(1, latest_episode_number+1):
    source_path = 'https://www.grc.com/sn/sn-%03d.txt' % episode
    dest_file_path = '%03d.txt' % episode
    try:
        path = get_file(dest_file_path, origin=source_path)
        downloaded_episode_files.append(path)
    except Exception as e:
        # Ignore: Some episodes might not exists
        print('Downloading episode %03d. Episode might not exist.' % episode)

print('Episodes downloaded: %d' % len(downloaded_episode_files))
print('First 5:\n%s' % ',\n'.join(downloaded_episode_files[:5]))
```

    Downloading data from https://www.grc.com/sn/sn-436.txt
    Downloading episode 436. Episode might not exist.
    Downloading data from https://www.grc.com/sn/sn-487.txt
    Downloading episode 487. Episode might not exist.
    Downloading data from https://www.grc.com/sn/sn-540.txt
    Downloading episode 540. Episode might not exist.
    Downloading data from https://www.grc.com/sn/sn-592.txt
    Downloading episode 592. Episode might not exist.
    Downloading data from https://www.grc.com/sn/sn-643.txt
    Downloading episode 643. Episode might not exist.
    Episodes downloaded: 641
    First 5:
    /home/sumit/.keras/datasets/001.txt,
    /home/sumit/.keras/datasets/002.txt,
    /home/sumit/.keras/datasets/003.txt,
    /home/sumit/.keras/datasets/004.txt,
    /home/sumit/.keras/datasets/005.txt


## Pre-processing

As with any machine learning problem, the first step is usually to peek at the dataset and clean it up for consumption by our algorithm.

### Non ascii characters
A preliminary examination of the files showed me that there are some non-ascii characters in it. We should probably get rid of them to keep things simpler.

We can use the `.encode('ascii', 'ignore')` construct to get rid of all the non-ascii characters. In python 2, this returns a string. However, in python 3, this returns a byte array, and we need to add a `decode('ascii')` to keep the code python 2/3 compliant.

We also convert the text to **lower-case** for simplification.

We print out a small sample from the first and last episode for a sanity check.


```python
episode_texts = []
for path in downloaded_episode_files:
    # Load episode texts.
    # Also convert to  lower case and remove non-ascii characters.
    episode_texts.append(
        io.open(path, encoding='ISO-8859-1')
        .read()
        .lower()
        .encode('ascii', 'ignore')
        .decode('ascii')
    )
    
print('Sample episode text:')
print('-' * 50)
print(episode_texts[0][:1000])  # First episode
print('-' * 50)
print(episode_texts[-1][:1000])  # Last episode
print('-' * 50)
```

    Sample episode text:
    --------------------------------------------------
    gibson research corporation	http://www.grc.com/
    
    series:		security now!
    episode:	#1
    date:		august 19, 2005
    title:		as the worm turns: the first internet worms of 2005
    speakers:	steve gibson & leo laporte
    source file:	http://media.grc.com/sn/sn-001.mp3
    file archive:	http://www.grc.com/securitynow.htm
    
    description:  how a never-disclosed windows vulnerability was quickly reverse-engineered from the patches to fix it and turned into more than 12 potent and damaging internet worms in three days.  what does this mean for the future of internet security?
    
    leo laporte:  hi, this is leo laporte, and i'd like to introduce a brand-new podcast to the twit lineup, security now! with steve gibson.  this is episode 1 for august 18, 2005.  you all know steve gibson.  he, of course, appears on twit regularly, this week in tech.  we've known him for a long time.  he's been a regular on the screensavers and call for help.  and, you know, he's well-known to computer users everywhere for his products.  he
    --------------------------------------------------
    gibson research corporation		https://www.grc.com/
    
    series:		security now!
    episode:	#646
    date:		january 16, 2018
    title:		the inspectre
    hosts:	steve gibson & leo laporte
    source:	https://media.grc.com/sn/sn-646.mp3
    archive:	https://www.grc.com/securitynow.htm
    
    description:  this week we discuss more trouble with intel's amt, what skype's use of signal really means, the uk's data protection legislation giving researchers a bit of relief, the continuing winding down of http, "progress" on the development of meltdown attacks, google successfully tackling the hardest to fix spectre concern with a return trampoline, some closing-the-loop feedback with our terrific listeners, and the evolving landscape of meltdown and spectre - including steve's just completed "inspectre" test and explanation utility.
    
    show tease:  it's time for security now!.  steve gibson is here.  this is a hot episode, i think perhaps will be one of the most listened-to security now! episodes of the year, maybe of the decad
    --------------------------------------------------


### Stripping metadata line

We can see that the file contains some metadata about the episode at the beginning. We are primarily interested in generating the podcast content, so we should strip this off. A manual inspection of the first few lines of random episodes indicates that the size of this metadata is consistent across all episodes. We typically have 12 lines of intro metadata at the start and around 4 lines of footer at the end.

We use strip_intro_lines() to remove these lines from all the episodes.

As before, we print out some processed episodes for a sanity check.


```python
def strip_intro_lines(episode_text, lines_to_strip_head, lines_to_strip_tail):
    """Return episode text after removing the first #lines_to_strip lines.
    
    Args:
        episode_text: The text of the episode.
        lines_to_strip: Number of lines to remove from the start of the episode.
        
    Returns:
        The episode text with lines removed.
    """
    lines = episode_text.split('\n')
    lines = lines[lines_to_strip_head:-lines_to_strip_tail]
    return '\n'.join(lines)

# The beginning 12 lines need to be stripped (show metadata).
# The last 4 lines need to be stripped (copyright).
episode_texts = [strip_intro_lines(e, 12, 4) for e in episode_texts]

print('Sample episode text after processing:')
print('-' * 50)
print(episode_texts[0][:1000])  # First episode
print('-' * 50)
print(episode_texts[-1][:1000])  # Last episode
print('-' * 50)
```

    Sample episode text after processing:
    --------------------------------------------------
    leo laporte:  hi, this is leo laporte, and i'd like to introduce a brand-new podcast to the twit lineup, security now! with steve gibson.  this is episode 1 for august 18, 2005.  you all know steve gibson.  he, of course, appears on twit regularly, this week in tech.  we've known him for a long time.  he's been a regular on the screensavers and call for help.  and, you know, he's well-known to computer users everywhere for his products.  he's very well known to consumers for spinrite, which was the inspiration for norton disk doctor and still runs rings around it.  it is the ultimate hard-drive diagnostic recovery and file-saving tool.  it's really a remarkable tool that everybody should have a copy of from grc.com.  but he's also been a very active consumer advocate, working really hard to help folks with their security.  he first came to my attention with the click of death, which was - that was the zip drive iomega...
    
    steve gibson:  right.
    
    leo:  ...hassle.  and was it you that kin
    --------------------------------------------------
    show tease:  it's time for security now!.  steve gibson is here.  this is a hot episode, i think perhaps will be one of the most listened-to security now! episodes of the year, maybe of the decade, i don't know, because we're going to give you, not only the latest on spectre and meltdown, but a program you can use on windows machines that will help you know whether you're safe, and what performance hit you're going to get.  security now! is next.
    
    leo laporte:  this is security now! with steve gibson, episode 646, recorded tuesday, january 16th, 2018:  the inspectre.
    
    it's time for security now!, the show where we come to you via the internet to protect you from the internet.  i don't know how that works.  but that's steve gibson.  he's figured it all out.  it's like inception.  steve is the man in charge at grc.com and our security hero.  and you're going to really be a hero this week, steve.
    
    steve gibson:  well, yes.  after last week's podcast i went back to work on sqrl that evenin
    --------------------------------------------------


### Exploring the dataset

We first check what sorts of characters are present in our dataset. We expect it to contain all the letters of the english alphabet with some common punctuations added in.


```python
unique_chars = set()
for episode_text in tqdm(episode_texts):
    chars_in_episode = set(episode_text)
    unique_chars |= chars_in_episode
    
print('Unique chars in dataset: %d' % len(unique_chars))
print(unique_chars)
```

    100%|██████████| 641/641 [00:00<00:00, 1713.04it/s]

    Unique chars in dataset: 72
    {'l', ';', '*', '3', '2', 'z', '0', 's', '_', 'h', 'u', '=', '#', 'j', '/', 'd', '^', '`', '9', '>', '?', 'a', "'", '-', '(', '8', '7', '.', '{', '[', 'b', 't', '|', '\x1e', '"', 'p', 'y', 'm', '%', '~', 'g', 'x', '$', '1', ' ', 'r', 'c', ',', '\\', '\t', 'k', 'o', 'w', 'n', '!', '@', ']', 'e', '4', '<', ')', '+', 'v', '}', 'f', ':', '\n', '&', '6', 'i', 'q', '5'}


    


This is mostly what I expected. But there is a strange '\x1e' character thrown in.

Some [preliminary investigation](http://facweb.cs.depaul.edu/sjost/it211/documents/ascii-npr.htm) tells me it's a "Record separator". Strange. I don't know what that is, but I'll leave it there anyways.

## Building the model

I've heard Keras is a great library for beginners. A quick search for "LSTM text generation keras" leads directly to this [file](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py).

The example uses Nietzsche's writings as the dataset. We can re-use significant portions of the code. But our dataset is an order of magnitude larger, which is going to create some problems as we will see.

### Input format

Lets assume we have a language with 4 unique characters A, B, C and D. Given a text of N characters, we want to predict the (N+1)st character.

We start by assigning each character from our corpus a unique integer id. When feeding input into our model, we will represent this id as a [one-hot](https://en.wikipedia.org/wiki/One-hot) encoded vector.

So assume, in out simplified case where we only have 4 characters, we assign the encodings as follows:

```
A -> 1 -> 0001
B -> 2 -> 0010
C -> 3 -> 0100
D -> 4 -> 1000
```

To proceed, we first assign each of the 72 character in our actual dataset a unique integer id. We store this in a dictionary so that it can be retrieved later. We also store a inverse dictionary mapping the integer id back to the character so that we can decode the output of our model.


```python
unique_chars = sorted(list(unique_chars))
char_indices = dict((c, i) for i, c in enumerate(unique_chars))
indices_char = dict((i, c) for i, c in enumerate(unique_chars))
```

Going back to our simplified situation with only 4 characters, assume we have a sentence in our corpus as follows:

```
ACDBACCDBDACDABA
```

To feed this into our network, we will chop the sentence into smaller examples that operate over a sliding window.

Assume, we want a window of size 5 with a step size of 2. What this means is that we will feed 5 characters into the network and attempt to predict the next character. For the next example, we will move 2 characters to the right and pick the next 5 characters.

So our original example sentence will produce the following input examples:

```
 [ACDBA](C) CDBDACDABA -> X: ACDBA, Y: C
AC [DBACC](D) BDACDABA -> X: DBACC, Y: D
ACDB [ACCDB](D) ACDABA -> X: ACCDB, Y: D
ACDBAC [CDBDA](C) DABA -> X: CDBDA, Y: C
ACDBACCD [BDACD](A) BA -> X: BDACD, Y: A
ACDBACCDBD [ACDAB](A)  -> X: ACDAB, Y: A
```

Now for each value of X, we encode each character as a one-hot encoded vector. So:

```
ACDBA -> [[0,0,0,1], [0,1,0,0], [1,0,0,0], [0,0,1,0], [0,0,0,1]]
```

So, if our dataset produces N input examples, the total size of X input matrix to our model would be:
```
N x <Window Size> x <Size of charater set>
```

This is a really small example, with a unrealistically small character set. But in reality, this way of representing the input data can lead to very high memory requirements.

Let's try to estimate the size of the input tensor X, if we used our entire dataset with a window size of 100, step size of 2 and character set size of 72.


```python
total_dataset_length = sum(len(e) for e in episode_texts)
assumed_input_window_size = 100
assumed_step_size = 2

estimated_input_examples = len(
    range(
        0,
        total_dataset_length - assumed_input_window_size,
        assumed_step_size
    )
)

print('Total dataset character lenght: %d' % total_dataset_length)
print('Assuming window size of %d and step size of %d:' % (assumed_input_window_size, assumed_step_size))
print('Estimated shape of X: %d x %d x %d' % (estimated_input_examples, assumed_input_window_size, len(unique_chars)))
print('Estimated size of X: %d' % (estimated_input_examples*assumed_input_window_size*len(unique_chars)))
```

    Total dataset character lenght: 45658091
    Assuming window size of 100 and step size of 2:
    Estimated shape of X: 22828996 x 100 x 72
    Estimated size of X: 164368771200


These are some huge numbers and we would definitely not be able to fit the entire dataset in our memory. 

### Solving the memory issue for our input tensor

A solution for this is to avoid holding the entire dataset in an in-memory tensor. Instead we want a way to generate input tensors in batches as and when needed.

We will use a Keras mechanism called [Sequence generators](https://keras.io/utils/#sequence) to achieve this. Implementing a sequence generator involves two methods:

1. A `__len__()` method that returns number of batches.  
2. A `__getitem__(batch_index)` method that returns a particular batch on demand.

We implement a EpisodesSequence class to handle this for us. An object of this class can be initialized by passing in a array of episode texts.

We will reuse this class for both the training sequence and the validation sequence.


```python
class EpisodesSequence(Sequence):

    def __init__(self, episode_texts, window_size, step_size, unique_chars, char_indices, batch_size):
        self.all_episodes_texts = '\n'.join(episode_texts)
        self.window_size = window_size
        self.step_size = step_size
        self.char_indices = char_indices
        self.batch_size = batch_size
        
        self.character_set_length = len(unique_chars)
        self.total_input_examples = int(
            math.ceil((len(self.all_episodes_texts) - self.window_size) / self.step_size)
        )
        self.total_batches = int(
            math.ceil(self.total_input_examples / self.batch_size)
        )
        
        print(
            'EpisodesSequence generator initialized. Total windows: %d, Total batches: %d' %(
                self.total_input_examples, self.total_batches
            )
        )

    def __len__(self):
        return self.total_batches

    def __getitem__(self, batch_index):
        examples_start_index = batch_index * self.batch_size
        examples_end_index = min(examples_start_index+self.batch_size, self.total_input_examples)
        current_batch_size = examples_end_index - examples_start_index
        sentences = []
        next_chars = []
        text_index = examples_start_index * self.step_size  # Start of batch
        for i in range(current_batch_size):
            sentences.append(self.all_episodes_texts[text_index: text_index + self.window_size])
            next_chars.append(self.all_episodes_texts[text_index + self.window_size])
        x = np.zeros((len(sentences), self.window_size, self.character_set_length), dtype=np.bool)
        y = np.zeros((len(sentences), self.character_set_length), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
        return x, y
```

### Dealing with failures and using checkpointing

Training a LSTM model like the one we intend to build takes a long time. It is prudent to checkpoint your model periodically.

Going one step further, we will implement a helper method that we use during our training step. This method will detect if there are model checkpoint files (.hdf5) lying around in a folder with the same name as the model. If yes, it figures out what was the highest epoch that was checkpointed, restores weights from that checkpoint and continues training from the next epoch.


```python
def initialize_from_last_checkpoint(model):
    model_dirs = [d for d in os.listdir('.') if os.path.isdir(os.path.join('.', d)) and d.startswith(model.name)]
    
    if not model_dirs:
        return 0, None
    
    # Get directory corresponding to last execution (should be one with highest timestamp)
    latest_run_dir = os.path.join('.', sorted(model_dirs)[-1])
    
    # Get list of saved model files
    checkpointed_models = [m for m in os.listdir(latest_run_dir) if m.endswith('.hdf5')]
    
    if not checkpointed_models:
        return 0, None
    
    # Sort by epoch number
    checkpointed_models = sorted(checkpointed_models, key=lambda m: int(re.search('ep-(\d+)', m).group(1)))
    
    # Choose model with highest epoch
    latest_model = checkpointed_models[-1]
    latest_model = os.path.join(latest_run_dir, latest_model)
    epoch = int(re.search('ep-(\d+)', latest_model).group(1))
    
    print('Loading weights from %s' % latest_model)
    model.load_weights(latest_model)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return epoch, latest_run_dir
```

### Training

We implement a helper method that runs training for some number of epochs on our dataset. It uses some of the helper methods we defined above to restore previously checkpointed model weights.

We also add a Keras [callback](https://keras.io/callbacks/) that checkpoints the model after each epoch.


```python
def train_model(model, generator, validation_generator, epochs, on_epoch_end,
                continue_from_checkpoint=False, lr_decay=False):
    
    initial_epoch = 0
    checkpoint_folder_name = '%s_%s' % (model.name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    if continue_from_checkpoint:
        initial_epoch, prev_folder_name = initialize_from_last_checkpoint(model)
        if prev_folder_name:
            print('Re-using existing folder %s.' % prev_folder_name)
            checkpoint_folder_name = prev_folder_name
    
    print('Checkpoint folder: %s' % checkpoint_folder_name)
    
    if not os.path.isdir(checkpoint_folder_name):
        os.makedirs(checkpoint_folder_name)
        
    lambda_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    filepath='%s/ep-{epoch:02d}-vloss-{val_loss:.3f}.hdf5' % checkpoint_folder_name
    checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss',verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callbacks_list = [lambda_callback, checkpoint_callback]
    
    if lr_decay:
        decay_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, epsilon=0.001)
        callbacks_list.append(decay_callback)
    
    model.fit_generator(
        generator=generator,
        epochs=epochs,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        shuffle=False,
        initial_epoch=initial_epoch,
    )
```

### Using the model to generate text

Next we will implement a method to generate text by using our trained model.

The input to our model takes a list of one-hot encoded character vectors. This list is our seed text. The ouput of the model is a list of probabilities for each of the characters in our corpus.

```
[[0,0,0,1], [0,1,0,0], [1,0,0,0], [0,0,1,0], [0,0,0,1]] -> model -> [0.01, 0.03, 0.85, 0.02]
```

The simplest approach would be to choose the character with the highest probability. We then shift the window one step to the right (i.e. our newly predicted character becomes the last character in the next input we feed to the model). But this often leads to text that is repetitive.

The solution is to take into account a temperature parameter that tweaks (scales) our probabilities such that there is an increased likelihood of picking a different character. I learnt this from the keras text generation example code. Later, I found some links that explain what is going on here [[1]](https://stackoverflow.com/questions/37246030/how-to-change-the-temperature-of-a-softmax-output-in-keras), [[2]](https://www.machinelearningpython.org/single-post/Text-Generation---An-Improvement).

We put this toghther in a `generate_sample_text` function that starts with a random seed text of length = window size. We encode this string in our typical input format as a vector of one-hot encoded vectors. Passing this through our model produces an output which can be interpreted as probabilities of each output character. We use our sample method to decide on a predicted character.

At the next step, we shift our window one character to the right and repeat the process all over again. We can continue this till we generate as many characters as we want.


```python
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_sample_text(model, episode_texts, window_size, unique_chars, char_indices, indices_char):
    
    text = '\n'.join(episode_texts)
    start_index = random.randint(0, len(text) - window_size - 1)

    print('-' * 50)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('-- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + window_size]
        generated += sentence
        print('-- Generating with seed:\n%s\n--' % sentence)

        for i in range(400):
            x_pred = np.zeros((1, window_size, len(unique_chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
        
        print(generated)
        print('-' * 50)
```

## Trying some models

### Simple LSTM model
This is a very simple model (very similar to the Keras sample code).

1. It has one LSTM layer with 256 hidden units.  
2. We add a dropout of 0.5 to avoid over-fitting.  
3. We add a densely connected layer with output dimension equal to number of characters in our corpus.  
4. We finally end with a softmax layer.


```python
window_size=200
step_size=4
batch_size=256

lstm_layer_size=256

character_set_length = len(unique_chars)

model_name = 'lstm_%d_1layer_wsize_%d' % (lstm_layer_size, window_size)
model = Sequential(name=model_name)
model.add(LSTM(lstm_layer_size, input_shape=(window_size, character_set_length)))
model.add(Dropout(0.5))
model.add(Dense(character_set_length))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

train_episodes, val_episodes = train_test_split(episode_texts, train_size=0.75)

generator = EpisodesSequence(
    episode_texts=train_episodes,
    window_size=window_size,
    step_size=step_size,
    unique_chars=unique_chars,
    char_indices=char_indices,
    batch_size=batch_size
)

validation_generator = EpisodesSequence(
    episode_texts=val_episodes,
    window_size=window_size,
    step_size=step_size,
    unique_chars=unique_chars,
    char_indices=char_indices,
    batch_size=batch_size
)

def on_epoch_end(epoch, logs):
    print('\nWill generate text samples using model %s, after epoch %d' %(model.name, epoch))
    generate_sample_text(
        model,
        episode_texts,
        window_size,
        unique_chars,
        char_indices,
        indices_char
    )

train_model(model, generator, validation_generator, epochs=5,
            on_epoch_end=on_epoch_end, continue_from_checkpoint=True, lr_decay=True)
```

    /home/sumit/Venv/mlpy3venv/lib/python3.5/site-packages/sklearn/model_selection/_split.py:2026: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
      FutureWarning)


    EpisodesSequence generator initialized. Total windows: 8565022, Total batches: 33458
    EpisodesSequence generator initialized. Total windows: 2849562, Total batches: 11132
    Checkpoint folder: lstm_256_1layer_wsize_200_2018-03-19_19-29-30
    Epoch 1/5
    33457/33458 [============================>.] - ETA: 0s - loss: 2.2669
    Will generate text samples using model lstm_256_1layer_wsize_200, after epoch 0
    --------------------------------------------------
    -- diversity: 0.2
    -- Generating with seed:
    ove past using an md5 hash.  ssl v3 sort of fixed it a little bit by coming up with a clever solution.  they hashed both - they produced both an md5 hash and an sha-1 hash, and they xored the result. 
    --
    ove past using an md5 hash.  ssl v3 sort of fixed it a little bit by coming up with a clever solution.  they hashed both - they produced both an md5 hash and an sha-1 hash, and they xored the result.  ind the be so the the prome the sere the prome the dere and the prome the prome the sere the pere the the the prome the prome the sere the sere the perser the sere the sere the sere and the prope the proble the dose the perser the sere and the dine the sere the sere the sere the sere the sere the sere the perers the proble the sere the sere the prost the proble the the cars the perent the proble 
    --------------------------------------------------
    -- diversity: 0.5
    -- Generating with seed:
    ove past using an md5 hash.  ssl v3 sort of fixed it a little bit by coming up with a clever solution.  they hashed both - they produced both an md5 hash and an sha-1 hash, and they xored the result. 
    --
    ove past using an md5 hash.  ssl v3 sort of fixed it a little bit by coming up with a clever solution.  they hashed both - they produced both an md5 hash and an sha-1 hash, and they xored the result.  and the ade thes the sean the cerser a se the pers so the firecom.  so this it's to the the and an the the that of the pering to kere it were the perat on the seriget this the interest prereed inct a stine the to go the me the lere the sers to the necertine that the  and ant the stering was peant the the wat in the drers anous the ceting all into the proper inthe the was to the be sto the be the 
    --------------------------------------------------
    -- diversity: 1.0
    -- Generating with seed:
    ove past using an md5 hash.  ssl v3 sort of fixed it a little bit by coming up with a clever solution.  they hashed both - they produced both an md5 hash and an sha-1 hash, and they xored the result. 
    --
    ove past using an md5 hash.  ssl v3 sort of fixed it a little bit by coming up with a clever solution.  they hashed both - they produced both an md5 hash and an sha-1 hash, and they xored the result.  in seme sherceps, imd to som gyrwh, be bups want to ecrep a tedntew in pdikenecbed aepit it's uncpsisteaty wreed ite wisto.  there hastrgaytataid the wastise  lote or be lebleote liks how they jast inpor treerdle eve, a ein bast's fl guall, inthe saed it that acpr titiin doubse he grebngdor thing.  and yta dobusdounsem, speicha, if in peryss.  loy.  in? bit endel recopserition.  so the 's. , , it
    --------------------------------------------------
    -- diversity: 1.2
    -- Generating with seed:
    ove past using an md5 hash.  ssl v3 sort of fixed it a little bit by coming up with a clever solution.  they hashed both - they produced both an md5 hash and an sha-1 hash, and they xored the result. 
    --
    ove past using an md5 hash.  ssl v3 sort of fixed it a little bit by coming up with a clever solution.  they hashed both - they produced both an md5 hash and an sha-1 hash, and they xored the result.  i blybwerajlinthatme.  and i hails't weme sound ne is, strfw
    
    2mat, in's  bes. sc:  s .  ih corohtiraairy panteed.  of a virdsstecr,, te 
    wot, sune and an thay ebrakpe sald mamsteally.  ne in a a beclor therrseour, ind of oduyess 10y gogke.f nt.  yea2, wo be combee goaguerod.n chenaflik no wher hi go pe.  like verolso, po  butling.y ancropewey, 3oy geeaaly meg.  it, our ukartissf seepsrttecht.. f
    --------------------------------------------------
    33458/33458 [==============================] - 4849s 145ms/step - loss: 2.2669 - val_loss: 1.9241
    Epoch 2/5
    33457/33458 [============================>.] - ETA: 0s - loss: 1.8628
    Will generate text samples using model lstm_256_1layer_wsize_200, after epoch 1
    --------------------------------------------------
    -- diversity: 0.2
    -- Generating with seed:
    ived, they said, hmm, you know, chrysler seems not to be really on the ball here.  so let's take a closer look at their technology.  to make a long story short, they will be, unless they're sued into 
    --
    ived, they said, hmm, you know, chrysler seems not to be really on the ball here.  so let's take a closer look at their technology.  to make a long story short, they will be, unless they're sued into do the sechine the proble the securing the persersed about the proble porting the perserser and a lough interne the sereching to the persers in the proble the comper the server in the perserser in the compored the server and the come a stall securing the sersers the proble the serest to the because the proble the server a so the securing the proble the interners of the proble the sereching the bec
    --------------------------------------------------
    -- diversity: 0.5
    -- Generating with seed:
    ived, they said, hmm, you know, chrysler seems not to be really on the ball here.  so let's take a closer look at their technology.  to make a long story short, they will be, unless they're sued into 
    --
    ived, they said, hmm, you know, chrysler seems not to be really on the ball here.  so let's take a closer look at their technology.  to make a long story short, they will be, unless they're sued into me the proples the proble sime wo knownt because they kind the distem the persers wet had the percare they intelm to come a able to the goy go the for look wo mecting the errast are the labt to the sers on the casting a lo the bscent into read do they was the porcally the cersesting a compredion stith spattre that some take interne terss the proble the be and it would the  is go to the cectorally 
    --------------------------------------------------
    -- diversity: 1.0
    -- Generating with seed:
    ived, they said, hmm, you know, chrysler seems not to be really on the ball here.  so let's take a closer look at their technology.  to make a long story short, they will be, unless they're sued into 
    --
    ived, they said, hmm, you know, chrysler seems not to be really on the ball here.  so let's take a closer look at their technology.  to make a long story short, they will be, unless they're sued into go you dlike 5f .  so.  it's a interne de earcens probers on thearrspati alwoh the 's was the ussupigys to it pansouse.  this de nething to do shalis reat been trerathe inco finain from these suaghine, it's no.  of to ple "mement - wifl., i lane by hothers roun bpb aple str 's its im.  so nawshating pit they had sally, these.  then ver and on theipre than sething .  i dancers veriny aust ho drseve
    --------------------------------------------------
    -- diversity: 1.2
    -- Generating with seed:
    ived, they said, hmm, you know, chrysler seems not to be really on the ball here.  so let's take a closer look at their technology.  to make a long story short, they will be, unless they're sued into 
    --
    ived, they said, hmm, you know, chrysler seems not to be really on the ball here.  so let's take a closer look at their technology.  to make a long story short, they will be, unless they're sued into . crided.b coltinkby tecah bind about to you ccare, with have ie, on and server 0- unda6" we co there alked, roucmiay s na to neads.
    
    all inhelseb abse apknead.  it, ry edy x the about ht about inverciause fiy.
    
    steve:  recese-will salk.  in to keapplo now hagervitally intgerowes sto.
    
    steve:  ticesigis allke.  they'rl nact, i this indo but ard reamen gite, pouteo9.
    
    
    nd wond theo'r  siculion tas 
    --------------------------------------------------
    33458/33458 [==============================] - 4844s 145ms/step - loss: 1.8627 - val_loss: 1.7657
    Epoch 3/5
    33457/33458 [============================>.] - ETA: 0s - loss: 1.6798
    Will generate text samples using model lstm_256_1layer_wsize_200, after epoch 2
    --------------------------------------------------
    -- diversity: 0.2
    -- Generating with seed:
    with the prescott and all of that.  i can't remember what they called the new stuff.  but they were going to get rid of it.  and then they realized that was a terrible idea.
    
    steve:  was there itanium
    --
    with the prescott and all of that.  i can't remember what they called the new stuff.  but they were going to get rid of it.  and then they realized that was a terrible idea.
    
    steve:  was there itanium a long to the ersers and they do it was the problems a like the pass in the security some the securing the because they have to the erach a securing the persesting the security to the compare a like the problems the problems in the security now the compresting the persert the persest in the securing the destated the securing the persersting the security to be the server a long to the emartin on t
    --------------------------------------------------
    -- diversity: 0.5
    -- Generating with seed:
    with the prescott and all of that.  i can't remember what they called the new stuff.  but they were going to get rid of it.  and then they realized that was a terrible idea.
    
    steve:  was there itanium
    --
    with the prescott and all of that.  i can't remember what they called the new stuff.  but they were going to get rid of it.  and then they realized that was a terrible idea.
    
    steve:  was there itanium you macher, and and they have any gever a a bicin were logg the sister trat a like stated the securing inster.  they really beanstre.  i get it's no the compine the because it's a peronesting the because then for of the ess computer corderto for the server and then you  i viry some the partes to do this the stall so the because they dit the mantis of the securing the meger that they dris a like t
    --------------------------------------------------
    -- diversity: 1.0
    -- Generating with seed:
    with the prescott and all of that.  i can't remember what they called the new stuff.  but they were going to get rid of it.  and then they realized that was a terrible idea.
    
    steve:  was there itanium
    --
    with the prescott and all of that.  i can't remember what they called the new stuff.  but they were going to get rid of it.  and then they realized that was a terrible idea.
    
    steve:  was there itanium regec, no sechine, and a like wand to or no that we know or mechinates.
    
    leo:  cale hathore.  it, so the packediling, do, this weimenel.  in bourser offort ty int, it's theie wain sting anymetion.  their ealtes nached are ip podcialicess anywoine now, they any they do that was or there is it get hather because dethere's tame scrysty of be the pecharing on the sreaing as a witf is, and me op reits
    --------------------------------------------------
    -- diversity: 1.2
    -- Generating with seed:
    with the prescott and all of that.  i can't remember what they called the new stuff.  but they were going to get rid of it.  and then they realized that was a terrible idea.
    
    steve:  was there itanium
    --
    with the prescott and all of that.  i can't remember what they called the new stuff.  but they were going to get rid of it.  and then they realized that was a terrible idea.
    
    steve:  was there itanium-zeremontlunl
    gptas eflopebs readibl itpunciment continats infalled bi's poing to bsome t unged ticomristllabrobexafi, iimeever nead camsarylipsid, no.  but you about, you gud po stuy a rezorling tity ke saatenen a lere's no theybiogd nemp cither d dingofices in mights thatw know, oha well sever inkiinil.  put all of a som nemer of more nrht.  on or ii thenernancoope it arecoment hive 3demablist. 
    --------------------------------------------------
    33458/33458 [==============================] - 4822s 144ms/step - loss: 1.6798 - val_loss: 1.7246
    Epoch 4/5
    33457/33458 [============================>.] - ETA: 0s - loss: 1.5395
    Will generate text samples using model lstm_256_1layer_wsize_200, after epoch 3
    --------------------------------------------------
    -- diversity: 0.2
    -- Generating with seed:
     want that, you know, ive got a 128-bit certificate.  but it wont light up green unless i pay them a lot of green.
    
    leo:  thats a scamola.
    
    steve:  it really - it is such a scam.  that just reallybugs
    --
     want that, you know, ive got a 128-bit certificate.  but it wont light up green unless i pay them a lot of green.
    
    leo:  thats a scamola.
    
    steve:  it really - it is such a scam.  that just reallybugs the croblem the server the security now the drive to the cresting the server in the server and it's a security now the security of security to the cromaction, and i think it was because they have to the croblem the security now the because there is the because there is it the security to the because they have a some interne the because they have any sting the persect of the server the server the 
    --------------------------------------------------
    -- diversity: 0.5
    -- Generating with seed:
     want that, you know, ive got a 128-bit certificate.  but it wont light up green unless i pay them a lot of green.
    
    leo:  thats a scamola.
    
    steve:  it really - it is such a scam.  that just reallybugs
    --
     want that, you know, ive got a 128-bit certificate.  but it wont light up green unless i pay them a lot of green.
    
    leo:  thats a scamola.
    
    steve:  it really - it is such a scam.  that just reallybugs be the security to the incernerter the oper cortering firstem.  i was they dea comporent this me som in a look we dant they so id and on the problems to the because they did securiay, it's some farine in the mess in the - and oh, the enaws the problems the are and this net the gess a don't know the blikntly strest of some of you the erast the security to security to come to says, it's got no the 
    --------------------------------------------------
    -- diversity: 1.0
    -- Generating with seed:
     want that, you know, ive got a 128-bit certificate.  but it wont light up green unless i pay them a lot of green.
    
    leo:  thats a scamola.
    
    steve:  it really - it is such a scam.  that just reallybugs
    --
     want that, you know, ive got a 128-bit certificate.  but it wont light up green unless i pay them a lot of green.
    
    leo:  thats a scamola.
    
    steve:  it really - it is such a scam.  that just reallybugs tome is use op holss, which sod'belr this atall is.  it's get loee.  and it warn stpt hay idound a blews it's would a because and it and by sncuring on tys tefute crover were way interness to the penes...
    
    leo:  a croteargat has ane plither are a backly go in com higeven u'll go con.  i realus stlf fussting.  epasse, remeorely.
    
    steve:  i don't 9's able to my we's realon youcrenulf edaess because
    --------------------------------------------------
    -- diversity: 1.2
    -- Generating with seed:
     want that, you know, ive got a 128-bit certificate.  but it wont light up green unless i pay them a lot of green.
    
    leo:  thats a scamola.
    
    steve:  it really - it is such a scam.  that just reallybugs
    --
     want that, you know, ive got a 128-bit certificate.  but it wont light up green unless i pay them a lot of green.
    
    leo:  thats a scamola.
    
    steve:  it really - it is such a scam.  that just reallybugs be that where they is reaa1k banchingr do they on edrirvice, going ote, and you know thesrido...
    
    leo:  't's a xeeent prised ip , aedd it's just sexedat on, thers able they have t. routed up of incitef stuse .n , it's just wugh sartaflh taching people cimprowontel.  ikean, morneching by plrgins like sint reany the lengcaying you need hed .
    
    steve:  mian no jost tybark like about to itpid long are
    --------------------------------------------------
    33458/33458 [==============================] - 4805s 144ms/step - loss: 1.5395 - val_loss: 1.7332
    Epoch 5/5
    33457/33458 [============================>.] - ETA: 0s - loss: 1.4350
    Will generate text samples using model lstm_256_1layer_wsize_200, after epoch 4
    --------------------------------------------------
    -- diversity: 0.2
    -- Generating with seed:
    ings.  i rebooted my server.  leo, you and i were talking a week ago about how, you know, i - now if you go to ssl labs and check out grc.com, i got a better grade than i predicted i would a week ago.
    --
    ings.  i rebooted my server.  leo, you and i were talking a week ago about how, you know, i - now if you go to ssl labs and check out grc.com, i got a better grade than i predicted i would a week ago.  i mean, it's security now the problem the persest the because they have any sturing and the persort in the persort in the perserting any so i the erally to the compore the compare the persort in the security, in the server in the server in the compare the because they have any all oke the compure the persort in the compare the end the because they have any ster the server in the security now the
    --------------------------------------------------
    -- diversity: 0.5
    -- Generating with seed:
    ings.  i rebooted my server.  leo, you and i were talking a week ago about how, you know, i - now if you go to ssl labs and check out grc.com, i got a better grade than i predicted i would a week ago.
    --
    ings.  i rebooted my server.  leo, you and i were talking a week ago about how, you know, i - now if you go to ssl labs and check out grc.com, i got a better grade than i predicted i would a week ago.  ho really got in the securing the pissware is interne to renex.  i went to the pertast the persally be the read the privers of the craching the problem a security, they have to the server because they did the example possing on the problem in the ever because it's like, it's because there is the because it's the end the server so the pight.  i think it's he s art want to read like any our the br
    --------------------------------------------------
    -- diversity: 1.0
    -- Generating with seed:
    ings.  i rebooted my server.  leo, you and i were talking a week ago about how, you know, i - now if you go to ssl labs and check out grc.com, i got a better grade than i predicted i would a week ago.
    --
    ings.  i rebooted my server.  leo, you and i were talking a week ago about how, you know, i - now if you go to ssl labs and check out grc.com, i got a better grade than i predicted i would a week ago.
    
    steve:  are this some nobled, for the  intermation.
    
    leo:  so is you roame a like, then ?
    
    steve:  reguredn pogmas.  and it was happene, laker any serfer to xhyware security to some and you loke standana stuff bitug creatory, but, in usy it tepet instare that because it was like, it's tee it, you know the pectn on the interust on this, so thatbs was of their easped and farce this is labe a me do
    --------------------------------------------------
    -- diversity: 1.2
    -- Generating with seed:
    ings.  i rebooted my server.  leo, you and i were talking a week ago about how, you know, i - now if you go to ssl labs and check out grc.com, i got a better grade than i predicted i would a week ago.
    --
    ings.  i rebooted my server.  leo, you and i were talking a week ago about how, you know, i - now if you go to ssl labs and check out grc.com, i got a better grade than i predicted i would a week ago.  i wai it's all i have ecuris machide.  thas was "really, weftrase, a h, i becive.  so twhy eanns id his a sfeve' intecnrtat.  because the okey to ;re fabmebs mich robersing the down the posfage, some noc theabpolisty.  chatige it's necent with the most cancatima grasty ibworgy whie virsof.  gen fannat.
    
    steve:  "diy cecause o laeuf the fyembly cmmed, it's compore. .  me dror our a lo, bhem on th
    --------------------------------------------------
    33458/33458 [==============================] - 4819s 144ms/step - loss: 1.4350 - val_loss: 1.7345


### Bidirectional LSTM model
This is similar to the previous model, but uses a Bidirectional LSTM. A Bidirectional LSTM is just 2 passes of LSTM executed in forward and backward direction over the input.

1. It has one Bidirectional LSTM layer with 256 hidden units.  
2. We add a dropout of 0.5 to avoid over-fitting.  
3. We add a densely connected layer with output dimension equal to number of characters in our corpus.  
4. We finally end with a softmax layer.


```python
window_size=200
step_size=4
batch_size=256
lstm_layer_size=256

character_set_length = len(unique_chars)

model_name = 'Bilstm_%d_1layer_wsize_%d' % (lstm_layer_size, window_size)
model = Sequential(name=model_name)
model.add(Bidirectional(LSTM(lstm_layer_size), input_shape=(window_size, character_set_length)))
model.add(Dropout(0.5))
model.add(Dense(character_set_length))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

train_episodes, val_episodes = train_test_split(episode_texts, train_size=0.75)

generator = EpisodesSequence(
    episode_texts=train_episodes,
    window_size=window_size,
    step_size=step_size,
    unique_chars=unique_chars,
    char_indices=char_indices,
    batch_size=batch_size
)

validation_generator = EpisodesSequence(
    episode_texts=val_episodes,
    window_size=window_size,
    step_size=step_size,
    unique_chars=unique_chars,
    char_indices=char_indices,
    batch_size=batch_size
)

def on_epoch_end(epoch, logs):
    print('\nWill generate text samples using model %s, after epoch %d' %(model.name, epoch))
    generate_sample_text(
        model,
        episode_texts,
        window_size,
        unique_chars,
        char_indices,
        indices_char
    )

train_model(model, generator, validation_generator, epochs=2,
            on_epoch_end=on_epoch_end, continue_from_checkpoint=True, lr_decay=True)
```

    /home/sumit/Venv/mlpy3venv/lib/python3.5/site-packages/sklearn/model_selection/_split.py:2026: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
      FutureWarning)


    EpisodesSequence generator initialized. Total windows: 8486766, Total batches: 33152
    EpisodesSequence generator initialized. Total windows: 2927817, Total batches: 11437
    Checkpoint folder: Bilstm_256_1layer_wsize_200_2018-03-20_02-11-51
    Epoch 1/2
    33151/33152 [============================>.] - ETA: 0s - loss: 2.2734
    Will generate text samples using model Bilstm_256_1layer_wsize_200, after epoch 0
    --------------------------------------------------
    -- diversity: 0.2
    -- Generating with seed:
    ere trying as hard as they could to attack another, well, their maximum upstream or outgoing bandwidth would be 256k, or 250k, for example, a quarter of a megabit, which is one quarter of what youre a
    --
    ere trying as hard as they could to attack another, well, their maximum upstream or outgoing bandwidth would be 256k, or 250k, for example, a quarter of a megabit, which is one quarter of what youre a that the pastine to the ore this is the protered that the to that the partered that the pastions the propletters the to and the pactered the tore the proplestions thing to the pasterters that the is that's the partered a the to it the to the alled the is the some the abletting to the pasters the to that's a that that the is the prople that that's is that's in the propletion the sore the paresters
    --------------------------------------------------
    -- diversity: 0.5
    -- Generating with seed:
    ere trying as hard as they could to attack another, well, their maximum upstream or outgoing bandwidth would be 256k, or 250k, for example, a quarter of a megabit, which is one quarter of what youre a
    --
    ere trying as hard as they could to attack another, well, their maximum upstream or outgoing bandwidth would be 256k, or 250k, for example, a quarter of a megabit, which is one quarter of what youre a lote tire oft this intes, in that the onele, bot that's bate a parpout this that so this and to prointe the torether a diated of the to that that's ding that this that the innowe thing to the is a doine to to a now been that in of that the to ant there dusted that's there a saute that it that's so it's an that's if the astisters of the eralestedititelestither the sone use to tay, so that the to t
    --------------------------------------------------
    -- diversity: 1.0
    -- Generating with seed:
    ere trying as hard as they could to attack another, well, their maximum upstream or outgoing bandwidth would be 256k, or 250k, for example, a quarter of a megabit, which is one quarter of what youre a
    --
    ere trying as hard as they could to attack another, well, their maximum upstream or outgoing bandwidth would be 256k, or 250k, for example, a quarter of a megabit, which is one quarter of what youre and the,, and a gled revicheriteresile, ca wuin, ouns.  oh't that wing to usy blanked to on forys are warl that's btpdaes, that,.
    
    leo:  s we and itge thingri's a the erali itpintonn sore, red beade catee, wis briselerersseners dinllire.  theap povaed"d is there wand i go bichsert thnuve nowres i to rondaclabled.s and prolseaser, got millike, ip ans moving, in thas  ac0
    one batusoded haldareling, s
    --------------------------------------------------
    -- diversity: 1.2
    -- Generating with seed:
    ere trying as hard as they could to attack another, well, their maximum upstream or outgoing bandwidth would be 256k, or 250k, for example, a quarter of a megabit, which is one quarter of what youre a
    --
    ere trying as hard as they could to attack another, well, their maximum upstream or outgoing bandwidth would be 256k, or 250k, for example, a quarter of a megabit, which is one quarter of what youre al  wromngt, mldtuodidoty wit's prihtlen it nes..
    soure tritelllexricall.
    
    soem:  wo dritipelys, now suvistt wur it on.
    
    noc
    poope"tiriem sofuprery.  that the rettyedo asat besoreper, aprethork bietite so suone.  an't sabpentrerontinscgnainivelrig teffocly tialy, wo hive yyothrersyllyporg i elhlefwisew of fives discce fimbte wn've, toionr theseeftn abley-opring alwallai's to anpilbyours that it to 
    --------------------------------------------------
    33152/33152 [==============================] - 9161s 276ms/step - loss: 2.2734 - val_loss: 1.9746
    Epoch 2/2
    33151/33152 [============================>.] - ETA: 0s - loss: 1.8688
    Will generate text samples using model Bilstm_256_1layer_wsize_200, after epoch 1
    --------------------------------------------------
    -- diversity: 0.2
    -- Generating with seed:
    g.
    
    leo:  and just get in the habit of not clicking links in email, period, no matter who they come from.  dont click links in email.  thats a very dangerous thing to do.
    
    steve:  yeah.
    
    leo:  eric st
    --
    g.
    
    leo:  and just get in the habit of not clicking links in email, period, no matter who they come from.  dont click links in email.  thats a very dangerous thing to do.
    
    steve:  yeah.
    
    leo:  eric ster, and the saye that the same this is the sone that this a thing to the saye the sant to the parters that the partore of the out that the saye there that the sand to the one the partert that it the pasting to the sare this is the sone that it a a so that's a there that's a come the parters the saye this is a little the sayer of the saying, there's the saye that the saye the out that this is the s
    --------------------------------------------------
    -- diversity: 0.5
    -- Generating with seed:
    g.
    
    leo:  and just get in the habit of not clicking links in email, period, no matter who they come from.  dont click links in email.  thats a very dangerous thing to do.
    
    steve:  yeah.
    
    leo:  eric st
    --
    g.
    
    leo:  and just get in the habit of not clicking links in email, period, no matter who they come from.  dont click links in email.  thats a very dangerous thing to do.
    
    steve:  yeah.
    
    leo:  eric stevers to the pactorted steved a lot little thing to all is now that which a ans a that this allerver the partore of the ore that prothers and there proplem, and the proted with problems of things a of the proplem becouse able to that is that's allike il of in the pactore there a that that mant a compering to would that this is that's now parters of the sace there windows a out that the goce, got f
    --------------------------------------------------
    -- diversity: 1.0
    -- Generating with seed:
    g.
    
    leo:  and just get in the habit of not clicking links in email, period, no matter who they come from.  dont click links in email.  thats a very dangerous thing to do.
    
    steve:  yeah.
    
    leo:  eric st
    --
    g.
    
    leo:  and just get in the habit of not clicking links in email, period, no matter who they come from.  dont click links in email.  thats a very dangerous thing to do.
    
    steve:  yeah.
    
    leo:  eric stertudeiditisidetont, i pamcal terabsection is bosions, it's of co prathe, of dond are cryntever thering to yours, .n.cthar, uptercan is go de-pas re, i reveorsconlved to will, the about if yours, where that that's are dof and  there's son be thice about.  and that's for a leculityle.  but.  out in erelistionrets of wo couls a dode, this slaw that about i beyalentee.  suaning, and forem.  arowaded 
    --------------------------------------------------
    -- diversity: 1.2
    -- Generating with seed:
    g.
    
    leo:  and just get in the habit of not clicking links in email, period, no matter who they come from.  dont click links in email.  thats a very dangerous thing to do.
    
    steve:  yeah.
    
    leo:  eric st
    --
    g.
    
    leo:  and just get in the habit of not clicking links in email, period, no matter who they come from.  dont click links in email.  thats a very dangerous thing to do.
    
    steve:  yeah.
    
    leo:  eric stcendital, duchwure ittem.  i use ard castpecast.  i'ven't ni"bledoy of beed core vintosi1cality webuse, of pi hakerins mahpersoutly a core fourm.c whey0nery's are tise us of filerkey of mellied, hilnle rvepples momatlatiusd do  manted thatsgbetady ftereed do"l because pleass an some it to inctetcine.  a padfaga t no", to liknl hadpparinglt.  i dided a know.  i alory've doupterd a cecoronayeedial h
    --------------------------------------------------
    33152/33152 [==============================] - 9163s 276ms/step - loss: 1.8688 - val_loss: 1.8152


## Observations
1. The generated text starts to feel more coherent as the number of epochs increases.  
1. The temperature parameter used during the text generation phase makes a huge difference.  
    1. When temperature is low (Eg. 0.2), the generated text has less spelling mistakes, but it gets into repetitive loops. Eg.:
    
    ```
    leo:  eric ster, and the saye that the same this is the sone that this a thing to the saye the sant to the parters that the partore of the out that the saye there that the sand to the one the partert that it the pasting to the sare this is the sone that it a a so that's a there that's a come the parters the saye this is a little the sayer of the saying, there's the saye that the saye the out that this is the s
    ```
    
    2. When temperature it set high, the text has more spelling mistakes, but it seems to be less repititive. Eg.:
    
    ```
     i wai it's all i have ecuris machide.  thas was "really, weftrase, a h, i becive.  so twhy eanns id his a sfeve' intecnrtat.  because the okey to ;re fabmebs mich robersing the down the posfage, some noc theabpolisty.  chatige it's necent with the most cancatima grasty ibworgy whie virsof.  gen fannat.
    ```

## References
1. The Unreasonable Effectiveness of Recurrent Neural Networks [[Link]](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).  
2. Keras text generation sample [[Link]](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py).  
3. Stack Overflow discussion about temperature of Softmax output [[Link]](https://stackoverflow.com/questions/37246030/how-to-change-the-temperature-of-a-softmax-output-in-keras).

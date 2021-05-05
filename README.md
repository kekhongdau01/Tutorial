# Chatbot Tutorial
Simple chatbot implementation with PyTorch. 

- NLP Theory and Concept
- Create training data
- Pytorch model
- Save/load model and implement the chat


## NLP Theory and Concept
### Tokenization
Tokenization is one of the first steps in NLP, and it’s the task of splitting a sequence of text into units with semantic meaning.The sentence would be seperate to an array with couple of string which would be easier to analyze.

Example:
"Hello, how are you?"
--> ["Hello",",","how","are","you","?"]



### Stemming
Generate the root form of the words by removing the suffix of the words which will help minimizing words.

Example:
This was not the good idea after we investivate the case.
--> Thi wa not the good idea afte we invest the case

Stemming can lose the actual meaning of the words in some case.

Example:
"Universe","university"
--> ["univers","univers"]

### Bag of words
Bag of words is a Natural Language Processing technique of text modelling. Because process the string is not a good idea for computer then we have to change them to quantitive element which would be easier for computer. Bag of words is a basically algorithm to transfer string to vector.

To do the bag of words, we have to collect all of the words in the input and put all the words in an array by tokenization.

For example:
Input contain two sentence "Hello, how are you doing?"(Greeting),"Bye, see you later."(Goodbye)

All words:["hello","how","are","you","doing","bye","see","late"]

![bag of word](https://github.com/kekhongdau01/Tutorial/blob/main/Bow.png)

This is our pipeline for the preparation step:

![Pipeline](https://github.com/kekhongdau01/Tutorial/blob/main/Pipeline.png)


## Code
```console
#Import important library  
import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


#Tokenization
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
    
    
#Stemming
def stem(word):
    return stemmer.stem(word.lower())
    
    
#Bag of words
def bag_of_words(tokenized_sentence,all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
```

### Create Training data
In this tutorial we will process json file so we need to crete a file to read and apply the preprocession . Here is a sample Json file.
```console
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey :-)",
        "Hello, what do you want to eat today?",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?"
      ]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye"],
      "responses": [
        "See you later",
        "Have a good day",
        "Bye! Enjoy your food"
      ]
    },
    {
      "tag": "thanks",
      "patterns": ["Thanks", "Thank you", "That's helpful", "Thank's a lot!"],
      "responses": ["Happy to help!", "Any time!", "My pleasure"]
    },
    {
      "tag": "feature",
      "patterns": [
        "How can you help?",
        "What is your purpose?",
        "Tell me about yourself?"
      ],
      "responses": [
        "I recommend food base on your ingridient"
      ]
    },
    {
      "tag": "beef",
      "patterns": [
        "I have beef",
      ],
      "responses": [
        "You can make beef stir fried with mixed of vegetable",
        "You can make beef deep fried with oyster sauce"
      ]
    },
    {
      "tag": "funny",
      "patterns": [
        "Tell me a joke!",
        "Tell me something funny!",
        "Do you know a joke?"
      ],
      "responses": [
        "Why did the hipster burn his mouth? He drank the coffee before it was cool.",
        "What did the buffalo say when his son left for college? Bison."
      ]
    }
  ]
}
```
In this stage we will try to collect the sentence follow each tag and transform them from sentence to array of string. This one is the most time consuming in the whole process.
```console
import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))
```

### Pytorch model
Whatever you prefer (e.g. `conda` or `venv`)
```console
mkdir myproject
$ cd myproject
$ python3 -m venv venv![Bag of word](https://user-images.githubusercontent.com/30191617/117074141-936a0d80-acf8-11eb-84b0-dcc819148e82.png)

```

### Save/load model and implement the chat
Mac / Linux:
```console
. venv/bin/activate
```
Windows:
```console
venv\Scripts\activate
```
### Install PyTorch and dependencies

For Installation of PyTorch see [official website](https://pytorch.org/).

You also need `nltk`:
 ```console
pip install nltk
 ```

If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage
Run
```console
python train.py
```
This will dump `data.pth` file. And then run
```console
python chat.py
```
## Customize
Have a look at [intents.json](intents.json). You can customize it according to your own use case. Just define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. You have to re-run the training whenever this file is modified.
```console
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?"
      ]
    },
    ...
  ]
}
```

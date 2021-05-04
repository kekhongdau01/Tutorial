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
Bag of words is a Natural Language Processing technique of text modelling. Because process the string is not a good idea for computer than we have to change them to quantitive element which would be easier for computer. Bag of words is a basically algorithm to transfer string to vector.

To do the bag of words, we have to collect all of the words in the input and put all the words in an array by tokenization.

For example:
Input contain two sentence "Hello, how are you doing?"(Greeting),"Bye, see you later."(Goodbye)

All words:["hello","how","are","you","doing","bye","see","late"]

![Bag of words training](kekhongdau01/Tutorial/bag of word.png)



## Create Training data

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

# Sentiment analysis on IMDB movie reviews

## Overview
This project has been created to gain an understanding of how recurrent neural networks (RNNs) work. Furthermore, an exploration of various RNN architectures has been made. Lastly, getting familiar with a deep learning framework, like **PyTorch**, has enabled me to write low-level, end-to-end RNN model.

## How to run
Run the project using the following command:
```
python classifier.py
```
Note that **Python 3.x or higher** is required to run the project.

The project requires the following:
- PyTorch
- NumPy
- torchtext
- matplotlib
- tqdm
- spaCy

Lastly, please note the project does not include the dataset and the word vector used and will have to be downloaded separately.

## Dataset

### About the dataset
The dataset used in this project is the [large movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/). This is one of the popular datasets used in natural language processing for sentiment analysis. This dataset contains 25000 training set movie reviews and 25000 test set movie reviews.

The dataset is already part of the [torchtext](https://github.com/pytorch/text) `datasets` module and has been used for convenience.

In this project, the training set is split into training and validation sets in order to optimize the parameters when training and validating the model. The split ratio is 70% and 30% for the training set and validation set, respectively.

### Preprocessing
For tokenization, the [spacy](https://spacy.io/) tokenizer has been used. It provides a more robust way to tokenize a sequence of text.

Then, a custom `preprocessing` function has been supplied to the torchtext `Field` class. The `preprocess` function I have added is taken from Jeremy Howard's [fast.ai](https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb) Practical Deep Learning for Coders Jupyter notebook.

In particular, the idea of this custom preprocessing function is to convert each token's encoded value to its decoded value (e.g., converting `quot;` to `'`). This makes each review easier to understand by using its proper, decoded values.

## Optimizer
Initially, the `optim.SGD` has been used with a learning rate of **1e-2**. However, I switched to an adaptive learning rate optimizer: `optim.Adam` with a learning rate of **1e-3** and keeping other parameters to its default values.

## Loss function
The loss function used is the `nn.BCEWithLogitsLoss` as this is a binary classification problem. Note that this loss function applies the sigmoid function before it calculates the binary cross entropy loss.

## Model architecture
The model architecture used is the following:
```
nn.Embedding -> nn.LSTM -> nn.Linear
```

### 1) Embedding layer
This layer is used to transform the sequence of tokens to its embedding vector representation which makes up the embedding matrix.

As a brief explanation, an **embedding matrix** contains the representation of words that contain meaningful information. This representation can be used to perform word algebra using similarity functions, like cosine similarity.

Note that weights from the **GloVe embeddings** were used as opposed to starting with random weights.

### 2) LSTM layer
This layer takes in the embedding layer's output as its input. It uses a bidirectional, 2-layer, LSTM layer.

First, it is bidirectional because the information learned from the forward pass can be useful for determining the outputs in the backward pass. As a result, the long-term dependency between each time step is established.

Second, a multi-layer LSTM is used to learn more information about the current input sequence and provides more prediction confidence at the last time step.

After the sequence went through this layer, the hidden sequences of the first and second layers are concatenated together.

### 3) Linear layer
This layer adds non-linearity to the concatenated values from the LSTM layer. This layer outputs a `1 x batch_size` vector. Note that a sigmoid function has not been applied as `nn.BCEWithLogitsLoss` applies the sigmoid function the model's output before calculating the logistic loss.

## Regularization
In order to regularize the model's predictive power, dropout has been added. In particular, `nn.LSTM` has takes an argument for `dropout` and I have utilized it with a value of **0.5** to drop some activation units when training the LSTM layer.

Also, when the hidden states are concatenated, a dropout layer is applied with probability **0.5**, by creating an `nn.Dropout` layer.

## Training and validation
The training and validation of this model is done for 10 epochs. After every training, the model is validated using the cross-validation set taken from the 30% of the training set data.

## Results
Due to hardware limitations, I have not been able to adjust the model layer's hyperparameters, like number of layers used in the LSTM or the number of hidden units used in the linear layer.

After several tweaks to some other viable hyperparameters, like dropout probability, etc., I have the following configuration:
```
LSTM:
num_layers = 2
dropout = 0.5

Linear:
hidden_size = 128
```
Also, I have recently started exploring the reduction of the vocabulary size used by the model. In particular, I have noticed that having _a smaller vocabulary_ allows the model **to focus on the most important words that determine whether a review is positive or negative**.

In order to make vocabulary size adjustments, the `max_size` property under the `Field` class is used.

### 1) **`max_size = 5000`**
**Training set accuracy:** 88.53% (loss: 0.300)

**Test set accuracy:** 87.17% (loss: 0.310)

### 2) **`max_size = 2500`**
**Training set accuracy:** 87.17% (loss: 0.315)

**Test set accuracy:** 87.03% (loss: 0.316)

Note that the best validation set accuracy is used as the final model to be evaluated on the test set.

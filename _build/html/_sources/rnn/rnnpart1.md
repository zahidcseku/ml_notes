---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Disentangling the LSTM model

Long Short Term Memory or LSTM is a popular model for sequence data. It is also a source of many confusions
for new learner. In this tutorial, I explain the LSTM from various perspective to make it understandable
to the new learners.

```{note}
There are abundance of online materials including videos and blogs to learn LSTM. I have learned a lot
from those sources (I listed a few at the end). I encourage everyone to read and learn from as many
sources as you can.
```

Long Short Term Memory or LSTM is a popular neural network model for sequence data. At the same time,
it is a source of much confusion for new learners. LSTM belongs to the family of Recurrent Neural
Networks (RNN). In this tutorial, I will explain the LSTM from various perspectives to make it more
understandable. There are many online materials including videos and blogs to learn LSTM. I have learned
much from those sources (I listed a few at the end). I encourage everyone to read and learn from as
many sources as possible.

I assume that you are convinced that LSTM is good for sequence data and can memorise the
order in sequence data. If you are not familiar with this please read [**The Unreasonable Effectiveness
of Recurrent Neural Networks**](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [**Understanding
LSTM Networks**](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). More than $90\%$ resources on
RNN and LSTM include one of the figures (or a variation of them) shown below.

```{figure} figure1.png
---
name: figure1_rnn
width: 450px
align: center
---
The most common illustrations on RNN and LSTM. (a) Time unrolled RNN model showing how RNNs process
inputs sequentially. Remember there is a single network processing the inputs. This image is illustrating the
recurrence relations among the inputs. (b) The inner connections in the LSTM model. (c) Showing variations of RNNs
based of the number of inputs and outputs. I will not discuss about all the variations in this article. I will
only mention in which category our examples belong.

Image sources: [**The Unreasonable Effectiveness of Recurrent Neural Networks**]
(https://karpathy.github.io/2015/05/21/rnn-effectiveness/),  [**Understanding LSTM Networks**]
(https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
```



## Computations in feed forward network
Before discussing LSTM, let's talk about simple feed forward neural networks (aka multi layer perceptrons or MLP). I like to think of a neural network as computational unit or a parametric function $f_{\Theta}(.)$ that takes numeric inputs ($\mathbf X$) of a certain dimension $n_X$ and produces outputs ($\mathbf y$) of dimension $n_y$ through a series of computations arranged in layers i.e., $\mathbf y = f_\Theta(\mathbf X)$ where $\Theta$ is the set of parameters i.e., weights and biases. If we have a three layer network then $\Theta$ includes weights and biases corresponding to the three layers $\{\mathbf W^{[1]}, \mathbf W^{[2]}, \mathbf W^{[3]}, \mathbf b^{[1]}, \mathbf b^{[2]}, \mathbf b^{[3]}\}$. (Read [Deep neural network notaion](https://zahidcseku.github.io/ml_notes/notations/notations.html) for a detailed discussion on deep neural network notations and computations).

```{note}
The input to a neuron unit can be of any dimension (depends on the input features or the dimension of the preceding layer's output), but an individual unit always provides one output. This implies that the number of units in a layer depends on the desired output dimension of the layer. 
```


### Code example

Let's generate a dataset $\mathbf X$ with $n_\mathcal D$ instances and feed $\mathbf X$ to a 2-layer MLP with $n^{[1]}=8$ and $n^{[2]}=1$ neuron units ($n^{[l]}$ represents the number of units in layer $l$). From the network, we get $n_y =1$ dimensional outputs $\mathbf y$ for each instance in $\mathbf X$. 

```{code-cell} ipython3
import torch
import torch.nn as nn

nd, nx = 15, 5

# number of units per layer
n1, n2 = 8, 1

# generating random inputs
X = torch.rand(nd, nx)

# create the model
layer1 = nn.Linear(nx, 8)
layer2 = nn.Linear(8, n2)

net = nn.Sequential(layer1, layer2)

# feed the inputs to the network
y = net(X)

print(y)
```

Here, `net()` transforms $\mathbf X \in \mathbb R^5$ to produce outputs $\mathbf y\in\mathbb R$. First, `layer1` takes $\mathbf X$ and produces outputs $\mathbf a^{[1]}$ (dimension $(n_\mathcal D, n^{[1]})$). Then, ativations $\mathbf a^{[1]}$ are fed to `layer2` and layer 2 generates $\mathbf a^{[2]} = \mathbf y$ (dimension $(n_\mathcal D, n_y)$). We consider liner activations in all layers. 

From [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) we find: `nn.Linear` transforms the data as $y = xA^\top + b$. We also learn that the shapes of the weights $A$ are `(out_features, in_features)` and biases are `(out_features)`. Hence, for PyTorch, the shapes of weights and biases of layer 1 are `shape($A^{[1]}$)` = $(n^{[1]}, n_X)$ = $(8, 5)$ and `shape($b^{[1]}$)=$8$`. The shapes of weights and biases of layer 2 are `shape($A^{[2]}$)` = $(n^{[2]}, n^{[1]})$ = $(1, 8)$ and `shape($b^{[2]}$)`=$1$. Let’s check the shapes in pytorch: 

```{code-cell} ipython3
print(f"Shape of layer 1 weight matrix: {layer1.weight.size()}")
print(f"Shape of layer 1 bias vector: {layer1.bias.size()}")
print(f"Shape of layer 2 weight matrix: {layer2.weight.size()}")
print(f"Shape of layer 2 bias vector: {layer2.bias.size()}")
```

We can investigate the outputs of the layers using the equation  $y = xA^\top + b$. The layer wise computations can be given by:

$$
a^{[1]} = xA^{[1]\top} + b^{[1]}\\
a^{[2]} = xA^{[2]\top} + b^{[2]}
$$

For layer 1 outputs we used $a$ to represent activations and for layer 2 the activations $a^{[2]}$ are the outputs $\mathbf y$. We can implement the equation in PyTorch without passing the inputs through the `net()`:

```{code-cell} ipython3
# output using pytorch layer
a1 = layer1(X)

# output using equation
# get the weights and biases in layer 1
w1 = layer1.weight.data
b1 = layer1.bias.data
a1_eq = torch.matmul(X, w1.T) + b1

print(f"Layer 1 outputs using pytorch and equation are same: {torch.equal(a1,a1_eq)}")

# output using pytorch layer
a2 = layer2(a1)

# output using equation
# get the weights and biases in layer 2
w2 = layer2.weight.data
b2 = layer2.bias.data
a2_eq = torch.matmul(a1, w2.T) + b2

print(f"Layer 2 outputs using pytorch and equation are same: {torch.equal(a2,a2_eq)}")
```

The network processes $n_\mathcal D$ instances at a time through matrix multiplication. However, we can achieve the same results using a loop as follows (this simple exercise will help us understand LSTM computations discussed below). 

```python
# initialize a variable same shape as y to store the step wise results
loop_y = torch.zeros_like(y)

for i in range(X.size()[0]):
    loop_y[i] = net(X[i])

print(loop_y)
```

```{note}
When implementing neural networks using any framework, such as PyTorch, Keras, TensorFlow, etc., it's important to understand the inputs and outputs specific to that framework. This article will focus on PyTorch.
```

From PyTorch documentation we find the input and output specification of linear layer `nn.Linear` as input: $(*, H_{in})$ and output: $(*, H_{out})$. $H_{in}$ and $H_{out}$ are the input and output dimensions of a layer. According to the above example task for the first layer $H_{in}=5$ and $H_{out}=8$. The $*$ symbol means that whatever many of $H_{in}$ dimensional instances we feed into the linear layer we get the same number of $H_{out}$ dimensional outputs. The $*$ depends on the batch size (the batch size is the number of instances we want to process at a time). For a batch size of 15 the inputs will be $(15, 5)$ and outputs of the layer will be $(15, 8)$.


## What is sequence data anyway?
**Difference between - input dimension, sequence length and batch size.** 

LSTM models process sequence data. Sequence data refers to **any data where the order of instances is important**. Text data, which is often used to explain LSTM models, is a prime example of sequence data. The order of words in a sentence can significantly influence its meaning. For instance, the same words can convey different meanings based on their context or position. Time series data, where observations are collected with timestamps, is another example of sequence data. The timestamps establish the sequence of the observations.

Let's consider a dataset $\mathbf X$ consisting of $n_S$ sequences. Each sequence $\mathbf X^{<s>} \in \mathbf X$ is made up of $n_E$ elements. Notice the $\_^{<>}$ notation in sequences which is different from $\_^{()}$. We use $\_^{(i)}$ to indicate the $i$-th instance in a flat dataset. Each element $\mathbf x_e^{<s>}|(s=1,2,\dots,n_S)$ of a sequence $\mathbf X^{<s>} \in \mathbf X$ consists of $n_X$ real numbers, meaning $\mathbf x_e^{<s>}$ is a $n_X$ dimensional vector.

In sequence data, both the length of the sequence $\mathbf X^{<s>}$, $n_E$ in our notation, and the length of each element $\mathbf x_e^{<s>}$, $n_X$ in our notation, within the sequence are important considerations. Remember that each input must be a sequence of $n_E$ elements, with each element being a vector of a certain length $n_X$.


$\mathbf X = [\mathbf X^{<1>}, \mathbf X^{<2>},\dots, \mathbf X^{<n_S>}]$

$\mathbf X^{<s>} = [\mathbf x^{<s>}_1, \mathbf x^{<s>}_2, \dots, \mathbf x^{<s>}_{n_E}]$

$\mathbf X^{<s>} = [\mathbf X^{<s>(1)}, \mathbf X^{<s>(2)}, \dots, \mathbf X^{<s>(n_E)}]$

$\mathbf X^{<s>} = [\mathbf X^{<s>}_1, \mathbf X^{<s>}_2, \dots, \mathbf X^{<s>}_{n_E}]$

$\mathbf x_e^{<s>}=[x^{<s>}_{e1}, x^{<s>}_{e2}, \dots, x^{<s>}_{en_X}]$

Inputs to an LSTM layer can be batched or unbatched. Unbatched inputs require a 2D tensor of shape $(n_E, n_X)$, meaning there are $n_E$ elements each with $n_X$ values. Batched inputs, on the other hand, are a collection of sequences. The shape of batched inputs is similar to the shape of our dataset $\mathbf X$ $(b_S, n_E, n_X)$, indicating that there are $b_S$ sequences $\mathbf X^{<1>}, \dots , \mathbf X^{<b_S>}$, and each $\mathbf X^{<s>}$ is a $(n_E, n_X)$ dimensional tensor. Usually, $b_S<n_S$. We call $b_S$ the **batch size**. 

| Symbol | Dimension | Definition |
| --- | --- | --- |
| $\mathbf X$ | $(n_S, n_E, n_X)$ | A data set consisting of $n_S$ sequences of length $n_E$. Each element in a sequence in $\mathbf X$ is $n_X$ dimensional.  |
| $n_S$ | 1 | Number of sequences in the dataset. |
| $n_E$ | 1 | Number of elements in each sequence. We call this the sequence length. |
| $n_X$ | 1 | Number of values in each element. We call this the input dimension or the input size. |
| $b_S$ | 1 | Number of sequences in a batch. We call this the batch size. |
| $\mathbf X^{<s>}$ | $(n_E, n_X)$ | The $s$-th sequence in our dataset.   |
| $\mathbf x_e^{<s>}$  | $(1, n_X)$ | The $e$-th element of the $s$-th sequence. |
| $x^{<s>}_{ei}$ | 1 | A real value corresponding to the $i$-th dimension of the $e$-th element of the $s$-th sequence.  |

Let’s see two examples of sequence data using our notation. 

### Example (text data)
The IMDB dataset contains 50,000 movie reviews, labeled as positive or negative. Each review is a sequence of words, and the machine learning task is to predict the sentiment of the review. What are the $n_S$, $n_E$, and $n_X$?

Each review is a sequence of words i.e., there are 50,000 sequences in the dataset ($n_S = 50000$). Each review has a certain number of words. Let’s restrict the number of words in the reviews to be 100 i.e., if a review has more than 100 words, we ignore the rest and if it has less than 100 words we add a special marker called `pad`. This gives the sequence length $n_E = 100$. We cannot use “words” as input to our neural net. We first need to convert the words to a numeric representation. The representations are known as word embeddings. After the transformation of the words to numbers, we get certain dimensional vectors for each word. This embedding dimension is our $n_X$. If we choose the embedding dimension to be 512 then $n_X=512$. To summarize, $\mathbf X \in \mathbb R^{50000\times100\times512}$ includes 50000 reviews, $\mathbf X^{<s>} \in \mathbb R^{100\times512}$, $\mathbf x_e^{<s>} \in \mathbb R^{100}$ and $x_{ei}^{<s>} \in \mathbb R$ is the actual number corresponding to the $i$-th embedding dimension of the $e$-th word in the $s$-th review. The batch size $b_S$ depends on the developer.

### Example (non text data)
The [Daily Minimum Temperatures in Melbourne](https://archive.ics.uci.edu/ml/machine-learning-databases/00356/daily-min-temperatures.csv) dataset contains daily minimum temperatures in Melbourne, Australia, recorded over 10 years. Each data point is a temperature reading, and the machine learning task is to predict future temperatures based on past readings. Each temperature reading is part of a time series. The value at each time step depends on previous values.

For 10 years, we have roughly 3650 records. If we consider the prediction task is to predict the next month's average temperature based on past months' temperatures, then we have $n_S = 1216$ sequences. Each sequence $\mathbf X^{<s>}$ has 30 elements, i.e., $n_E=30$. Each element $\mathbf x_e^{<s>}$ has the corresponding temperature, i.e., $n_X=1$. For this example, $\mathbf x_e^{<s>}=x_{ei}^{<s>}$ as we only have 1D observations. Again, batch size $b_S$ depends on the developer.


## Architectures (ref: {ref}`Figure <figure1_rnn>` (c))

Both of the above examples use the **many-to-one** architecture ({ref}`Figure <figure1_rnn>`(c)) because they involve processing a sequence of inputs to produce a single output. For the sentiment analysis task, the input is a sequence of words (many), and the output is a single sentiment label (one). The model processes the entire sequence of words to predict whether the review is positive or negative. For the second example, the input is a sequence of temperature readings over 30 days (many), and the output is the temperature for the next day (one). The model uses the sequence of past temperatures to predict the next value.

In both cases, the RNN (or LSTM) processes each sequence, maintaining a hidden state that captures the context, and finally produces a single output representing the sentiment or temperature.


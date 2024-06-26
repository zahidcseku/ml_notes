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
name: figure1
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
When implementing neural networks using any framework, such as Pytorch, Keras, TensorFlow, etc., it's important to understand the inputs and outputs specific to that framework. This article will focus on PyTorch.
```

From PyTorch documentation we find the input and output specification of linear layer `nn.Linear` as input: $(*, H_{in})$ and output: $(*, H_{out})$. $H_{in}$ and $H_{out}$ are the input and output dimensions of a layer. According to the above example task for the first layer $H_{in}=5$ and $H_{out}=8$. The $*$ symbol means that whatever many of $H_{in}$ dimensional instances we feed into the linear layer we get the same number of $H_{out}$ dimensional outputs. The $*$ depends on the batch size (the batch size is the number of instances we want to process at a time). For a batch size of 15 the inputs will be $(15, 5)$ and outputs of the layer will be $(15, 8)$.
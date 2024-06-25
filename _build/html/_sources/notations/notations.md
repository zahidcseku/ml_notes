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

# Deep neural network notation

```{contents}
```

A deep neural network (DNN) consists of multiple layers of computational units, called neurons. Each layer is arranged sequentially, and the units in each layer are arranged vertically. Each neuron (or unit) takes some input and produces an output. Consequently, a layer takes some inputs and produces some outputs using simple computations. When we say each layer takes inputs, we mean that each neuron in a layer takes inputs. In this section, I will discuss various notations used in deep neural networks. Most of the notations are adapted from Andrew Ng’s deep learning specialization course. The notations involve multiple dimensions that can sometimes be confusing (actually, all the time when you start learning from different sources). When I started learning DNN, I got familiar with one set of notations from Andrew’s course and started representing other materials using the same notations (it was not a trivial task at the beginning but helped me a lot).

## Inputs and outputs of a single neuron
Each neuron takes arbitrary $n_X$ dimensional inputs $\mathbf X$ and produces a single output $a$. 

```{figure} single-neuron.png
---
name: single_neuron
width: 200px
align: center
---
A single neural processing unit which takes $n_X$ dimensional inputs and generates a scalar output $a$.
```

The neuron's output is calculated in two steps, as indicated by a circular + symbol and a rectangular $\sigma$ symbol. A weighted sum is first computed as follows:

$$z = x_1*w_1 + x_2*w_2 + \cdots + x_{n_X}*w_{n_X} + b = \sum_{i=1}^{n_X}x_i*w_i +b$$

And then an activation function $\sigma$ is applied to $z$ to compute the output (called activation) of the unit as follows:

$$a = \sigma\left(z\right)$$

If we express the input and the weights as row vectors $\mathbf W = [w_1, w_2, \cdots, w_{n_X}]$ and $\mathbf X  =[x_1, x_2,\dots,x_{n_X}]$, we can achieve the same result using a vector operation:

$$a = \sigma(z) = \sigma(\mathbf W \cdot \mathbf X^\top + b)$$


**Shapes**: both the $\mathbf W$ and $\mathbf X$ are row vectors of shape $(1, n_X)$ and both $z$ and $a$ are scalars.

```{list-table} Notations - single neuron
:header-rows: 1
:name: notation-single-neuron

* - Symbol
  - Shape	
  - Definition
* - $\mathbf X$	
  - $(1, n_X)$	
  - The input row vector. 
* - $n_X$	
  - 1	
  - The dimension of the inputs.
* - $x_j$	
  - $1$	
  - The value of the $j$-th feature of an input object.
* - $\mathbf W$	
  - $(1, n_X)$	
  - Weight vector of the neuron.
* - $b$	
  - 1	
  - Bias parameter of the neuron.
* - $z$	
  - 1	
  - The intermediate output of the neuron before applying the activation function.
* - $a$
  - 1	
  - The output (or activation) of the neuron after applying the activation function $\sigma$ to $z$.
* - $\mathbf X^\top$
  - $(n_X, 1)$	
  - $\mathbf X$ transposed.
```


## Inputs and outputs of a single layer
A layer in a neural network is created by stacking multiple neural units. Each unit takes the same number of real numbers ($n_X$) and produces a single output. The number of outputs of a layer is determined by the number of units in the layer. Representing a layer by $l$, the number of units in the layer is expressed as $n^{[l]}$.

```{figure} single_layer.png
---
name: single_layer
width: 200px
align: center
---
A single layer processing unit that takes $n_X$ dimensional inputs and produces $n^{[l]}$ dimensional outputs. Here, $n^{[l]}$ is the number of processing units in layer $l$.
```

Now that we have multiple units, we can't express the weights as a vector anymore. The weights of each individual unit in the layer are still a vector (shape $(1, n_X)$), and we need a matrix notation (by stacking the vectors) to express the weights of layer $l$. For $n^{[l]}$ units, we will have $n^{[l]}$ biases expressed as $\mathbf b^{[l]}$, which is a vector of shape $(n^{[l]}, 1)$. We have multiple outputs from a layer, hence both $a$ and $z$ are now vectors of shape $(n^{[l]}, 1)$ and represented by: $\mathbf z^{[l]}$ and $\mathbf a^{[l]}$ respectively. Layer $l$ performs the following operations:

$$
a^{[l]}_1 = \sigma^{[l]} (z^{[l]}_1) = \sigma^{[l]} \left(w^{[l]}_{11}x_1 + w^{[l]}_{12}x_2+\dots+w^{[l]}_{1n_X}x_{n_X}+b^{[l]}_1\right) = \sigma^{[l]} \left(\sum_i^{n_X} w^{[l]}_{1i}x^i + b^{[l]}_1\right)
$$

$$
a^{[l]}_2 = \sigma^{[l]} (z^{[l]}_2) = \sigma^{[l]} \left(w^{[l]}_{21}x_1 + w^{[l]}_{22}x_2+\dots+w^{[l]}_{2n_X}x_{n_X}+b^{[l]}_2\right)=\sigma^{[l]}\left(\sum_{i}^{n_X} w^{[l]}_{2i}x_i + b^{[l]}_2\right)
$$

$$\dots$$

$$a^{[l]}_{n^{[l]}} = \sigma^{[l]} (z^{[l]}_{n^{[l]}}) = \sigma^{[l]} \left(w^{[l]}_{n^{[l]}1}x_1 + w^{[l]}_{n^{[l]}2}x_2+\dots+w^{[l]}_{n^{[l]}n_X}x_{n_X}+b^{[l]}_{n^{[l]}}\right)=\sigma^{[l]}\left(\sum_{i}^{n_X} w^{[l]}_{n^{[l]}i}x_i + b^{[l]}_{n^{[l]}}\right)$$

We can implement the above computations in python using only multiplications and additions.

```python
output = []
for i in range(nl):
   z = 0
   for j in range(n):
      z += W[i][j] * x[j]
   z += b[i]
   a = sigma(z)
   
   output.append(a)
```

The equations and computations can be simplified using matrix operations. The weights of the matrix can be represented by:

$$
\mathbf W^{[l]} = \begin{bmatrix} w^{[l]}_{11} & w^{[l]}_{12} & \cdots & w^{[l]}_{1n_X}\\ w^{[l]}_{21} & w^{[l]}_{22} & \cdots & w^{[l]}_{2n_X}\\\cdots&\cdots&\cdots&\cdots\\w^{[l]}_{n^{[l]}1} & w^{[l]}_{n^{[l]}2} & \cdots & w^{[l]}_{n^{[l]}n_X}\end{bmatrix}
$$

The dimension of $\mathbf W^{[l]}$ depends on the input dimension $n_X$ and the number of units (which is same as the output dimension of the layer) $n^{[l]}$ i.e., $(n^{[l]}, n_X)$. Using the matrix notations, the computations are:

$$\mathbf a^{[l]} = \sigma^{[l]} (\mathbf z^{[l]}) = \sigma^{[l]}\left(\mathbf W^{[l]} \cdot \mathbf X^\top + \mathbf b^{[l]}\right)$$

In python implementation,

```python
z = np.matmul(W, x.T) + b
a = sigma(z)
```


```{list-table} Notations - single layer of neurons
:header-rows: 1
:name: notation-single-layer
* - Symbol
  - Shape
  - Definition
* - $l$
  - $1$ 
  - The layer id.  
* - $n^{[l]}$ 
  - $1$
  - Number of units in layer $l$.
* - $\mathbf b^{[l]}$
  - $(n^{[l]}, 1)$ 
  - The bias vector corresponding to layer $l$. 
* - $b^{[l]}_i$ 
  - $1$
  - Bias of the $i$-th unit in layer $l$.
* - $\mathbf W^{[l]}$
  - $(n^{[l]}, n_X)$
  - The weights corresponding to the layer $l$.
* - $w^{[l]}_{ij}$
  - $1$
  - The weight corresponding to unit $i$ and feature $j$ in layer $l$.
* - $\mathbf a^{[l]}$
  - $(n^{[l]}, 1)$
  - Activations (outputs) of layer $l$.
* - $\mathbf z^{[l]}$
  - $(n^{[l]}, 1)$
  - The intermediate outputs of the neurons given the input  $\mathbf X$ before applying the activation function.
* - $a_i^{[l]}, z_i^{[l]}$
  - $1$
  - Elements of $\mathbf a^{[l]}$ and $\mathbf z^{[l]}$.
* - $\sigma^{[l]}$
  - $~~$
  - Activation function corresponding to layer $l$.
```


## Inputs and outputs of a multiple layers

A single neuron unit or a single layer network is never used for solving practical machine learning problems. Building upon the discussion of a single neuron unit and a single layer of neurons, we now discuss the practical scenario where multiple layers of neurons are arranged sequentially as shown in the figure below. The input $\mathbf X \in \mathbb R^{n_X}$ is fed to the model at layer $l=1$ which processes the input and produces an output vector $\mathbf a^{[1]} \in \mathbb R^{n^{[1]}}$. The layer $2$ receives inputs $\mathbf a^{n^{[1]}} \in \mathbb R^{n^{[1]}}$ from layer $1$ and produces outputs $\mathbf a^{[2]} \in \mathbb R^{n^{[2]}}$ and so on. Remember that each neuron receives the same dimensional inputs and the output dimension of a layer is equal to the number of units. Each layer (except the first layer) receives input from the previous layer. The number of units in a layer is a design choice (except the last layer $L$) determined through hyperparameter search. The last layer from which we obtain the outputs of the model is called the output layer. The input dimension of the first layer and the number of units of layer $L$ (or the output dimension of layer $L$) depends on the task. For example, for the digit classification of $28\times 28$ images of digits, the input dimension $n_X=28*28=784$ and the output dimension or the number of units in layer $L$ is $10$.


```{figure} multi_layer_model.png
---
name: multi_layer_model
width: 500px
align: center
---
A multilayer deep neural network (or a multilayer perceptron) model. Each layer $l$ consists of $n^{[l]}$ nuerons. 
```

All the symbols used in the above [figure](multi_layer_model) are covered in the previous section: Inputs and Outputs of a Single Layer. Often, we use $\mathbf {\hat y}$ to represent the output of the model. So, we can say $\mathbf{\hat y}=\mathbf a^{[L]}$. In addition to our previous symbols, we introduce the following symbol for multilayer models.


```{list-table} Notations - multiple layers of neurons
:header-rows: 1
:name: notation-multiple-layers
* - Symbol
  - Shape
  - Definition
* - $L$
  - 1
  - The number of layers.
* - $1,2,\dots,l,\dots,L$
  - $1$
  - Layer id.
* - $\mathbf{\hat y} = \mathbf a^{[L]}$
  - $(a^{[L]}, 1)$
  - Output of the model.
* - $\mathbf X = \mathbf a^{[0]}$
  - $(1, n_X)$
  - Inputs in terms of activation.
```

Using the activation notations, $\mathbf a^{[0]}=\mathbf X^\top$ and $\mathbf a^{[L]}=\mathbf{\hat y}$ we can draw as general eqaution ov the network as:

$$\mathbf z^{[l]} = W^{[l]}\cdot \mathbf a^{[l-1]} + \mathbf b^{[l]}$$

$$a^{[l]} = \sigma^{[l]}\left (\mathbf z^{[l]}\right)$$

for $l=1,2,\dots,L$


## Dataset

In the above discussion, we considered our input $\mathbf X\in\mathbb R^{n_X}$ to be a $n_X$ dimensional row vector. However, in real applications, the input is a $2$ or $3$ dimensional tensor (we consider 2d cases only for now). For supervised learning, we have many instances of $\mathbf X$ and their associated labels $\mathbf y$. For unsupervised learning tasks, we do not have $\mathbf y$. We represent the dataset as $\mathcal D=(\mathbf X, \mathbf y)$. The shape of $\mathbf X$ is given by $(n_{\mathcal D}, n_X)$ and the shape of $\mathbf y$ is $(n_\mathcal D, n_y)$. To refer to any instance of $\mathcal D$, we use the superscript notation i.e., $\mathbf X^{(i)}$ refers to the $i$-th instance of $\mathbf X$ and $\mathbf y^{(i)}$ is the corresponding label. The superscript notation can also be used to refer to any layer computation, e.g., $\mathbf z^{(i)[l]}$ refers to the output of layer $l$ corresponding to the $i$ instance in the dataset.

```{note} we used the same $\mathbf X$ to refer to the input as a vector above for simplicity. We should have used $\mathbf X^{(i)}$. With this dataset notation in the above computations, all output shapes will change as follows: $\mathbf z^{[l]}$ and $\mathbf a^{[l]}$ now include $n_{\mathcal D}$ instances corresponding to the number of samples in the dataset.
```

```{note} For efficiency, the dataset is divided into batches of inputs and the network processes a batch at a time rather than the entire batch at a time.
```

### Example

The [iris dataset:](https://archive.ics.uci.edu/dataset/53/iris) includes 150 instances of iris flower. Each isntance is described by four feautures {sepal length, sepal width, petal length, petal width}. In our notation,  the iris dataset can be described as follows:

- $n_{\mathcal D} =150$
- $n_X=4$
- $n_y=1$
- $\mathbf X^{(2)}= [4.9, 3.0, 1.4, 0.2]$
- $\mathbf y^{(2)} = [C1]$
- $x^{(150)}_3=5.1$

```{figure} dataset.png
---
name: dataset
width: 250px
align: center
---
The iris dataset.
```

## Code example
In this code example, we will generate 10 random instances of 5 dimensional inputs i.e., $n_\mathcal D=10$, $n_X=5$. Let's consider the network consists of three layers ($L=3$) with $n^{[1]} = 8$, $n^{[2]} = 12$, and $n^{[3]} = 2$. The dimensions of our weight metrices will be $\mathbf W^{[1]}$, $\mathbf W^{[2]}$ and $\mathbf W^{[3]}$ are $(8, 5)$, $(12, 8)$ and $(2, 12)$. The shapes of the biases $\mathbf b^{[1]}$, $\mathbf b^{[2]}$ and $\mathbf b^{[3]}$ are $(8, 1)$, $(12, 1)$ and $(2, 1)$. We will generate the parameters randomly and using the equations discussed above we will compute $\mathbf a^{[l]}, l = 1,2,3$. We assume $\mathbf a^{[l]} = \mathbf z^{[l]} | l =1,2,3$ i.e., we do not apply the activation functions. 

```{code-cell} ipython3
# set up the variables

import numpy as np

nd, nx, nl = 10, 5, 3
n1, n2, n3 = 8, 12, 2

# input and parameters initializations
x = np.random.rand(nd, nx)
W1 = np.random.rand(n1, nx)
b1 = np.random.rand(n1, 1)
W2 = np.random.rand(n2, n1)
b2 = np.random.rand(n2, 1)
W3 = np.random.rand(n3, n2)
b3 = np.random.rand(n3, 1)

# layer wise computations
a0 = x.T
a1 = np.matmul(W1, a0) + b1
a2 = np.matmul(W2, a1) + b2
a3 = np.matmul(W3, a2) + b3

print(f"Outputs of the network")
print(a3)
```

Let's, verify the results using pytorch. We will initialize the pytorch parameters and inputs using the same random tensors as our previous example.

```{code-cell} ipython3
import torch 

import torch.nn as nn

l1 = nn.Linear(nx, n1)
l2 = nn.Linear(n1, n2)
l3 = nn.Linear(n2, n3)

tensorX = torch.from_numpy(x)

# initialize same weights
l1.weight.data = torch.from_numpy(W1)
l1.bias.data = torch.from_numpy(b1.squeeze())

l2.weight.data = torch.from_numpy(W2)
l2.bias.data = torch.from_numpy(b2.squeeze())

l3.weight.data = torch.from_numpy(W3)
l3.bias.data = torch.from_numpy(b3.squeeze())

a1 = l1(tensorX)
a2 = l2(a1)
a3 = l3(a2)

print(a3)
```

```{note} 
pytorch uses broadcasting in computations. Here, we used `squeeze()` to match the shape of the bias parameter with pytorch internals. 
```

We see that outputs of both implementations are identical. That's what we expect!


## Summary

An entire multilayer neural network can be expressed as a set of parameters - the weights and biases.

$\mathbf{\hat y} = f_\Theta(\mathbf X)$ where $\Theta = \{\mathbf W^{[l]}, \mathbf b^{[l]}~|~l = 1,2, \dots, L\}$.

Let's consider the number of units of a 3 layer network to be $n^{[1]}=8, n^{[2]}=16, n^{[3]}=3$ and the input dimension $(n_\mathcal D, n_X) = (500, 4)$. In layer 1, we have 8 units i.e., the output dimensions of layer 1 (also the input dimension of layer 2) is 8, $\mathbf W^{[1]}$ will have 8 rows (weights of each unit is a row in the weight matrix) and each unit of layer 1 will process 4 dimensional inputs (as $n_X=4$) that means each unit will require 4 weights i.e, $\mathbf W^{[1]}$ has 4 columns. The shape of $\mathbf W^{[1]}$ is $(8, 4)$. In our notation, the input $\mathbf X^\top = \mathbf a^{[0]}$, the output of layer 1 is $\mathbf a^{[1]}$ which is $8$ dimensional that implies each unit of layer 2 will process $8$ dimensional inputs i.e., $\mathbf W^{[2]}$ will have $8$ columns and with $n^{[2]}=16$ units the number of rows in $\mathbf W^{[2]}$ is $16$. Similarly, the shape of $\mathbf W^{[3]}$ is $(3, 16)$.

The dimensions of $\mathbf W$ and $\mathbf b$ are:

- For, $l=1$, shape of $\mathbf W^{[1]}=(8,4)$ and shape of $\mathbf b^{[1]}=(8,1)$.
- For, $l=2$, shape of $\mathbf W^{[2]}=(16,8)$ and shape of $\mathbf b^{[2]}=(16,1)$.
- For, $l=3$, shape of $\mathbf W^{[3]}=(3,16)$ and shape of $\mathbf b^{[3]}=(3,1)$.

Computations in the network are:

$$\mathbf a^{[1]} = \sigma^{[1]}\left(W^{[1]}\cdot \mathbf a^{[0]} + \mathbf b^{[1]}\right)$$

$$\mathbf a^{[2]} = \sigma^{[2]}\left(W^{[2]}\cdot \mathbf a^{[1]} + \mathbf b^{[2]}\right)$$

$$\mathbf a^{[3]} = \sigma^{[3]}\left(W^{[3]}\cdot \mathbf a^{[2]} + \mathbf b^{[3]}\right)=\mathbf{\hat y}$$

The shapes of $a^{[1]}, a^{[2]}$ and $a^{[3]}$  are $(8, 500), (16, 500)$ and $(3, 500)$. 


In future, I will discuss the forward and backward propagation through a neural network using a complete example. Stay tuned:

<style>
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview {
display: flex !important;
flex-direction: column !important;
justify-content: center !important;
margin-top: 30px !important;
padding: clamp(17px, 5%, 40px) clamp(17px, 7%, 50px) !important;
max-width: none !important;
border-radius: 6px !important;
box-shadow: 0 5px 25px rgba(34, 60, 47, 0.25) !important;
}
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview,
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview *{
box-sizing: border-box !important;
}
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview .preview-heading {
width: 100% !important;
}
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview .preview-heading h5{
margin-top: 0 !important;
margin-bottom: 0 !important;
}
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview .preview-input-field {
margin-top: 20px !important;
width: 100% !important;
}
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview .preview-input-field input {
width: 100% !important;
height: 40px !important;
border-radius: 6px !important;
border: 2px solid #e9e8e8 !important;
background-color: #fff;
outline: none !important;
}
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview .preview-input-field input {
color: #000000 !important;
font-family: "Montserrat" !important;
font-size: 14px;
font-weight: 400;
line-height: 20px;
text-align: center;
}
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview .preview-input-field input::placeholder {
color: #000000 !important;
opacity: 1 !important;
}

.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview .preview-input-field input:-ms-input-placeholder {
color: #000000 !important;
}

.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview .preview-input-field input::-ms-input-placeholder {
color: #000000 !important;
}
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview .preview-submit-button {
margin-top: 10px !important;
width: 100% !important;
}
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview .preview-submit-button button {
width: 100% !important;
height: 40px !important;
border: 0 !important;
border-radius: 6px !important;
line-height: 0px !important;
}
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .form-preview .preview-submit-button button:hover {
cursor: pointer !important;
}
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .powered-by-line {
color: #231f20 !important;
font-family: "Montserrat" !important;
font-size: 13px !important;
font-weight: 400 !important;
line-height: 25px !important;
text-align: center !important;
text-decoration: none !important;
display: flex !important;
width: 100% !important;
justify-content: center !important;
align-items: center !important;
margin-top: 10px !important;
}
.followit--follow-form-container[attr-a][attr-b][attr-c][attr-d][attr-e][attr-f] .powered-by-line img {
margin-left: 10px !important;
height: 1.13em !important;
max-height: 1.13em !important;
}

</style>
<div class="followit--follow-form-container" attr-a attr-b attr-c attr-d attr-e attr-f>
<form data-v-c76ccf54="" action="https://api.follow.it/subscription-form/TUo5R2xpLzYwVVJQeER5Sk5HeXpaUFk2WXlWRXVLUUhNeWgxeVg4V1c4Q0FGVmFHSEttNXlnVkdmTUJCZDBpMlV4MlJwaVpGa2NOc1hOeGdGU2x0eTZVV040Y2pZcTdrMWt1RHRsRzJtRUVhLzg1Nzh2anU5NkF3UE5zZ3RJVlN8K3gzWVA2OGRldnNMVlFHTTIrc2x5MG1tL1ZrTjdEL2xzaktHZ1A3RmlYWT0=/21" method="post"><div data-v-c76ccf54="" class="form-preview" style="background-color: rgb(255, 255, 255); position: relative;"><div data-v-c76ccf54="" class="preview-heading"><h5 data-v-c76ccf54="" style="text-transform: none !important; font-family: Arial; font-weight: bold; color: rgb(0, 0, 0); font-size: 16px; text-align: center;">Get notification of new posts:</h5></div><div data-v-c76ccf54="" class="preview-input-field"><input data-v-c76ccf54="" type="email" name="email" required="" placeholder="Enter your email" spellcheck="false" style="text-transform: none !important; font-family: Arial; font-weight: normal; color: rgb(0, 0, 0); font-size: 14px; text-align: center; background-color: rgb(255, 255, 255);"></div><div data-v-c76ccf54="" class="preview-submit-button"><button data-v-c76ccf54="" type="submit" style="text-transform: none !important; font-family: Arial; font-weight: bold; color: rgb(255, 255, 255); font-size: 16px; text-align: center; background-color: rgb(0, 0, 0);">Subscribe</button></div></div></form><a href="[https://follow.it](https://follow.it/)" class="powered-by-line">Powered by <img src="https://follow.it/static/img/colored-logo.svg" alt="[follow.it](http://follow.it/)" height="17px"/></a>
</div>

<div id="disqus_thread"></div>
<script>
/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables    */
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://https-zahidcseku-github-io-ml-notes.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>

<script id="dsq-count-scr" src="[//https-zahidcseku-github-io-ml-notes.disqus.com/count.js](notion://https-zahidcseku-github-io-ml-notes.disqus.com/count.js)" async></script>
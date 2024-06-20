# Deep neural network notation

A deep neural network consists of multiple layers of computational units called neurons. Each layer is arranged sequentially and the units in each layer are arranged vertically. Each neuron (or unit) takes some input and produces an output. Consequently, a layer takes some inputs and produces some outputs using simple computations. When we say each layer takes inputs, we mean that each neuron takes inputs. In this section, I will discuss various notations used in deep neural networks. Most of the notations are adapted from Andrew Ng’s deep learning specialisation course. The notations involve multiple dimensions that can be confusing sometimes (actually all the time when you start learning from different sources). When I started learning DNN, I got familiar with one set of notations from Andrew’s course and started representing other materials using the same notations (it was not a trivial task at the beginning).


## Inputs and outputs of a single neuron
Each neuron takes arbitrary $n_X$ dimensional input $\mathbf X$ and produces a single output $a$. 

```{figure} single-neuron.png
---
name: single_neuron
width: 200px
align: center
---
A single neural processing unit.
```

The output of the neuron is computed in two steps. For an instance $\mathbf X$, first a weighted sum is computed as: 

$$z = x_0*w_0 + x_1*w_1 + \cdots + x_n*w_{n_X} + b = \sum_{i=0}^{n_X}x_i*w_i +b$$

and then an activation function $\sigma$ is applied to $z$ to compute the output (called activation) as:

$$a = \sigma\left(z\right)$$

If we organise the weights as a row vector $\mathbf W = [w_0, w_1, \cdots, w_{n_X}]$  and our input as a row vector $\mathbf X =[x_1, x_2,\dots,x_{n_X}]$, we can achive the same result using vector operation:

$$a = \sigma(z) = \sigma(\mathbf W \cdot \mathbf X^\top+b).$$

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
* - $x_j$	
  - $1$	
  - The value of the $j$-th feature of input object.
* - $\mathbf W$	
  - $(1, n_X)$	
  - Weight vector of the neuron.
* - $b$
  - $1$	
  - Bias parameter of the neuron.
* - $z$
  - $1$	
  - The intermediate output of the neuron given the input $\mathbf X$ before applying the activation function.
* - $a$
  - $1$	
  - The output (or activation) of the neuron given $\mathbf X$ after applying the activation function $\sigma$ to $z$.
* - $\mathbf X^\top$
  - $(n_X, 1)$	
  - $\mathbf X$ transposed.
```


## Inputs and outputs of a single layer
A layer in a neural network is created by stacking multiple neural units. Each unit takes the same number of inputs ($n_X$) and produces a single output. The number of outputs of a layer is determined by the number of units in the layer. Representing a layer by $l$, the number of units in the layer is expressed as $n^{[l]}$. 

```{figure} single_layer.png
---
name: single_layer
width: 200px
align: center
---
A single layer processing unit.
```

Now that we have multiple units, we cannot express the weights as a vector anymore. The weights of each individual unit in the layer are still a vector (shape $(1, n_X)$) and we need a matrix (by stacking the vectors) to express the weights of layer $l$. For $n^{[l]}$ units we will have  $n^{[l]}$ biases expressed as $\mathbf b^{[l]}$ which is a vector of shape $(n^{[l]}, 1)$.  We have multiple outputs from a layer, hence both $a$ and $z$ are now vectors or shape $(n^{[l]}, 1)$ and represented by: $\mathbf z^{[l]}$ and $\mathbf a^{[l]}$. Layer $l$ performs the following operations:

$$a^{[l]}_0 = \sigma^{[l]} (z^{[l]}_0) = \sigma^{[l]} \left(w^{[l]}_{00}x_0 + w^{[l]}_{01}x_1+\dots+w^{[l]}_{0n_X}x_{n_X}+b^{[l]}_0\right) = \sigma^{[l]} \left(\sum_i^{n_X} w^{[l]}_{0i}x^i + b^{[l]}_0\right)$$

$$a^{[l]}_1 = \sigma^{[l]} (z^{[l]}_1) = \sigma^{[l]} \left(w^{[l]}_{10}x_0 + w^{[l]}_{11}x_1+\dots+w^{[l]}_{1n_X}x_{n_X}+b^{[l]}_1\right)=\sigma^{[l]}\left(\sum_{i}^{n_X} w^{[l]}_{1i}x_i + b^{[l]}_1\right)$$

$$\cdots$$

$$a^{[l]}_{n^{[l]}} = \sigma^{[l]} (z^{[l]}_{n^{[l]}}) = \sigma^{[l]} \left(w^{[l]}_{n^{[l]}0}x_0 + w^{[l]}_{n^{[l]}1}x_1+\dots+w^{[l]}_{n^{[l]}n_X}x_{n_X}+b^{[l]}_{n^{[l]}}\right)=\sigma^{[l]}\left(\sum_{i}^{n_X} w^{[l]}_{n^{[l]}i}x_i + b^{[l]}_{n^{[l]}}\right)$$


```python
output = []
for i in range(nl):
	z = []
	for j in range(n):
		z += W[i][j] * x[j]
	z += b[i]
	a = sigma(z)
	
	output.append(a)
```

The weights of the matrix can be represented by:

$$\mathbf W^{[l]} = \begin{bmatrix} w^{[l]}_{00} & w^{[l]}_{01} & \cdots & w^{[l]}_{0n_X}\\ w^{[l]}_{10} & w^{[l]}_{11} & \cdots & w^{[l]}_{1n_X}\\\cdots&\cdots&\cdots&\cdots\\w^{[l]}_{n^{[l]}0} & w^{[l]}_{n^{[l]}1} & \cdots & w^{[l]}_{n^{[l]}n_X}\end{bmatrix}$$

The dimension of $\mathbf W^{[l]}$ depends on the input dimension $n_X$ and the number of units (which is same as the output dimension of the layer) $n^{[l]}$ i.e., $(n^{[l]}, n_X)$. Using the matrix notations, the computations are:

$$\mathbf a^{[l]} = \sigma^{[l]} (\mathbf z^{[l]}) = \sigma^{[l]}\left(\mathbf W^{[l]} \cdot \mathbf X^\top + \mathbf b^{[l]}\right)$$

```{list-table} Notations - single layer
:header-rows: 1
:name: notation-single-layer

* - Symbol
  - Shape
  - Definition
* - $l$	
  - $1$	
  - Layer id. 
* - $n^{[l]}$	
  - $1$	
  - Number of units in layer $l$.
* - $\mathbf b^{[l]}$	
  - $(n^{[l]}, 1)$	
  - The bias vector corresponding to layer $l$.
* - $b^{[l]}$
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
  - The intermediate outputs of the neurons given the input $\mathbf X$ before applying the activation function.
* - $a_i^{[l]}, z_i^{[l]}$	
  - $1$	
  - Elements of $\mathbf a^{[l]}$ and $\mathbf z^{[l]}$.
* - $\sigma^{[l]}$
  - $—$	
  - Activation function corresponding to layer $l$.
```
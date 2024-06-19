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
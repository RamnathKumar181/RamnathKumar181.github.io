---
layout: post
title: How to represent part-whole hierarchies in a neural network
published: true
---

An overview of the paper “[Meta-Learning: Learning to Learn Fast](https://arxiv.org/pdf/2102.12627.pdf)”.
<!--break-->
A good meta-learning model should be capable of well adapting or generalizing to new tasks and environments that have never been encountered during training time. This is why meta-learning is known as "learning to learn". All images and tables in this post are from their respective paper.

## Defining the Meta-Learning Model

### A simple view

A good meta-learning model should be trained over a variety of learning tasks and optimized for the best performance on a distribution of tasks, potentially unseen tasks. The optimal model parameters are:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\theta&space;^{*}&space;=&space;\arg&space;\min_{\theta}&space;\mathbb{E}_{D\sim&space;p(D)}[L_{\theta}(D)]" title="\theta ^{*} = \arg \min_{\theta} \mathbb{E}_{D\sim p(D)}[L_{\theta}(D)]" />
</p>
This is very similar to a normal learning task, but one dataset is considered as one data sample.
<b>Few-shot classification</b> is an instantiation of meta-learning in the field of supervised learning. The dataset <img src="https://latex.codecogs.com/svg.latex?D" title="D" /> is often divided into two parts, a support set <img src="https://latex.codecogs.com/svg.latex?S" title="S" /> for learning and a prediction set <img src="https://latex.codecogs.com/svg.latex?B" title="B" /> for training or testing, <img src="https://latex.codecogs.com/svg.latex?D&space;=&space;<S,B>" title="D = <S,B>" />. Often we consider a K-shot N-class classification task: the support set contains <img src="https://latex.codecogs.com/svg.latex?K" title="K" /> labelled examples for each of <img src="https://latex.codecogs.com/svg.latex?N" title="N" /> classes.

<p align="center">
<b>An example for 2-way 4-shot image classification.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

### Training in the same way as testing

A dataset <img src="https://latex.codecogs.com/svg.latex?D" title="D" /> contains pairs of feature vectors and labels, where each labels belong to a known label set <img src="https://latex.codecogs.com/svg.latex?l^{label}" title="l^{label}" />. Our classifer <img src="https://latex.codecogs.com/svg.latex?f_{\theta}" title="f_{\theta}" />, like any other ckassifier with parameter <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> outputs a probability of a data point belonging to the class <img src="https://latex.codecogs.com/svg.latex?y" title="y" /> given the feature vector <img src="https://latex.codecogs.com/svg.latex?x" title="x" />, <img src="https://latex.codecogs.com/svg.latex?P_{\theta}(y|x)" title="P_{\theta}(y|x)" />. The optimal parameters maximize the probability of true labels across multiple training batches.

In few-shot classification, the goal is to reduce the prediction error on data samples with unknown labels given a small support set for "fast learning" (similar to fine-tuning). To make the training process mimic what happens during inference, we would like to "fake" datasets with a subset of labels to avoid exposing all lables and modify the optimization procedure such as:
* Sample a subset of labels <img src="https://latex.codecogs.com/svg.latex?L&space;\subset&space;L^{labels}" title="L \subset L^{labels}" />.
* Sample a support set <img src="https://latex.codecogs.com/svg.latex?S^{L}&space;\subset&space;D" title="S^{L} \subset D" /> and a training batch <img src="https://latex.codecogs.com/svg.latex?B^{L}&space;\subset&space;D" title="B^{L} \subset D" />. Both of them only contain data points with labels belonging to the sampled label set <img src="https://latex.codecogs.com/svg.latex?L" title="L" />.
* The support set is part of the model input
* The final optimization uses the mini-batch to compute the loss and update the model parameters.

Note, that the new optimal parameters is computed using:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\theta&space;^*&space;=&space;\arg&space;\max&space;_{\theta}&space;E_{L&space;\subset&space;L}[E_{S^L&space;\subset&space;D,&space;B^L&space;\subset&space;D}[\sum&space;_{x,y&space;\in&space;B^L}P_{\theta}(x,y,S^L)]]" title="\theta ^* = \arg \max _{\theta} E_{L \subset L}[E_{S^L \subset D, B^L \subset D}[\sum _{x,y \in B^L}P_{\theta}(x,y,S^L)]]" />
</p>

### Learning as Meta-Learner

A popular view of meta-learning decomposes the model update into two stages:
* A classifier <img src="https://latex.codecogs.com/svg.latex?f_{\theta}" title="f_{\theta}" /> is the learner model trained for operating a given task.
* In the meantime, a optimizer <img src="https://latex.codecogs.com/svg.latex?g_{\phi&space;}" title="g_{\phi }" /> learns how to update the learner model's parameters via the support set <img src="https://latex.codecogs.com/svg.latex?S" title="S" />.

### Common Approaches

There are three common approaches to meta-learning: metric-based, model-based and optimization-based.

## Metric-Based

The core idea in metric based meta-learning is similar to nearest neighbors algorithm and kernel density estimation. The predicted probability over a set of known labels <img src="https://latex.codecogs.com/svg.latex?y" title="y" /> is a weighted sum of labels of support set samples. The weight is generated by a kernel function <img src="https://latex.codecogs.com/svg.latex?k_{\theta}" title="k_{\theta}" />, measuring the similarity between two data samples.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P_{\theta}(y|x,S)&space;=&space;\sum&space;_{(x_i,y_i)\in&space;S}&space;k_{\theta}(x,x_i)y_i" title="P_{\theta}(y|x,S) = \sum _{(x_i,y_i)\in S} k_{\theta}(x,x_i)y_i" />
</p>
All the models introduced introduced below learn embedding vectors of input data explicitly and use them to design proper kernel functions.

### Convolutional Siamese Neural Network

The siamese neural network is composed of two twin networks and their outputs are jointly trained on top with a function to learn the relationship between pairs of input data samples. The twin networks are identical, sharing the same weights and network parameters. In other words, both refer to the same embedding network that learns an efficient embedding to reveal relationship between pairs of data points.
[Koch et al.](http://www.cs.toronto.edu/~rsalakhu/papers/oneshot1.pdf) proposed a method to use the siamese neural network to do one-shot image classification. First, the siamese network is trained for a verification task for twlling whether two input images are in the same class. It outputs the probability of two images belonging to the same class. Then, during test time, the siamese network processes all the image pairs between a test image and every image in the support set. The final prediction is the class of the support image with the highest probability.

<p align="center">
<b>The architecute of convolutional siamese neural network for few-shot image classificaation.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-2.png?raw=true" alt="Figure 2"/>
</p>

* First, convolutional siamese network learns to encode two images into feature vectors via a embedding function <img src="https://latex.codecogs.com/svg.latex?f_{\theta}" title="f_{\theta}" /> which contains a couple of convolutional layers.
* The L1-distance between two embeddings is <img src="https://latex.codecogs.com/svg.latex?\begin{vmatrix}&space;f_{\theta}(x_i)-f_{\theta}(x_j)&space;\end{vmatrix}" title="\begin{vmatrix} f_{\theta}(x_i)-f_{\theta}(x_j) \end{vmatrix}" />.
* The distance is converted into a probability <img src="https://latex.codecogs.com/svg.latex?p" title="p" /> by a linear feedforward layer and sigmoid. It is the probability of whether two images are drawn from the same class.
* Intuitively the loss is cross entropy because the label is binary.

The assumption is that the learned embedding can be generalized to be useful for measuring the distance between images of unknown categories. This is the same assumption behind transfer learning via the adoption of a pre-trained model; for example, the convolutional features learned in the model pre-trained with ImageNet are expected to help other image tasks. However, the benefit of a pre-trained model decreases when the new task diverges from the original task that the model was trained on.

### Matching Networks

The task of matching networks proposed by [Vinyals et al.](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) is to learn to classify <img src="https://latex.codecogs.com/svg.latex?c_S" title="c_S" /> for any given support set <img src="https://latex.codecogs.com/svg.latex?S" title="S" />. This classifier defines a probability distribution over output labels <img src="https://latex.codecogs.com/svg.latex?y" title="y" /> given a test example <img src="https://latex.codecogs.com/svg.latex?x" title="x" />. Similar to other metric-based models, the classifier output is defined as a sum of labels of support samples weighted by attention kernel <img src="https://latex.codecogs.com/svg.latex?a(x,x_i)" title="a(x,x_i)" /> - which should be proportional to the similarity between <img src="https://latex.codecogs.com/svg.latex?x" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?x_i" title="x_i" />.

<p align="center">
<b>The architecute of Matching Networks.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-3.png?raw=true" alt="Figure 3"/>
</p>
The attention kernel depends on two embedding functions, <img src="https://latex.codecogs.com/svg.latex?f" title="f" /> and <img src="https://latex.codecogs.com/svg.latex?g" title="g" />, for encoding the test sample and the support set samples respectively. The attention weight between two data points is the cosine similarity between their embedding vectors, normalized by softmax:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?a(x,x_i)&space;=&space;\frac{\exp(cosine(f(x),g(x_i)))}{\sum&space;_{j=1}^k&space;\exp(cosine(f(x),g(x_j)))}" title="a(x,x_i) = \frac{\exp(cosine(f(x),g(x_i)))}{\sum _{j=1}^k \exp(cosine(f(x),g(x_j)))}" />
</p>

In the <b>Simple embedding</b> version, an embedding function is a neural network with a single data sample as input. Potentially, we can set <img src="https://latex.codecogs.com/svg.latex?f=g" title="f=g" />.

However, the embedding vectors are critical inputs for building a good classifier and taking a single point might not be efficient. Hence, we create <b>full context embeddings</b>. The matching network model further proposed to enhance the embedding functions by taking as input the whole support set <img src="https://latex.codecogs.com/svg.latex?S" title="S" /> in addition to the original input, so that the learned embedding can be adjusted based on the relationship with other support samples.
* <img src="https://latex.codecogs.com/svg.latex?g_{\theta}(x_i,S)" title="g_{\theta}(x_i,S)" /> uses a bidirectional LSTM to encode <img src="https://latex.codecogs.com/svg.latex?x_i" title="x_i" /> in the context of the entire support set <img src="https://latex.codecogs.com/svg.latex?S" title="S" />.
* <img src="https://latex.codecogs.com/svg.latex?f_{\theta}(x,S)" title="f_{\theta}(x,S)" /> encodes the test sample <img src="https://latex.codecogs.com/svg.latex?x" title="x" /> via an LSTM with read attention over the support set <img src="https://latex.codecogs.com/svg.latex?S" title="S" />. First the test sample goes through a simple neural network to extract basic features. Then an LSTM is trained with a read attention vector over the support set as part of the hidden state.
This embedding method does help performance on a hard task (few-shot classification on mini Imagenet) but makes no difference on a simple task (Omniglot).

### Relation Network

Relation Network was proposed by [Sung et al.](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Sung_Learning_to_Compare_CVPR_2018_paper.pdf), and is similar to the siamese network with a few differences:
* The relationship is not captured by a simple L1 distance in the feature space, but predicted by a CNN classifier <img src="https://latex.codecogs.com/svg.latex?g_{\phi}" title="g_{\phi}" />. The relation score between a pair of inputs <img src="https://latex.codecogs.com/svg.latex?x_i" title="x_i" /> and <img src="https://latex.codecogs.com/svg.latex?x_j" title="x_j" />, is <img src="https://latex.codecogs.com/svg.latex?r_{ij}&space;=&space;g_{\phi}([x_i,x_j])" title="r_{ij} = g_{\phi}([x_i,x_j])" /> where <img src="https://latex.codecogs.com/svg.latex?[.,.]" title="[.,.]" /> is concatenation.
* The objective function is MSE loss, vecause conceptually RN focuses more on predicting relation scores which is more like regression, rather than binary classification.
<p align="center">
<b>Relation Network architecture for a 5-way 1-shot problem with one query example.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-4.png?raw=true" alt="Figure 4"/>
</p>

### Prototypical Networks

Prototypical networks proposed by [Snell et al.](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf) uses an embedding function <img src="https://latex.codecogs.com/svg.latex?f_{\theta}" title="f_{\theta}" /> to encode each input into a <img src="https://latex.codecogs.com/svg.latex?M" title="M" />-dimensional feature vector. A prototype feature vector is define for every class <img src="https://latex.codecogs.com/svg.latex?c" title="c" /> as the mean vector of embedded support data samples in this class.
The prediction is made by computing the softmax of distances between these prototypical mean embeddings and query set or test set.

<p align="center">
<b>Prototypical networks in the few-shot and zero-shot scenarios.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-5.png?raw=true" alt="Figure 5"/>
</p>

## Model-Based

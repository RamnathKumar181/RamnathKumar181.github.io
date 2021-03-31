---
layout: post
title: Meta-Learning-Learning to Learn Fast
published: true
---

An overview of the topic “[Meta-Learning: Learning to Learn Fast](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)”.
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
Model-based met-learning models make no assumption on the form of <img src="https://latex.codecogs.com/svg.latex?p_{\theta}(y|x)" title="p_{\theta}(y|x)" />. Rather, it depends on a model designed specifically for fast learning. This rapid parameter update can be achieved by its internal architecture or controlled by another meta-learner model.

### Memory-Augmented Neural Networks

A family of model architectures use external memory storage to facilitate the learning process of neural networks, including Neural Turing Machines and Memory Networks. With an explicit storage buffer, it is easier for the network to rapidly incorporating new information and not to forget in the future. Such a model is known as <b>MANN</b>.
Because MANN is expected to encode new information fast and thus to adapt to new tasks after only a few samples, it fits well for meta-learning. Neural Turning Machines couples a controller neural network with external memory storage. The controller learns to read and write memory rows by soft attention, while the memory serves as a knowledge repository. The attention weights are generated by its addressing mechanism: content-based+location based.
<p align="center">
<b>The architecture of Neural Turning Machine.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-6.png?raw=true" alt="Figure 6"/>
</p>

To use MANN for meta-learning tasks, we need to train it in a way that memory can encode and capture information of new tasks fast and, in the meantime, any stored representation is easily accessible. In each training episode, the truth label <img src="https://latex.codecogs.com/svg.latex?y_t" title="y_t" /> is presented with one step offset <img src="https://latex.codecogs.com/svg.latex?(x_{t&plus;1},y_t)" title="(x_{t+1},y_t)" />: it is the true lable for the input at the previous time step <img src="https://latex.codecogs.com/svg.latex?t" title="t" />, but presented as part of the input at time step <img src="https://latex.codecogs.com/svg.latex?t&plus;1" title="t+1" />

<p align="center">
<b>Task setup in MANN for meta-learning.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-7.png?raw=true" alt="Figure 7"/>
</p>
In this way, MANN is motivated to memorize the information of a new dataset, because the memory has to hold the current input until the label is present later, and then retrieve the old information to make a prediction accordingly.

Aside from the training process, a new pure content-based addressing mechanism is utilized to make the model better suitable for meta-learning. The read attention is constructed purely based on the content similarity. First, a key feature vector <img src="https://latex.codecogs.com/svg.latex?k_t" title="k_t" /> is produced at the time step <img src="https://latex.codecogs.com/svg.latex?t" title="t" /> by the controller as a function of the input <img src="https://latex.codecogs.com/svg.latex?x" title="x" />. Similar to the NTM, a read weighting vector <img src="https://latex.codecogs.com/svg.latex?w_t^r" title="w_t^r" /> of <img src="https://latex.codecogs.com/svg.latex?N" title="N" /> elements is computed as the cosine similarity between the key vector and every memory vector row, normalized by softmax. The read vector <img src="https://latex.codecogs.com/svg.latex?r_t" title="r_t" /> is a sum of memory records weighted by such weightings.The addressing mechanism for writing newly received information into memory operates a lot like cache replacement policy. The Least Recently Used Access writed is designed for MANN to better work in the scenario of meta-learning.

### Meta Networks

Meta Networks proposed by [Mukhdalai et al.](https://arxiv.org/abs/1703.00837) is short of MetaNet, is a meta-learning model with architecture and training process for rapid generalization across tasks. The rapid generalization of MetaNet relies on "fast weights". Normally, weights in the neural networks are updated by SGD in an objective function and this process is known to be slow. One faster way to learn weights are called fast weights. In MetaNet, loss gradients are used as meta information to populate models that learn fast weights. Slow and fast weights are combined to make predictions in neural networks.
<p align="center">
<b>Combining slow and fast weights in a MLP.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-8.png?raw=true" alt="Figure 8"/>
</p>

Key components of MetaNet are:
* An embedding function <img src="https://latex.codecogs.com/svg.latex?f_{\theta}" title="f_{\theta}" />, parametrized by <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />, encodes raw inputs into feature vectors. Similar to Siamese Networks, these embeddings are trained to be useful for telling whether two inputs are of the same class.
* A base learner model <img src="https://latex.codecogs.com/svg.latex?g_{\phi}" title="g_{\phi}" /> parameterized by weights <img src="https://latex.codecogs.com/svg.latex?\phi" title="\phi" />, completes the actual learning task.
To get fast weights, we need to create two functions <img src="https://latex.codecogs.com/svg.latex?f" title="f" /> and <img src="https://latex.codecogs.com/svg.latex?g" title="g" /> respectively:
* <img src="https://latex.codecogs.com/svg.latex?F_w" title="F_w" />: a LSTM parameterized by <img src="https://latex.codecogs.com/svg.latex?w" title="w" /> for learning fast weights of the embedding function <img src="https://latex.codecogs.com/svg.latex?f" title="f" />. It takes as input gradients of <img src="https://latex.codecogs.com/svg.latex?f" title="f" />'s embedding loss for verification task.
* <img src="https://latex.codecogs.com/svg.latex?G_v" title="G_v" />: a neural network parameterized by <img src="https://latex.codecogs.com/svg.latex?v" title="v" /> learning fast weights for the base learner <img src="https://latex.codecogs.com/svg.latex?g" title="g" /> from its loss gradients. In MetaNet, the learner's loss gradients are viewed as meta information of the task.

<p align="center">
<b>The MetaNet architecture.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-9.png?raw=true" alt="Figure 9"/>
</p>

## Optimization-Based
Deep learning models learn through backpropagation of gradients. However, the gradient-based optimization is neither designed to cope with a small number of training samples, nor to coverage within a small number of optimization steps.

### LSTM Meta-Learner

The optimization algorithm can be explicitly modeled. [Ravi et al.](https://openreview.net/pdf?id=rJY0-Kcll) did so and named it "meta-learner". The goal of the meta-learner is to efficiently update the learner's parameters using a small support set so that the learner can  adapt to the new tasks quickly.
The meta-learner is modeled as a LSTM for two reasons:
* There is a similarity between the gradient based update in backpropagation and the cell-state update in LSTM.
* Knowing a history of gradients benefits the gradient updata.

The update for the learner's parameters at time step <img src="https://latex.codecogs.com/svg.latex?t" title="t" /> with a learning rate <img src="https://latex.codecogs.com/svg.latex?\alpha_t" title="\alpha_t" /> is:<img src="https://latex.codecogs.com/svg.latex?\theta_t&space;=&space;\theta_{t-1}-\alpha&space;\bigtriangledown_{\theta-1}L_t" title="\theta_t = \theta_{t-1}-\alpha \bigtriangledown_{\theta-1}L_t" />. It has the same form as the cell state update in LSTM if we set forget gate <img src="https://latex.codecogs.com/svg.latex?f_t&space;=&space;1" title="f_t = 1" />, input gate <img src="https://latex.codecogs.com/svg.latex?i_t&space;=&space;\alpha_t" title="i_t = \alpha_t" />, cell state <img src="https://latex.codecogs.com/svg.latex?c_t&space;=&space;\theta_t" title="c_t = \theta_t" /> and new cell state <img src="https://latex.codecogs.com/svg.latex?\widetilde{c_t}&space;=&space;-\triangledown_{\theta-1}L_{t}" title="\widetilde{c_t} = -\triangledown_{\theta-1}L_{t}" />:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?c_t&space;=&space;f_t\odot&space;c_{t-1}&space;&plus;&space;i_t\odot&space;\widetilde{c_t}" title="c_t = f_t\odot c_{t-1} + i_t\odot \widetilde{c_t}" />
</p>
While fixing <img src="https://latex.codecogs.com/svg.latex?f_t&space;=&space;1" title="f_t = 1" /> and <img src="https://latex.codecogs.com/svg.latex?i_t&space;=&space;\alpha_t" title="i_t = \alpha_t" /> might not be optimal, both of them can be learnable and adaptable to different datasets. <img src="https://latex.codecogs.com/svg.latex?f_t" title="f_t" /> indicates how much to forget the old value of parameters. Whereas, <img src="https://latex.codecogs.com/svg.latex?i_t" title="i_t" /> is used as learning rate at time step <img src="https://latex.codecogs.com/svg.latex?t" title="t" />.

<p align="center">
<b>Model setup.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-10.png?raw=true" alt="Figure 10"/>
</p>
The training process mimics what happens during test. During each training epoch, we first sample a dataset and then sample mini-batches out of train set to update <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> for <img src="https://latex.codecogs.com/svg.latex?T" title="T" /> rounds. The final state of the learner parameter <img src="https://latex.codecogs.com/svg.latex?\theta_T" title="\theta_T" /> is used to train the meta-learner on the test data.

<p align="center">
<b>Algorithm for meta-learner.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-11.png?raw=true" alt="Figure 11"/>
</p>

### MAML

MAML is a fairly general optimization algorithm, compatible with any model that learns through gradient descent.

<p align="center">
<b>The general MAML algorithm.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-12.png?raw=true" alt="Figure 12"/>
</p>

The meta-optimization step relies on second derivatives. To make computation less expensive, a modified version of MAML omits second derivatives resulting in a simplified and cheaper implementation, known as First-Order MAML.

### Reptile

Reptile proposed by [Nichol et al.](https://arxiv.org/abs/1803.02999) is a simple meta-learning optimization algorithm which works by repeatedly:
* Sampling a task
* Training on it by multiple gradient descent steps
* Moving the model weights towards new parameters.

<p align="center">
<b>Batched version of Reptile Algorithm.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-12.png?raw=true" alt="Figure 12"/>
</p>

To find a solution that is good across tasks, we would like to find a parameter close to all the optimal manifolds of all tasks.

<p align="center">
<b>The Reptile algorithm updates the parameter alternatively to be closer to the optimal manifolds of different tasks.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/19/Figure-13.png?raw=true" alt="Figure 13"/>
</p>

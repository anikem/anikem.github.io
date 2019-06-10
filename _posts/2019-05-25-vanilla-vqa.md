---
title: Vanilla VQA
excerpt: "An introduction to visual question answering"
header:
  overlay_color: "#1d1d1d"
  image: /assets/images/vanilla-vqa-multiple-choice.png
  og_image: /assets/images/vanilla-vqa-multiple-choice.png
classes: wide
---

Visual Question Answering (VQA) is the task of answering questions about a given piece of visual content such as an image, video or infographic.
As seen in the examples below, answering questions about visual content requires a variety of skills include recognizing entities and objects, reasoning about their interactions with each other, both spatially and temporally, reading text, parsing audio, interpreting abstract and graphical illustrations as well as using external knowledge not directly present in the given content.

![](/assets/images/vanilla-vqa-intro.png)

While seemingly an easy task for humans, VQA affords several challenges to AI systems spanning the fields of natural language processing, computer vision, audio processing, knowledge representation and reasoning. Over the past few years, the advent of deep learning, availability of large datasets to train VQA models as well as the hosting of a number of benchmarking contests have contributed to a surge of interest in VQA amongst researchers in the above disciplines.

# The VQA-v1 dataset and metric

One of the early and popular datasets for this task was the [VQA-v1](http://arxiv.org/pdf/1505.00468.pdf) dataset. The VQA-v1 dataset is a very large dataset consisting of two types of images: natural images (referred to as _real images_) as well as synthetic images (referred to as _abstract scenes_) and comes in two answering modalities: _multiple choice_ question answering (the task of selecting the right answer amongst a set of choices) as well as _open ended_ question answering (the task of generating an answer with an open ended vocabulary). Owing to its difficulty and real world applicability, _open ended_ question answering about natural image content has become the most popular VQA task amongst the four dataset flavors.

The _real images_ fraction of the VQA-v1 dataset consists of over 200,000 natural images sourced from the MS-COCO dataset, a large scale dataset of images used to benchmark tasks such as object detection, segmentation and image captioning. Each image is paired with 3 questions written down by crowdsourced annotators. The dataset contains a variety of question types such as: _What color_, _What kind_, _Why_, _How many_, _Is the_, etc. To account for potential disagreements between humans for some questions, as well as account for crowd sourcing noise, each question is accompanied by 10 answers. Most answers are short and to the point, such as _yes_, _no_, _red_, _dog_ and _coca cola_ with close to 99% of the answer containing 3 or fewer words.

Given an image and a question, the goal of a VQA system is to produce an answer that matches those provided by human annotators. For the _open ended_ answering modality, the evaluation metric used is:

$$
\textrm{accuracy} = \textrm{min}\left ( \frac{\textrm{number of annotator answers that match the generated answer}}{3} , 1\right )
$$

The intuition behind this metric is as follows. If a system generated answer matches one produced by at least 3 unique annotators, it gets a maximum score of 1 on account of producing a popular answer. If it generates an answer that isn't present amongst the 10 candidates, it gets a score of 0, and it is assigned a fractional score if it produces an answer that is deemed rare. If the denominator 3 is lowered, wrong and noisy answers in the dataset (often present due to annotation noise) will receive a high credit. Conversely, if it is raised towards 10, a system producing the right answer may only receive partial credit, if the answer choices consist of synonyms or happen to contain a few noisy answers.

Learn more about the VQA-v1 dataset here:

{% include paper_post.html
  title="VQA: Visual Question Answering"
  url="http://arxiv.org/pdf/1505.00468.pdf"
  authors="Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, Devi Parikh"
  venue="ICCV 2015"
%}

# When open ended isn't really that open ended

Generating answers for open ended questions requires training a model that inputs an image and corresponding question and outputs a sequence of words using a decoder. However, training decoders is often more cumbersome than training a system that simply picks amongst a pool of K possible answers. As it turns out, most answers in datasets such as VQA-v1 tend to cluster into a manageable set of a few thousand answers. For instance, the 1000 most common answers in VQA-v1 cover roughly 83% of all answers in the dataset. This allows us to reformulate the open ended question answering problem into a K-way multiple choice problem where K is a large number such as 1000 or 2000 that covers most answers in the dataset. This is akin to a multiple choice answering paradigm where the same K choices are presented for every question. As K increases, the number of answer choices that the model has to choose from increases. This allows the model to potentially answer a higher fraction of questions correctly, but also usually requires a larger model and more training data. Most VQA researchers work with the top 1000 answer classes.

![](/assets/images/vanilla-vqa-open-ended.png)

# A baseline VQA model

The image below shows the architecture of a simple VQA neural network. The image is fed into a convolutional neural network (CNN) such as ResNet-18 which outputs a feature vector encoding the contents of the image and is referred to as an _image embedding_. The question is featurized by computing an average of the word2vec vectors over all the words in the question, resulting in a _question embedding_. These embedding vectors, which compactly represent the image and question contents have different dimensions. Hence they are first projected into the same number of dimensions using corresponding fully connected layers (a linear transformation) and then combined using pointwise multiplication (multiplying values at corresponding dimensions). The final stage of the VQA model is a multilayer perceptron with a softmax non-linearity at the end that outputs a score distribution over each of the top K answers. Converting the open ended question answering to a K-way classification task allows us to train the VQA model using a cross entropy loss between the generated answer distribution and the ground truth.

![](/assets/images/vanilla-vqa-baseline.png)

The image backbone is initialized with weights obtained from a network such as ResNet-18, trained on the ImageNet classification dataset. This initialization provides several advantages: _First_, it leads to faster training, since training a large image CNN from scratch is usually quite expensive. _Second_, this allows the VQA model to exploit knowledge obtained from the ImageNet dataset, which may be very useful to answer questions about objects in images. _Third_, if the image CNN weights are not fine-tuned on the VQA dataset and instead frozen to the ImageNet ones, the image representations for the entire training dataset can be pre-computed and stored on disk, which results in less memory consumption while training the VQA model.

# Many simple variants of the baseline model

One can easily tweak this simple VQA architecture in several ways to arrive at a bunch of different VQA models as shown in the image below.

_Image Embedding_ : The image can be passed through a variety of image embedding neural networks (each pre-initilaized with the corresponding ImageNet variant). Typically, VQA task accuracy correlates well with the accuracy obtained by the network architecture on the ImageNet classification task (for example, a VQA model with ResNet-50 does better than a model with AlexNet).

This suggests that (a) Network architectures that perform well at image classification also perform well at VQA, and (b) The knowledge obtained from ImageNet and encoded within the parameters of the image embedding network aids the task of VQA, and a better knowledge representation leads to improved metrics on VQA.

_Text Embedding_ : word2vec embeddings encode each word into a fixed dimensional vector which captures the syntax and semantics of that word. One can easily substitute this embedding by any other word embedding such as Glove. Word embeddings that work better on a variety of NLP tasks, typically work better for the VQA task.

Another simple variant, which is the de facto standard today, is to use a recurrent neural network such as an LSTM to encode the sequence of word embeddings, instead of simply averaging the word vectors. In contrast to averaging, using an LSTM preserves information regarding the order of the words in the question and leads to an improved VQA accuracy.

_Combining the word vectors_ : There are several easy and cheap ways of combining representations from the image and text modality: Pointwise operations as well as concatenation. Concatenation leads to an increase in the number of parameters for the ensuing network. There isn't a clear winner amongst these simple variants, although pointwise multiplication tends to be the most common option across research papers.

![](/assets/images/vanilla-vqa-baseline-variants.png)

# Multiple choice VQA systems

When the answer choices are known a-priori (such as in multiple choice settings), the above VQA network can be easily extended to also consume the embedding of a single answer choice. Instead of predicting an answer choice (as was done above), the network now predicts if the provided answer choice is True or False. A simple yet important modification involves replacing the final softmax function with a sigmoid function and predicting a single score. If M answer choices are provided in the multiple choice scenario, the network needs to be run M times, once for each answer choice and the final prediction involves choosing the answer choice that provided the highest score.

![](/assets/images/vanilla-vqa-multiple-choice.png)

When the answer choices are fed into the network, the resulting models obtain higher VQA accuracies. The disadvantage is that the network needs to be M times. Researchers have also used this variant in the open ended setting by running the model through K times, once each for the top K answers. This is slow, but it also provided improved performance.

Learn more about simple yet powerful VQA baselines for the multiple choice and open ended settings here:

{% include paper_post.html
  title="Revisiting Visual Question Answering Baselines"
  url="https://arxiv.org/pdf/1606.08390.pdf"
  authors="Allan Jabri, Armand Joulin, Laurens van der Maaten"
  venue="ECCV 2016"
%}

# How well do these simple models perform ?

On the VQA-v1 dataset, a simple baseline similar to the one described above obtained an accuracy of 57.75 on the real images open ended question answering setting. This is a rather high accuracy when you consider that the number of answer choices in the open ended scenario is very large (the top 1000 answers cover roughly 83% of all answers).

However, this needs to be put into context by comparing the number to another simple baseline, the _question only baseline_ which does not consider the image at all. Not looking at an image while answering a question about an image seems very counter intuitive right ? However, such a question only baseline (depicted in the image below) performs reasonably well on the VQA-v1 dataset. It obtains an accuracy of 50.39 on VQA-v1. How is this possible ?

![](/assets/images/vanilla-vqa-question-only.png)

While the number of answer choices for the open ended scenario is very large, the inherent structure and regularities in the world result in just a few plausible answers given a question. For instance, the question "What color is the construction cone ?" can likely be answered by simply choosing a common color, reducing the space of answers from 1000s to roughly 10. Furthermore, construction cones are dominantly orange in the color in the world. Hence simply memorizing this fact and always answering "orange" for this question results in a reasonable VQA system that does not need to look at the image. Neural networks are very good at exploiting such data biases that can be learnt from the training set, and this is precisely what is happening with the _question only baseline_.

# Conclusion

To summarize, Visual Question Answering (VQA), the task of answering questions about visual content, has received a lot of attention recently amongst researchers. In this blog post I described a few VQA baseline models that perform reasonably well on standard benchmarks. These models have been extended in many interesting ways including the use of visual attention modules, object detection frameworks, counting modules, methods to fuse multi-modal information, etc. which I shall discuss in future blog posts. Stay tuned!

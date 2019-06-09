---
title: May I have your attention please ?
excerpt: "The attention mechanism and its numerous variants explored in Visual Question Answering"
header:
  overlay_color: "#1d1d1d"
  image: /assets/images/attention-q-i.png
  og_image: /assets/images/attention-q-i.png
classes: wide
---

[Vanilla architectures for Visual Question Answering (VQA)](/vanilla-vqa) represent the question into a single question embedding vector, represent the entire contents of the image into a single image embedding vector, combine the two and then pass this into a multilayer perceptron to produce a distribution over all the answer choices. This works reasonably well, especially for questions that require a holistic understanding of the scene.

Many questions however, require one to focus on specific parts of the image or certain words in the question or a combination of both. For example, consider the image and question below. Answering this question requires the model to focus on the tennis racquet and tennis ball. The vanilla VQA model struggles to focus on such small yet relevant regions in images, primarily because the contents of the two modalities (image and text) get compressed into single vector representations.

![](/assets/images/attention-motivation.png)

The attention mechanism is a clever architectural modification that enables neural networks to focus in onto small portions of the provided context. It was first proposed for the task of machine translation where the decoder could focus on relevant parts of the source language text as it generated the target language text. This improved the quality of machine translation beyond models that encoded the entire source text into a single embedding which was then provided to the decoder.

In the vision and language space, attention was first used for the task of image captioning and then quickly made its way to VQA models. In its early incarnations, it allowed the model to focus in onto small regions of the image, conditioned on the input question. In this post, I'll discuss some basic methods to use attention mechanisms as well as discuss some variations proposed over the past few years.

# Single Hop Attention

There are many different ways to obtain embeddings for the question and the image for the purpose of VQA. I discuss some of those variations in a [previous post](/vanilla-vqa). For the purpose of this discussion, lets assume the following embedding mechanism (as shown in the figure below).

_Question_ : The question is passed through a bi-directional LSTM, following which the final hidden states from the two directions are concatenated to form the $$d$$ dimensional question embedding. Lets name this $$v_Q \in \mathbb{R}^{d \times 1}$$.

_Image_ : The image is passed through a CNN and the output of a late-stage convolution layer is extracted which results in a tensor of size $$s \times c \times c$$. One can think of this as dividing the image into a uniform grid of size $$c \times c$$ and then representing each of those regions with an $$s$$ dimensional embedding. (Note: While this is a useful abstraction, it isn't quite accurate since convolution and pooling operations lead to information transfer across spatial extents of tensors in a CNN which is dictated by the size of their receptive fields.) In contrast to vanilla VQA architectures which represented the entire image as a single embedding, we now have a set of embeddings that represent different parts of the image. This tensor is reshaped and then projected via a linear transformation onto a $$d$$ dimensional space, so as to match the dimensionality of the question embedding. Lets name the resultant matrix $$v_I \in \mathbb{R}^{d \times M}$$, where $$M = c^2$$ and the i-th column vector referring to the i-th region as $$v_i$$.

![](/assets/images/attention-q-i.png)

The key element of the attention mechanism is the interaction between the question embedding and each of the image embeddings, that allows the network to focus on the different regions of image, conditioned on the question. Each interaction leads to a score referred to as an _attention weight_ which signifies the relevance of that image region to the question, and the resulting vector of scores, $$p_{att}$$,  is referred to as the _attention weight vector_. A softmax operation is applied to this vector which enforces that the attention weights sum to 1.

The interaction between the question and image regions can be modeled in several ways. One simple method is to compute the attention weight as a cosine similarity between the question and image embedding.

$$
p_{att} = softmax( \textbf{1} (v_I \odot v_Q) )
$$

where $$\odot$$ denotes the elementwise multiplication between each column of the matrix $$v_I$$ and the column vector $$v_Q$$ and $$\textbf{1}$$ is a row vector of all ones.

This interaction is parameter free. A straightforward extension that works better is to make this interaction parametric such as:

$$
h_{att} = W^I v_I \oplus (W^Q v_Q + b^Q) \\
p_{att} = softmax(W^h h_{att} + b^h)
$$

where $$W^I \in k \times d$$ and $$W^Q \in k \times d$$ project the embeddings into a k dimensional space, $$\oplus$$ denotes the elementwise addition between each column of the matrix and the full vector and $$W^h$$ converts the k-dimensional attention values into a single score per interaction.

The attention weight vector $$p_{att}$$ signifies the relevance of different parts of the image to the question. The set of image embeddings are linearly combined to a single embedding vector using these attention weights. The resultant embedding $$v_I^{filt}$$ can be thought of as a filtered version of the image contents that best help answer the question.

$$
v_I^{filt} = \sum_i p_{att}[i] v_i
$$

The rest of the VQA model resembles the vanilla architecture. The filtered image embedding is added to the question embedding to produce the aggregated vector $$\phi$$

$$
\phi = v_I^{filt} + v_Q
$$

which is passed to a multi layer perceptron to obtain a distribution over the answer choices.

The entire network including the attention module is trained end-to-end by minimizing the cross entropy loss between the predicted answer distribution and the ground truth. This mechanism is depicted in the figure below. This is commonly known as single hop attention since the interaction between the question and image modalities occurs a single time.

![](/assets/images/attention-single-hop.png)

Attention mechanisms have been widely adopted in many different VQA architectures and have produced good gains in accuracy. Another interesting outcome of attention is the ability to visualize the attention weights. This is done by simply visualizing the weight vector as a heatmap, with large weights reflecting more relevant and small weights reflecting less relevant portions of the image. Examples are shown below. Its interesting to see that the model chooses to _attend_ to the baseball glove to determine the sport, the sink to determine the room and the snow (essentially everywhere) to determine the season.

![](/assets/images/attention-examples.png)

# Multiple Hop Attention

While a single attention hop allows models to focus on small regions of the image, certain questions may benefit from a series of attentions, each focussing on a different part of the image. For instance, consider the question: _What is the color of the bus next to the blue car ?_. This requires first localizing the blue car, and then subsequently finding a bus next to it and determining its color. This requires the model to focus on two parts of the image (the car and the bus). Research has shown that invoking the attention mechanism multiple times in a sequential fashion can improve the models performance.

Converting a single hop attention architecture to a two hop attention is conceptually straightforward. The output of the first attention hop is the aggregated information vector $$\phi$$ that is obtained as a sum of the question embedding and the filtered image embedding. $$\phi$$ which can be thought of as an augmented question given the first attention and is used as the question embedding for the next attention hop. The image embeddings are the same ones used in the first attention hop. This procedure can be repeated several times to produce a multi hop attention architecture as shown below.

![](/assets/images/attention-multi-hop.png)

This model can also be trained end to end using the same loss function as before. As before, the attention weights at each hop can be visualized as heatmaps. In the figure below, the model attends to several parts of the image (presumably to find the bicycle and basket) and then attends to the contents of the basket in the second hop.

![](/assets/images/attention-examples-multi-hop.png)

Learn more about single and multi-hop VQA attention architectures here:

{% include paper_post.html
  title="Stacked Attention Networks for Image Question Answering"
  url="https://arxiv.org/pdf/1511.02274v2.pdf"
  authors="Zichao Yang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Smola"
  venue="CVPR 2016"
%}

Two-hop architectures have shown improvements over single hop ones, but moving to more hops hasn't shown significant gains. This is likely due to the fact that existing VQA datasets do not have many questions that require the model to hop around the image too many times and the fact that current mechanisms for adding hops aren't capable enough to yield good results. More research efforts are needed in this direction.

# Attention as a conditional feature selection mechanism

The process of attention between the question and image embeddings can be thought of as a conditional feature selection mechanism, where the set of features are the set of image region embeddings and the condition is the question embedding.

This idea can be applied to conditionally filter not just within the image regions but also within channels in the image embedding by computing the affinity matrix as before but then pooling across the spatial dimension to arrive at an attention weight for each channel of the image embedding tensor.

The standard VQA architectures use a single tensor extracted from the CNN as the image embedding, but the answer to the question may potentially lie in a different tensor in the network. For instance, if answering the question required parsing the texture of an object, this information may potentially lie within a tensor in the earlier stages of the network. To account for this, one can use the attention mechanism to select amongst image embedding tensors at different layers of the CNN. The basic idea is to compute attention weights across all tensors, sum the weights within a tensor and then softmax across the layers to arrive at the attention weight matrix.

![](/assets/images/attention-channel.png)

Learn more about similar ideas and their application to VQA and image captioning here:

{% include paper_post.html
  title="Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering"
  url="http://openaccess.thecvf.com/content_cvpr_2018/papers/Nguyen_Improved_Fusion_of_CVPR_2018_paper.pdf"
  authors="Duy-Kien Nguyen, Takayuki Okatani"
  venue="CVPR 2018"
%}

{% include paper_post.html
  title="SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networks for Image Captioning"
  url="http://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_SCA-CNN_Spatial_and_CVPR_2017_paper.pdf"
  authors="Long Chen, Hanwang Zhang, Jun Xiao, Liqiang Nie, Jian Shao, Wei Liu, Tat-Seng Chua"
  venue="CVPR 2017"
%}

# Attending to the image and question

While the above VQA models are able to attend to specific portions of the image, it may also be useful to attend to certain words or phrases in the question as the model parses the image in search of the answer.

Conceptually, attending to the words in a question is similar to attending to the regions of an image. Just as the image is converted to a set of embeddings, one per region, the question is converted to a set of embeddings, one per word; and given a conditioning vector, the attention mechanism acts as a filter over the given set of embeddings. There are several nuances to consider such as the nature of the conditioning vector, the order of the attention between the two modalities, etc. which leads to a few variations.

_Independent attention_ : One method is to attend to the words in a question in a manner similar to the attention over image regions and do this independently at each attention hop. The question attends to the image regions, and the image attends to the question words and this is repeated across hops. The resultant aggregate vectors are from the two modalities are combined and processed to get the answer.

_Independent attention with joint conditioning_ : This method is similar to the above proposal but using an integrated conditioning vector, which is simply a sum of the two modal conditioning vectors at each attention hop.

_Joint attention_ : This method involves creating an affinity matrix $$C \in T \times N$$ (T=number of image regions, N=number of words) that measures the relevance between every word in the question and every region in the image using a learnt affinity computation.

$$
C = tanh(v_Q^T W v_I)
$$

This affinity matrix is transformed to a set of image attention weights (relevance scores per image region) either non parametrically by computing a max over each row or parametrically using a small neural network. Similarly, the affinity matrix is also transformed to a set of word attention weights (relevance scores per word). The attention weight vectors are finally used to obtain filtered versions of the image and word embeddings.

![](/assets/images/attention-joint.png)

In each of the above methods, the result is a set of filtered embeddings. These vectors are combined and sent into an MLP to produce the answer distribution. Converting a model from a one side attention over image regions to a two sided attention over the question and image is a straightforward swap of the attention module and usually results in better performance.

Learn more about attending to the image region and question words here:

{% include paper_post.html
  title="Hierarchical Question-Image Co-Attention for Visual Question Answering"
  url="https://arxiv.org/pdf/1606.00061.pdf"
  authors="Jiasen Lu, Jianwei Yang, Dhruv Batra , Devi Parikh"
  venue="Neurips 2016"
%}

{% include paper_post.html
  title="Dual Attention Networks for Multimodal Reasoning and Matching"
  url="http://openaccess.thecvf.com/content_cvpr_2017/papers/Nam_Dual_Attention_Networks_CVPR_2017_paper.pdf"
  authors="Hyeonseob Nam, Jung-Woo Ha, Jeonghee Kim"
  venue="CVPR 2017"
%}

# Contextualizing image embeddings

A common mechanism to contextualize word embeddings is to pass them through a bi-directional LSTM as is done with most present day VQA models. This adds context to each word embedding from its neighbors on both sides. When the image is passed through a CNN, the receptive field of convolution and pooling operations adds a little bit of context to each image embedding from pixels in close proximity. Researchers have also experimented with more explicitly adding context to image embeddings.

An obvious way to add context to image embeddings is to pass them through a bi-directional LSTM (with 1 or more layers), in a manner very similar to word embeddings. Image embeddings obtained from a CNN represent information about regions that are uniformly spaced on a 2 dimensional grid. While passing them through a 1 dimensional recurrent model seems to break this assumption, many works (outside of VQA) have found that this works very well and does not degrade performance compared to a more complex 2-d network of processing units.

Another way to add neighboring context to image embeddings is to add self-attention layers to the image embeddings. This amounts to treating a single embedding as a conditioning vector, use it to attend to the set of image embeddings and then create an aggregated vector that is a sum of the original embedding and the filtered embedding. This self attention is repeated for each image embedding, resulting in a set of contextualized image embeddings.

These methods are depicted in the image below.

![](/assets/images/attention-image-context.png)

Learn more about the effects of contextualizing image embeddings with recurrent models here:

{% include paper_post.html
  title="Multi-level Attention Networks for Visual Question Answering"
  url="https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/Multi-level-Attention-Networks-for-Visual-Question-Answering.pdf"
  authors="Dongfei Yu, Jianlong Fu, Tao Mei, Yong Rui"
  venue="CVPR 2017"
%}

Learn more about the effects of contextualizing image embeddings with self attention, including more sophisticated mechanisms than depicted above, in this paper:

{% include paper_post.html
  title="Dynamic Fusion with Intra- and Inter-modality Attention Flow for Visual Question Answering"
  url="https://arxiv.org/pdf/1812.05252.pdf"
  authors="Peng Gao, Zhengkai Jiang, Haoxuan You, Pan Lu, Steven Hoi, Xiaogang Wang, Hongsheng Li"
  venue="CVPR 2019"
%}

Also, a systematic architecture that incorporates multiple layers self attention with cross attention is the Transformer architecture that has recently become quite popular in the vision and NLP communities. Learn more about Transformers here:

{% include paper_post.html
  title="Attention Is All You Need"
  url="https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf"
  authors="Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin"
  venue="NeurIPS 2017"
%}

# Conclusion

To summarize, the attention mechanism is a conceptually simple and effective mechanism to enable VQA models to focus on specific regions in an image and specific words in a question. Several variations have been proposed over the years, which has led to a gradual improvement in the performance of these models; and I have covered some key ones in the above discussion.

All the methods discussed above divide the image into a uniform grid of pixels and used their embeddings for attention. But recent works have created embeddings for non uniform segmentation masks obtained from an image as well as objects detected in the image, with the latter approach particularly effective and popular. I'll discuss these approaches and more fun aspects of VQA in future blog posts. Stay tuned!

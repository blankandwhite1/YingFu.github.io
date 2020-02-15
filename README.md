# How LSTM solves the vanishing/exploding problem of RNN

## 1. Introduction

RNN enables the modeling of time-dependent and sequential data tasks effectively, such as clinical prediction, machine translating , text generation, speech recognition, image description generation and etc. 

The main difference between RNN and traditional neural networks is that each time the previous output will be brought to the next hidden layer and trained together. However, RNN suffers from the problem of vanishing/exploding problems, which hinder it from learning longer data sequences.  If you are not familiar with RNN, you can click this tutorial : [Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

This blog is intended to explain the vanishing/exploding problem of RNN and how LSTM sloves these problems from a perspective of backpropagation through time. To start with,  learning how to compute neural network gradient in a completely vectorized way is essential.

## 2. Computing Neural Network Gradients

### 2.1 Vectorized Gradients

Suppose we have a function $\mathbf{f}:\mathbb{R^{n}} \rightarrow \mathbb{R^{m}}$ which maps a vector of length $n$ to a vector of length $m$: $\mathbf{f(x)}=[f_1(x_1,x_2,...x_n),f_2(x_1,x_2,...x_n),...f_m(x_1,x_2,...x_n)]$. Then its Jacobian is the following $m\times n$ matrix:
$$
\frac{\partial{\mathbf{f}}}{\partial x} = \left[
 \begin{matrix}
   \frac{\partial{f_1}}{\partial x_1}\ & \cdots & \frac{\partial{f_1}}{\partial x_n} \\
   \vdots & \vdots & \vdots \\
   \frac{\partial{f_m}}{\partial x_1} & \cdots & \frac{\partial{f_m}}{\partial x_n}
  \end{matrix}
  \right]
$$
That is ,${(\frac{\partial{f}}{\partial x})}_{ij} = \frac{\partial{f_i}}{\partial x_j}$ (which is just a standard non-vector derivative). The Jacobian matrix will be useful for us because we can apply the chain rule to a vector-valued function just by multiplying Jacobian.

### 2.2 Useful Identities

This part will introduce how to compute Jacobian of several simple functions which will be used in taking neural network gradients. It will be more detailed in original tutorials, anyone who is interested can download [Computing Neural Network Gradients](https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf).

**(1) Matrix times column vector with respect to the column vector** ($\mathbf{z = Wx}$ , what is $\frac{\partial{\mathbf{z}}}{\partial \mathbf{x}}$? )

Suppose that $\mathbf{W}\in \mathbb{R}^{n\times m}$. Then we can think $\mathbf{z}$ as a function of $\mathbf{x}$ taking an $m$-dimensional vector to an $n$-dimensional vector. So its Jacobian will be $n \times m$. According to the rule of matrix multiplication, we can easily know : 
$$
z_i = \sum_{k=1}^{m}W_{ik}x_k
$$
So an entry ${(\frac{\partial{f}}{\partial x})}_{ij} $ of the Jacobian will be
$$
{(\frac{\partial{f}}{\partial x})}_{ij} = \frac{\partial{z_i}}{\partial x_j}=\frac{\partial}{\partial x_j}\sum_{k=1}^{m}W_{ik}x_k = \sum_{k=1}^{m}\frac{\partial}{\partial{x_j}}x_k = W_{ij}
$$
So we can see that $\frac{\partial{\mathbf{z}}}{\partial \mathbf{x}} = \mathbf{W}$ 

**(2) A vector with itself**($\mathbf{z = x}$,  what is $\frac{\partial{\mathbf{z}}}{\partial \mathbf{x}}$? )

Since $z_i = x_i$. So 
$$
{(\frac{\partial{f}}{\partial x})}_{ij} = \frac{\partial{z_i}}{\partial x_j}=\frac{\partial}{\partial x_j}x_i=
\begin{cases}
    1 & \text{if $i=j$}.\\
    0 & \text{if otherwise}.
  \end{cases}
$$
So we can see that the Jacobian $\frac{\partial{\mathbf{z}}}{\partial \mathbf{x}}$ is a diagonal matrix where the entry at $(i,i)$ is 1. This is just the identity matrix. $\frac{\partial{\mathbf{z}}}{\partial \mathbf{x}} = \mathbf{I}$ 

**(3) An elementwise function applied a vector** ($\mathbf{z} = f(\mathbf{x})$ , what is $\frac{\partial{\mathbf{z}}}{\partial \mathbf{x}}$? )

Since $z_i = f(x_i)$. So 
$$
{(\frac{\partial{f}}{\partial x})}_{ij} = \frac{\partial{z_i}}{\partial x_j}=\frac{\partial}{\partial x_j}f(x_i)=
\begin{cases}
    f'(x) & \text{if $i=j$}.\\
    0 & \text{if otherwise}.
  \end{cases}
$$
So we can see that the Jacobian $\frac{\partial{\mathbf{z}}}{\partial \mathbf{x}}$ is a diagonal matrix where the entry at $(i,i)$ is the derivative of $f$ applied to $x_i$. So we can write this as $\frac{\partial{\mathbf{z}}}{\partial \mathbf{x}} = diag(f'(x))$

**(4) Matrix times column vector with respect to the matrix** ($\mathbf{z = Wx}, \delta= \frac{\partial J}{\partial \mathbf{z}}$, what is $ \frac{\partial J}{\partial \mathbf{W}} = \frac{\partial J}{\partial \mathbf{z}} \frac{\partial \mathbf{z}}{\partial \mathbf{W}} = \delta \frac{\partial \mathbf{z}}{\partial \mathbf{W}}$? )

While taking gradients, we must clear the dimensions of each parameters.  J is a scalar, while $\mathbf{z}$ is a  $n$-dimensional vector, so the Jacobian $\delta= \frac{\partial J}{\partial z}$ is a $1\times n$ vector. Suppose that $\mathbf{W}\in \mathbb{R}^{n\times m}$,$\frac{\partial \mathbf{z}}{\partial \mathbf{W}}$ will be an $n\times n\times m$ tensor which is quite complicated. However, we can take the gradient with respect to a single weight $W_{ij}$. $\frac{\partial \mathbf{z}}{\partial W_{ij}}$ is just a $n\times 1$ vector, which is much easier to deal with. We have
$$
z_k = \sum_{l=1}^{m}W_{kl}x_l
$$

$$
\frac{\partial z_k}{\partial W_{ij}} =\sum_{l=1}^{m}x_l\frac{\partial}{\partial W_{ij}}W_{kl}=
\begin{cases}
    x_j & \text{if $i=k$ and $j=l$}.\\
    0 & \text{if otherwise}.
  \end{cases}
$$

Another way of writing this is 
$$
\frac{\partial \mathbf{z}}{\partial W_{ij}} =
 \left[
 \begin{matrix}
   0  \\
   \vdots \\
   0 \\
   x_j\\
   0\\
   \vdots\\
   0
  \end{matrix}
  \right]\leftarrow \text{$i$th element}
$$
Now let's compute $\frac{\partial J}{\partial W_{ij}}$
$$
\frac{\partial J}{\partial W_{ij}} = \frac{\partial J}{\partial \mathbf{z}}\frac{\partial \mathbf{z}}{\partial W_{ij}} =\left[
\begin{matrix}
\delta_1 & \cdots & \delta_i& \cdots & \delta_n
\end{matrix}
\right]
\left[
 \begin{matrix}
   0  \\
   \vdots \\
   0 \\
   x_j\\
   0\\
   \vdots\\
   0
  \end{matrix}
  \right]=\delta_ix_j
$$
To get $\frac{\partial J}{\partial \mathbf{W}}$ we want a matrix where entry $(i,j)$ is $\delta_ix_j$. 

This matrix is equal to the outer product. So $\frac{\partial J}{\partial \mathbf{W}}=\delta^{T}x^{T}$

These basic identities will be enough for us to compute the gradients for RNN and LSRM networks. 

### 2. The vanishing/exploding gradient problem of RNN:

As we mentioned before, the main difference between RNN and traditional neural networks is that each time the previous output will be brought to the next hidden layer and trained together.

![](https://github.com/blankandwhite1/YingFu.github.io/blob/master/img/RNN.jpg?raw=true)
<center>Image Source: CS224n, Stanford</center>
The hidden vector and the output is computed as such:
$$
\mathbf{h}_t =\sigma( \mathbf{W}\mathbf{h}_{t-1}+\mathbf{W^{hx}}\mathbf{x}_{t})\\
\mathbf{\hat{y}_t}=\mathbf{W^{(S)}}\mathbf{h}_t
$$
**Notations:**

- $\mathbf{x}_t\in \mathbb{R}^d:$ the input vectors at time $t$
- $\mathbf{W^{hx}}\in\mathbb{R}^{D_h\times d}$: weights matrix used to condition the input vector, $x_t$
- $\mathbf{W}\in\mathbb{R}^{D_h\times D_h}$: weights matrix used to condition the output of the previous time-step, $h_{t-1}$
- $\mathbf{h}_{t-1}\in \mathbb{R}^{D_h}$: output of the non_linear function at the previous time-step , $t-1$. $h_0\in \mathbb{R}^{D_h}$ is an initialization vector for the hidden layer at time-step $t=0$.
- $\sigma( )$: the non-linearity activation function (sigmoid here) 
- $\mathbf{y}_t\in \mathbb{R}^V$: the output vectors at time t
- $\mathbf{W^{S}}\in\mathbb{R}^{V\times D_h}$

To do backpropagation through time to update weight parameters , we need to compute the gradient of error with respect to $\mathbf{W}$. Total error is the sum of each error at time step $t$, as $\mathbf{W}$ is fixed all the time steps, so the overall error gradient is:
$$
\frac{\partial E}{\partial  \mathbf{W}}=\sum_{t=1}^{T}\frac{\partial E_t}{\partial \mathbf{W}}
$$

So what is the derivative of $E_t$ with respect to the repeated weight matrix ？Just apply the multivariable chain rule:
$$
\frac{\partial E_t}{\partial  \mathbf{W}}=\sum_{k=1}^{t}\frac{\partial E_t}{\partial \mathbf{W}}\bigg{|}_{k}\frac{\partial \mathbf{W}\big |_{k}}{\partial  \mathbf{W}}=\sum_{k=1}^{t}\frac{\partial E_t}{\partial  \mathbf{W}}\bigg|_{k}
$$
It can be further written as:
$$
\frac{\partial E_t}{\partial  \mathbf{W}}=\sum_{k=1}^{t}\frac{\partial E_t}{\partial \mathbf{W}}\bigg{|}_{k} = \sum_{k=1}^{t}\frac{\partial E_t}{\partial {\mathbf{y_t}}}\frac{\partial \mathbf{y_t}}{\partial \mathbf{h_t}}\frac{\partial \mathbf{h_t}}{\partial \mathbf{h_k}}\frac{\partial \mathbf{h_k}}{\partial \mathbf{W}}
$$
For the term $\frac{\partial \mathbf{h_t}}{\partial \mathbf{h_k}}$, we use another chain rule application to compute it
$$
\frac{\partial \mathbf{h_t}}{\partial \mathbf{h_k}}=\frac{\partial \mathbf{h_t}}{\partial \mathbf{h_{t-1}}}\frac{\partial \mathbf{h_{t-1}}}{\partial \mathbf{h_{t-2}}}...\frac{\partial \mathbf{h_{k+1}}}{\partial \mathbf{h_k}}=\prod_{j=k+1}^{t}\frac{\partial \mathbf{h_j}}{\partial \mathbf{h_{j-1}}}
$$
Thus, we have 
$$
\frac{\partial E}{\partial  \mathbf{W}}=\sum_{t=1}^{T}\sum_{k=1}^{t}\frac{\partial E_t}{\partial {\mathbf{y_t}}}\frac{\partial \mathbf{y_t}}{\partial \mathbf{h_t}} (\prod_{j=k+1}^{t}\frac{\partial \mathbf{h_j}}{\partial {\mathbf{h_{j-1}}}}) \frac{\partial \mathbf{h_k}}{\partial \mathbf{W}}
$$
As the  hidden vector is computed as 
$$
\mathbf{h_t} =\sigma( \mathbf{W}\mathbf{h_{t-1}}+\mathbf{W^{hx}}\mathbf{x_{[t]})}
$$
Using the chain and identities in part 2 we have derived, therefore :
$$
\frac{\partial \mathbf{h_j}}{\partial \mathbf{h_{j-1}}}=diag(\sigma'(\mathbf{W}\mathbf{h_{t-1}}+\mathbf{W^{hx}}\mathbf{x_{[t]}}))\mathbf{W}
$$
As the paper [On the difficulty of training Recurrent Neural Networks](http://proceedings.mlr.press/v28/pascanu13.pdf) has shown:

> To understand the vanishing gradient phenomenon, we need to look at the form of each temporal component, and in particular the matrix factors $\frac{\partial h_t}{\partial h_{k}}$ that take the form of a product of $t-k$ Jacobian matrices. (In the same way a product of $t-k$ real numbers can shrink to zero or explode to infinity, so does this product of matrices).

Consider [matrix $L_2 $ norms](https://en.wikipedia.org/wiki/Matrix_norm)，we have
$$
\left\|\frac{\partial \mathbf{h_j}}{\partial \mathbf{h_{j-1}}}\right\|\leq  \left\|diag(\sigma'(\mathbf{W}\mathbf{h_{t-1}}+\mathbf{W^{hx}}\mathbf{x_{[t]}}))\right\|\left\|\mathbf{W}\right\|
$$
So, 


$$
\left\|\frac{\partial \mathbf{h_t}}{\partial \mathbf{h_{k}}}\right\|=\prod_{j=k+1}^{t}\frac{\partial \mathbf{h_j}}{\partial {\mathbf{h_{j-1}}}}\leq\left\|\mathbf{W}\right\|^{(t-k)}\prod_{j=k+1}^{t} \left\|diag(\sigma'(\mathbf{W}\mathbf{h_{t-1}}+\mathbf{W^{hx}}\mathbf{x_{[t]}}))\right\|
$$
The value of $\sigma'(\mathbf{W}\mathbf{h_{t-1}}+\mathbf{W^{hx}}\mathbf{x_{[t]}})$is the derivation of activation function, so  it can only be as large as 1 given the sigmoid non-linearity function. In the paper [On the difficulty of training Recurrent Neural Networks](http://proceedings.mlr.press/v28/pascanu13.pdf), Pascanu et al. showed that if the largest eigenvalue of $\mathbf{W}$ is less than $1$, then the gradient will shrink exponentially. 

The consequence of gradient vanishing is that the gradient from faraway is lost, and the model weights are only updated with respect to near effects, not long-term effects. There is also another perspective that gradient can be viewed as a measure of the effect of the past on the future. 

There is a similar proof relating a largest eigenvalue larger than 1, the  gradient explodes. If the gradient becomes too big, then the SGD update step becomes too big , causing the divergence of network training. However, we can clip the gradient if the norm of the gradient is greater than some threshold. 

 ## 3. How LSTM solves the vanishing gradient problem?

As we can see above, the consequence of gradient vanishing is that the weights are updated only with respect to near effects, not long-term effects. So is there any possible ways to solve this problem? One possible way is to replace the sigmoid or tanh activation function to relu activation function or initialize the weight matrix  correctly. Another possible way is using the LSTM netwoks. 

### 3.1 Long Short-Term Memory (LSTM)

Long Short-Term Memory, a type of RNN, is proposed by Hochreiter and Schmidhuber in 1997 as a solution to the vanishing gradients problem. The paper is here: [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)

![1581651699866](https://github.com/blankandwhite1/YingFu.github.io/blob/master/img/LSTM.jpg?raw=true)

<center>Image Source:http://colah.github.io/posts/2015-08-Understanding-LSTMs/</center>
We have a sequence of input $\mathbf{x}_{t}$, and we will compute a sequence of hidden states $\mathbf{h}_{t}$ and cell states $\mathbf{c}_{t}$, both are vectors of length $n$.  
$$
\mathbf{f}_{t} = \sigma(\mathbf{W}_f\mathbf{h}_{t-1}+\mathbf{U}_f\mathbf{x}_{t})\\
\mathbf{i}_{t} = \sigma(\mathbf{W}_i\mathbf{h}_{t-1}+\mathbf{U}_i\mathbf{x}_{t})\\
\mathbf{o}_{t} = \sigma(\mathbf{W}_o\mathbf{h}_{t-1}+\mathbf{U}_f\mathbf{x}_{t})
$$

$$
\tilde{\mathbf{c}}_{t} = tanh(\mathbf{W}_c\mathbf{h}_{t-1}+\mathbf{U}_c\mathbf{x}_{t})\\
\mathbf{c}_{t} = \mathbf{f}_{t}\circ \mathbf{c}_{t-1} + \mathbf{i}_{t}\circ \tilde{\mathbf{c}}_{t}\\
\mathbf{h}_{t}=\mathbf{o}_{t}\circ tanh  (\mathbf{c}_{t})
$$

**Note:** **$\circ$ represents element-wise product.** That means gates are applied using element-wise product.

**On time $t$ :**

- $\mathbf{f}_{t}$: **Forget gate**, controls what part of previous cell state are forgotten or kept.
- $\mathbf{i}_{t}$: **Input gate**, controls what part of the new cell content are written to the cell.
- $\mathbf{o}_{t}$: **Output gate**, controls what part of cell are output to hidden state.

- $\tilde{\mathbf{c}}_{t}$: **New cell content ** to be written to the cell
- $\mathbf{c}_t$: **Cell state.** Erase ("forget") some content from the last cell state, and write ("input") some new cell content.
- $\mathbf{h}_t$: Read ("output") some content from the cell.

As mentioned before, the reason why RNN has vanishing gradient problem is the recursive derivative $\frac{\partial \mathbf{h_j}}{\partial \mathbf{h_{j-1}}}$, which is fixed for all time step $t$ .  if the value is greater than 1, the gradient exploding occurs. On the contrary, if the value is less than 1, the gradient vanishing occurs.

In LSTM, we also calculated the recursive derivative  $\frac{\partial \mathbf{c_j}}{\partial \mathbf{c_{j-1}}}$ to loot at what is the main differences compared to RNN.

As the LSTM equation shows: $\mathbf{c_j}$ is the function of $\mathbf{f}_j$ (forget gate), $\mathbf{i}_j$ (input gate), $\mathbf{\tilde{c}}_j$, and each is the function of  $\mathbf{h}_{j-1}$ and further, $\mathbf{h}_{j-1}$is the function of $\mathbf{c}_{j-1}$. Applying the multivariable chain rule, we will have:
$$
\begin{align} 
\frac{\partial \mathbf{c}_j}{\partial \mathbf{c}_{j-1}}&=\frac{\partial \mathbf{c}_j}{\partial \mathbf{f}_j}\frac{\partial \mathbf{f}_j}{\partial \mathbf{h}_{j-1}}\frac{\partial \mathbf{h}_{j-1}}{\partial \mathbf{c}_{j-1}}+\frac{\partial \mathbf{c}_j}{\partial \mathbf{c}_{j-1}}+\frac{\partial \mathbf{c}_j}{\partial \mathbf{i}_{j}}\frac{\partial \mathbf{i}_j}{\partial \mathbf{h}_{j-1}}\frac{\partial \mathbf{h}_{j-1}}{\partial \mathbf{c}_{j-1}}\\
&+\frac{\partial \mathbf{c}_j}{\partial \mathbf{\tilde{c}_j}}\frac{\partial \mathbf{\tilde{c}_j}}{\partial \mathbf{h}_{j-1}}\frac{\partial \mathbf{h}_{j-1}}{\partial \mathbf{c}_{j-1}}
\end{align}
$$

$$
\begin{align}
\frac{\partial \mathbf{c}_j}{\partial \mathbf{c}_{j-1}}
&=diag(\mathbf{c}_{t-1})diag(\sigma'(\cdot))\mathbf{W}_h\ast\mathbf{o}_{j-1}diag(tanh'(\mathbf{c}_{j-1}))\\
&+f_{j}\\
&+diag(\mathbf{\tilde{c}}_j)diag(\sigma(\cdot))\mathbf{W}_i\ast\mathbf{o}_{j-1}diag(tanh'(\mathbf{c}_{j-1}))\\
&+diag(\mathbf{i}_j)diag(tanh(\cdot))\mathbf{W}_o\ast\mathbf{o}_{j-1}diag(tanh'(\mathbf{c}_{j-1}))





\end{align}
$$

So as to RNN, if we want to backpropagate back $k$ times, we just multiply this term k times to update the parameters. 

In RNN, at any step $j$, the recursive derivative $\frac{\partial \mathbf{h_j}}{\partial \mathbf{h_{j-1}}}$ is fixed and can only be one value.  However,  in LSTM, the recursive derivative $\frac{\partial \mathbf{c_j}}{\partial \mathbf{c_{j-1}}}$ can take different values with the different value of $f_j$, so it can solves the gradient vanishing problem to a certain degree.

LSTM doesn't guarantee that there is no vanishing/exploring gradient, but it does provide an easier way for the model to learn long-distance dependencies.

Last but not least, vanishing/exploding gradient problem is not just a problem for RNN. Actually, due to the chain rule and the choice of nonlinearity activation function, gradient can become vanishingly small or explodingly large as it backpropagates, expecially in deep neural networks.  








## 4. Reference
1. [Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

2. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

3. [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/index.html)

4. [Why LSTMs Stop Your Gradients From Vanishing: A View from the Backwards Pass](https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html)


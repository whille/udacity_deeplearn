# linear transaction

<img src="http://latex.codecogs.com/gif.latex?\frac{\partial J}{\partial
\theta_k^{(j)}}=\sum_{i:r(i,j)=1}{\big((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\big)x_k^{(i)}}+\lambda
\theta_k^{(j)}" />

<img src="http://latex.codecogs.com/gif.latex?y = Wx + b" />

# softmax
$$S(y) = \frac{e^{y_i}}{\sum(e^{y_i})} $$


# cross entropy
$$D =(S,L) = - \sum_i L_i log(S_i)$$
notes:
    L: one-hot labels
        S: softmax

# find best W, b
Loss function
$$  L = \frac{1}{N} \sum_i D(S(WX_i + b) , L_i) $$
use gradient dependence


# SGD

momentum
$$M \leftarrow  0.9M + \Delta \alpha $$
learning rate decay


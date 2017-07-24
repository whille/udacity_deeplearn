# linear transaction

$$ y = Wx+b$$

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


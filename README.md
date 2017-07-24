# math

## linear transaction
<img src="http://latex.codecogs.com/gif.latex?y=Wx+b" />

## softmax
<img src="http://latex.codecogs.com/gif.latex?S(y)=\frac{e^{y_i}}{\sum(e^{y_i})}" />


## cross entropy
<img src="http://latex.codecogs.com/gif.latex?D=(S,L)=-\sum_iL_ilog(S_i)" />

notes:
* L: one-hot labels
* S: softmax

## find best W, b
Loss function
<img src="http://latex.codecogs.com/gif.latex?\mathcal%20L=\frac{1}{N}\sum_iD(S(WX_i+b),L_i)" />

use gradient dependence

## SGD

* momentum
* learning rate decay

<img src="http://latex.codecogs.com/gif.latex?M\leftarrow0.9M+\Delta\alpha" />

# deep network
## relu

<img src="http://latex.codecogs.com/gif.latex?hidden=relu(y)" />

## prevent overfitting
* early termination
* regularization

<img src="http://latex.codecogs.com/gif.latex?\mathcal%20L%20\leftarrow\%20\mathcal%20L%20+\beta\frac{1}{2}\|W\|_2^2" />


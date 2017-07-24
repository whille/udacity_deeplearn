# math

## linear transaction
<img src="http://latex.codecogs.com/gif.latex?y=Wx+b" />

## softmax
<img src="http://latex.codecogs.com/gif.latex?S(y)=\frac{e^{y_i}}{\sum(e^{y_i})}" />


## cross entropy
<img src="http://latex.codecogs.com/gif.latex?D=(S,L)= - \sum_i L_i log(S_i)" />

notes:
* L: one-hot labels
* S: softmax

## find best W, b
Loss function
<img src="http://latex.codecogs.com/gif.latex? L = \frac{1}{N} \sum_i D(S(WX_i + b) , L_i)" />

use gradient dependence

## SGD

* momentum
* learning rate decay

<img src="http://latex.codecogs.com/gif.latex?M \leftarrow  0.9M + \Delta \alpha " />


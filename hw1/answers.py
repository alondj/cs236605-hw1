r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
we can see that for a realy small k, eg. 1 ,we will overffit because we will predict blindly according to the training examples.
if k is very large we will consider too much data points and that can introduce noise to the prediction in the exterme our prediction can be seen as a majority vote accros the entire dataset.
the sweet spot is somewhere in the middle where we consider a reasonable amount of close examples to determine a datapoint classification
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
the hinge loss assures that the score of the correct label will larger by at least delta
from all othe label. because we return the label with the heighest score it does not matter
which value of delta we choose as long as its positive.

"""

part3_q2 = r"""
1.the linear classifier is trying to find a linear seperator(weights) that seperates between the different classes as good as possible, trying to come close as much as possible to one that seperates them with a distance of delta. 
2. KNN doesnt seperate between the classes in a linear way he just classifies the exampels according to arbitrery partitions of the domain.
"""

part3_q3 = r"""
we would say the learning rate is not too high because if it was then we would have seen spikes in the loss and accuracy metrics and slower convergence due to irratic changes in the model parameters.
also it's not too low because we see that the curve is steep and 4 epochs is enough for reasonable loss and accuracy values.

in regard to the over/under fitting of the model we would say that the model is slightly overffited on the training set because after epoch 5 we can see the margin between the graphs(before they were almont identical) thus we assume that we start to generalize less on the validtaion set and hinder the improvement of the validation loss and accuracy 
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
we would like to see that all points are along the line y=0
meaning that the difference between the pridected value and the actual value is almost 0.

"""

part4_q2 = r"""
because the regularization loss is less effected by lambda compared to w, eg. small change in lambda will drasticly changge the loss.
but in logspace small change in the value will yield much larger results enabling us to travese a large interval quickly.

in total we tried 60 combinations of hyperParams:
3 polynom degrees and 20 lambda values
"""

# ==============

This repo contains code for building a simple feed-forward neural network from scratch.

Most of the code here should look very familiar to those of you who have taken the deep learning specialization course as taught by Andrew Ng. The only major difference is that this implementation uses a recursive function in lieu of for-loops for forward and backward propagation. This approach enables us to use the call stack as a cache, thus eliminating the need for a explicit (dict, list, tuple) cache.

The actual code for doing this is within the "forward_and_backward_propagate" function - the actual recursive function

measure.ipynb is a notebook that compares the performance of the course implementation and the recursion implementation particularly, the loss, accuracy, time, and memory usage.

If you run this code in google colabs there seems to be a bug with regard to use of the %%timeit magic, if you encounter this error, remove that line, run the code, put the line magic back and then run the code again. Everything seems to work fine afterwards.

Spoiler alert: is there isn't much difference between the two implementations.

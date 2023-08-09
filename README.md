This repo contains code for building a simple feed-forward neural network from scratch.

Most of the code here should look very familiar to those of you who have taken the deep learning specialization course taught by Andrew Ng. The only major difference is that this implementation uses a recursive function in lieu of for-loops for forward and backward propagation. This approach enables us to use the function call stack as a cache for storing the data we will need for backpropagation calculations, thus eliminating the need for a explicit (dict, list, tuple) cache.

The actual recursive function for doing this is within the "forward_and_backward_propagate" function.

measure.ipynb is a notebook that does a janky performance comparison of the course implementation and the recursion implementation. The code compares the loss, accuracy, time, and memory usage. This notebook begins by preparing a small dataset "data_banknote_authentication.txt" that we feed into our neural networks. The dataset itself has 1372 samples and 4 features.

If you run this code in google colabs there seems to be a bug with regard to use of the %%timeit cell magic, if you encounter this error, delete the cell magic temporarily, run the code, put the cell magic back and then run the code again. Everything seems to work fine afterwards.

Spoiler alert: the difference in performance between the two implementations is negligible.

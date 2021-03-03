# CSCI390-lab-2 assignment by Delwys Glokpor 7/17/2020

Contains a project that sets a mock neural network and performs basic NN operations on sets provided to it. The program in _BackProp.py_ read std input of <cmd>, <config_file>, and <data_file>. It uses the data oulled from <config_file> to to build a mimic neural network. It then uses data pulled from <data_file> to train the network, using the backpopagation algorithm to generate errors and derivatives.

_BackProp.py_ python module on run expect <cmd> <config_file> <data_file>. two possible <cmd>: 
-   <verify> <CfgEx.json> <DataEx.json>:
  -   build a model based on <CfgEx.json> and use the inputs in <DataEx.json> to run a single forward propagation. Prints the outcome vector and check it against the expected outcome vector to produce an error estimate. Run a backpropagation and print out the error layer to layer on each weight. 
  -   Tweak weights and report on errors.
-   <run> <CfgTrain.json> <DataTrain.json>:
  -   build a network based on <CfgTrain.json> contents and use that model to run the input output pairs from <DataTrain.json>. 
  -   Perform a mimic training with 3/4 of the data, on batches of 32, forward propagating, backpropagating and updating weights after each batch to reduce error by a rate of .01. Report on the error changes throughout training. 
  -   Uses the last 1/4 of the data as a validation set. Only forward propagate and error reporting. No tweaking weights as this is not training.

(1) Written and tested on *Python 3.8.3*. Make sure to install external *Numpy* library.

(2) <config_file> and <data_file> paths must be passed in to the program through the console std input.

(3) <config_file> and <data_file> must follow a specific *.json* structure (see _CfgEx.json_ for reference).

(4) For quick verify test, `$ python3 BackProp.py verify CfgEx.json DataEx.json` to see the outcome of the NN compared to the expected outcome, the error on the outcome and the error on each weight of the NN.

(5) For quick run test, `$ python3 BackProp.py verify CfgTrain.json DataTrain.json` to see the progressive training and validation process showing you for each input outout pairs, error on estimate compared to expected outcome, and error on each weight.

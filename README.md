# CSCI390-lab-2 assignment by Delwys Glokpor 7/17/2020
Contains code that read std input of <cmd> <config_file> <data_file>, build a mimic neural network from <config_file> data and then use <data_file> to train the network, using backrpopagation algorithm to generate errors and derivatives.
Written and tested on Python 3.8.3.
"BackProp.py" python module. On run expect <cmd> <config_file> <data_file>
  two possible <cmd>: 
    <verify> <CfgEx.json> <DataEx.json> - build a model based on <CfgEx.json> and use the inputs in
      <DataEx.json> to verify that the network is behaving properly through forward and backrpropagation.
      Tweak weights and report on errors.
     <run> <CfgTrain.json> <DataTrain.json> - build a network based on <CfgTrain.json> contents and use that model to run the         input output pairs from <DataTrain.json>. Perform a mimic training with 3/4 of the data, on batches of 32, forward      propagating, backpropagating and updating weights after each batch to reduce error by a rate of .01. Report on the error  changes throughout training. Uses the lats 1/4 of the data as a validation set. Only forward propagate and error reporting. No  tweaking weights as this is not training.
  

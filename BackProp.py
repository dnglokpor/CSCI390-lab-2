import numpy as np
import numpy.random as rnd
import json, sys

class Layer:
    # dim  -- number of neurons in the layer
    # prev -- prior layer, or None if input layer.  If None, all following
    #  params are ignored.
    # act -- activation function: accept np.array of z's; return
    #  np.array of a's
    # act_prime -- derivative function: accept np.arrays of z's and of a's,
    #  return derivative of activation wrt z's, 1-D or 2-D as appropriate
    # weights -- initial weights. Set to small random if "None".
    #
    # All the following member data are None for input layer
    # in_weights -- matrix of input weights
    # in_derivs -- derivatives of E/weight for last sample
    # zs -- z values for last sample
    # z_derivs -- derivatives of E/Z for last sample
    # batch_derivs -- cumulative sum of in_derivs across current batch
    def __init__(self, dim, prev = None, act = None, act_prime = None, weights=None):
        
        self.dim = dim
        self.outputs = np.zeros(self.dim)
        self.prev = prev
        if self.prev != None: # input layer
            self.act = act
            self.act_prime = act_prime
            self.zs = np.zeros(dim) # array neurons
            self.in_weights = np.array(weights) # list of list of weights
            self.z_derivs = np.zeros(self.dim) # list of nNeurons dE/dz
            self.in_derivs = np.zeros((self.dim, len(self.in_weights[0]))) # list of list of nNeurons dE/dw
            self.batch_derivs = np.zeros((self.dim, len(self.in_weights[0])))
    
    def get_dim(self):
        return self.dim
     
    # Return the dE/dW for the weight from previous layer's |src| neuron to
    # our |trg| neuron. ???
    def get_deriv(self, src, trg):
        # get_deriv(0,1) will return the [1,0] element of the in_derivs
        # print(self.in_derivs, src, trg)
        return self.in_derivs[trg][src]
   
    # Compute self.outputs, using vals if given, else using outputs from
    # previous layer and passing through our in_weights and activation.
    def propagate(self, vals = None):
        if self.prev == None: # input layer
            if type(vals) == type(None): # fatal exception: quit program
                sys.exit()
            else:
                self.outputs = vals
        else: # not input layer
            input_vector = vals
            if input_vector == None:
                input_vector = self.prev.outputs # get output of previous layer
            input_vector = np.append(input_vector, 1) # extra input for bias
            self.zs = np.dot(self.in_weights, input_vector) # compute inner sums
            self.outputs = self.act(self.zs)
    
    # Compute self.in_derivs, assuming 
    # 1. We have a prev layer (else in_derivs is None)
    # 2. Either
    #    a. There is a next layer with correct z_derivs, OR
    #    b. The provided err_prime function accepts np arrays 
    #       of outputs and of labels, and returns an np array 
    #       of dE/da for each output
    def backpropagate(self, err_prime = None, labels = None):
        if self.prev != None: # any layer but input layer
            jacobian = self.act_prime(self.outputs, self.zs)
            if err_prime != None: # and labels != None: # output layer
                a_derivs = err_prime(labels, self.outputs)
                self.z_derivs = np.dot(jacobian, a_derivs)
            self.in_derivs = np.outer(self.z_derivs, np.append(self.prev.outputs, 1))
            self.batch_derivs += self.in_derivs
            if self.prev.prev != None: # not first hidden layer
                prev_jacobian = self.prev.act_prime(self.prev.outputs, self.prev.zs)
                prev_a_derivs = np.dot(np.transpose(self.in_weights), self.z_derivs)
                self.prev.z_derivs = np.multiply(prev_jacobian, prev_a_derivs[:-1]) # previous layer's ds
            
    # Adjust all weights by avg gradient accumulated for 
    #  current batch * -|rate|
    def apply_batch(self, batch_size, rate):
        self.in_weights -= (self.batch_derivs / batch_size) * rate
     
    # Reset internal data for start of a new batch
    def start_batch(self):
        self.batch_derivs = np.zeros((self.dim, len(self.in_weights[0])))

    # Add delta to the weight from src node in prior layer
    # to trg node in this layer.
    def tweak_weight(self, src, trg, delta):
        self.in_weights[trg][src] += delta
        # print(src, trg, self.in_weights)
    
    # Return string description of self for debugging
    def __repr__(self):
        descr = str()
        descr += "weights matrix:\n{}\ninner sums: {}\noutputs: {}\n".format(self.in_weights,
                                                                    self.zs, self.outputs)
        return descr

class Network:
    # arch -- list of (dim, act) pairs
    # err -- error function: "cross_entropy" or "mse"
    # wgts -- list of one 2-d np.array per layer in arch
    def __init__(self, arch, err, wgts = None):
        self.model = list()
        if err == "cross_entropy":
            self.loss_func = crossentropy
            self.loss_prime = crossentropy_prime
        N_NEURONS = 0
        ACTIVATION = 1
        prev_layer = None
        for index, layer_descr in enumerate(arch):
            act = None
            act_prime = None
            if layer_descr[ACTIVATION] == "": # input layer
                prev_layer = Layer(layer_descr[N_NEURONS])
            else: #layer_descr[ACTIVATION] != "" # not input layer
                act = relu # set it to relu. 
                act_prime = relu_prime
                if layer_descr[ACTIVATION] == "softmax":
                    act = softmax # set it to softmax if it's not relu
                    act_prime = softmax_prime
                prev_layer = Layer(layer_descr[N_NEURONS], prev_layer, act, act_prime,
                    weights = wgts[index - 1]) # build layer object
            self.model.append(prev_layer) # add the layer to the network
            
    # Forward propagate, passing inputs to first layer, and returning outputs
    # of final layer
    def predict(self, inputs):
        self.model[0].propagate(inputs)
        for layer in self.model[1:]:
            layer.propagate()
        return self.model[-1].outputs
    
    # Assuming forward propagation is done, return current error, assuming
    # expected final layer output is |labels|
    def get_err(self, labels):
        return self.loss_func(self.model[-1].outputs, labels)
    
    # Assuming a predict was just done, update all in_derivs, and add to
    # batch_derivs
    def backpropagate(self, labels):
        for index, layer in reversed(list(enumerate(self.model))):
            if index == len(self.model) - 1: # last layer
                layer.backpropagate(self.loss_prime, labels)
            else:
                layer.backpropagate()
    
    # Verify all partial derivatives for weights by adding an
    # epsilon value to each weight and rerunning prediction to
    # see if change in error correctly reflects weight change
    def validate_derivs(self, inputs, outputs):
        epsilon = .01
        predicted = self.predict(inputs)
        ini_error = self.get_err(outputs)
        print("{} vs {} for {}".format(predicted, outputs, ini_error))
        self.backpropagate(outputs)
        for index, layer in enumerate(self.model):
            if index > 0: # not input layer
                #print("layer[{}] in_derivs:\n{}".format(index, self.model[index].in_derivs))
                #sys.exit()
                for trg in range(layer.get_dim()):
                    for src in range(layer.prev.get_dim() + 1): # extra bias dimension
                        layer.tweak_weight(src, trg, epsilon) # adjust by .01
                        self.predict(inputs) # forward propagation
                        cur_error = self.get_err(outputs)
                        change = cur_error - ini_error
                        if change != 0:
                            cur_ratio = abs(change / (epsilon * layer.get_deriv(src, trg)) - 1) * 100
                            #print("expected: {}\tactual: {}".format(abs(epsilon * layer.get_deriv(src, trg)), change))
                        else:
                            cur_ratio = 0.
                        print("Test {}/{} to {}/{}: {:.6f} - {:.6f} = {:.6f} ({:.4f}% error)".format(
                            index - 1, src, index, trg, cur_error, ini_error, change, cur_ratio)
                        )
                        layer.tweak_weight(src, trg, -epsilon) # return to initial value
            
    # Run a batch, assuming |data| holds input/output pairs comprising the 
    # batch. Forward propagate for each input, record error, and backpropagate.
    # At batch end, report average error for the batch, and do a derivative 
    # update.
    def run_batch(self, data, rate):
        batch_error = 0
        for pair in data:
            self.predict(pair[0]) # forward propagation
            batch_error += self.get_err(pair[1]) # record error
            self.backpropagate(pair[1]) # backpropagation
        print("Batch error: {:.3f}".format(batch_error / len(data))) # report error
        for index, layer in enumerate(self.model):
            if index > 0:
                layer.apply_batch(len(data), rate) # update weights
                layer.start_batch() # reset batch derivs record

# relu activation
def relu(zs_vector):
    activated = np.ndarray.copy(zs_vector) # makes a duplicate of passed
    for index, value in enumerate(activated):
        if value < 0:
            activated[index] = 0
    return activated    

# relu derivative
def relu_prime(as_vector, zs_vector = None):
    jacobian = np.zeros(len(as_vector))
    for index, a in enumerate(as_vector):
        if a > 0:
            jacobian[index] = 1
    return jacobian

# softmax activation
def softmax(zs_vector):
    activated = np.exp(zs_vector) # new array of same shape but filled with ones
    activated /= np.sum(activated)
    return activated

# softmax derivative
def softmax_prime(as_vector, zs_vector = None):
    # computes da/dz for softmax using dai/dzi = ai-ai^2 and dai/dzj = -ai*aj
    jacobian = np.zeros((len(as_vector), len(as_vector)))
    row = np.zeros(len(as_vector))
    for i, ai in enumerate(as_vector):
        for j, aj in enumerate(as_vector):
            if i == j:
                row[j] = (ai - ai * ai)
            else: # i != j
                row[j] = -(ai * aj)
        jacobian[i] = row
    return jacobian

# replace |0| elements by |1| to not crash np.log 
def no_zeros(vector):
        no_zero_vec = np.ndarray.copy(np.array(vector)) # duplicate
        for index in range(len(vector)):
            if vector[index] == 0: # get rid of possible 0 elements
                no_zero_vec[index] = 1
        return no_zero_vec
    
# crossentropy loss
def crossentropy(as_vector, ys_vector):
    error = - np.sum((np.multiply(ys_vector, np.log(no_zeros(as_vector))) -
                                        np.multiply(ys_vector, np.log(no_zeros(ys_vector)))))
    return error

# crossentropy derivative
def crossentropy_prime(ys_vector, as_vector):
    error_partials = -np.ndarray.copy(np.array(ys_vector, dtype = float))
    error_partials /= as_vector # how to deal with divides by 0?
    return error_partials

def main(cmd, cfg_file, data_file):
    # load files contents first
    with open(cfg_file, 'r') as cfg:
        config = json.load(cfg)
    with open(data_file, 'r') as data:
        io_pairs = json.load(data)
    # make a network object that fits the config
    model = Network(config["arch"], config["err"], config["wgts"]) 
    # program tree
    if cmd == "verify":
        for pair in io_pairs:
            model.validate_derivs(pair[0], pair[1])
    elif cmd == "run":
        rate = .01 # this rate is gives results in accordance with project description sample
        sep = (len(io_pairs) // 4) * 3
        train_set = io_pairs[:sep]
        valid_set = io_pairs[sep:]
        batch_size = 32
        limit = 0
        n_epochs = (len(train_set) // batch_size) + 1 # max possible num of epochs
        for epoch in range(n_epochs): # training loop
            if limit + batch_size < len(train_set):
                limit += batch_size
            else:
                limit = len(train_set)
            batch = train_set[epoch*batch_size:limit] # slicing the current batch
            print("Batch {}:{}".format(epoch * batch_size, limit))
            model.run_batch(batch, rate)
        valid_err = 0
        for pair in valid_set: # validation loop
            model.predict(pair[0])
            valid_err += model.get_err(pair[1])
        print("Validation error: {:.3f}".format(valid_err / len(valid_set)))
    else:
        sys.exit()

main(sys.argv[1], sys.argv[2], sys.argv[3])
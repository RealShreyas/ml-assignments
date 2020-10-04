import numpy as np

class RegressionTree:
    def fit(self,input,output):
        #indexes holds indices of the subset of data that is being considered at the current node
        #initially, the entire dataset is under consideration at the root node
        indexes = np.array(np.arange(len(output)))
        self.tree = Dtree_Node(input,output,indexes)
        return self

    def predict(self,input):
        return self.tree.predict(input.values)



class Dtree_Node:
    def __init__(self,input,output,indexes):
        self.input = input
        self.output = output
        self.indexes = indexes
        self.score = float('inf') #initialise the variance of node to infinity
        self.split_dataset()  # function to find where to split the dataset


    def find_best_attribute_to_split(self,split_attribute_index):  #checks if current attribute improves variance of node
        input = self.input.values[self.indexes,split_attribute_index]
        for row in range(len(self.indexes)):  #for each row
            left_split = input <= input[row]
            right_split = input > input[row]

            curr_score = self.score_node(left_split, right_split)
            if curr_score < self.score:   #if variance has reduced, update new score and consider this attribute for splitting on
                self.split_attribute_index = split_attribute_index
                self.score = curr_score
                self.split_index = input[row]



    def score_node(self,left_split,right_split):
        output = self.output[self.indexes]
        var_left = output[left_split].var()   #compute variances of left and right split of data
        var_right = output[right_split].var()
        return var_left * left_split.sum() + var_right * right_split.sum()   #use weighted average of variance as score



    def split_dataset(self):
        for col in range(self.input.shape[1]):
            self.find_best_attribute_to_split(col)   #check each attribute to find the best attribute to split the data at current node
        if self.is_leaf:    #if can't improve variance by splitting on any attribute, the node is a leaf node and is not split further
            return
        #if not a leaf node
        input = self.split_on_best_attribute   #we split the dataset on the attribute found earlier
        left_split = np.nonzero(input <= self.split_index)[0]
        right_split = np.nonzero(input > self.split_index)[0]
        self.left_split = Dtree_Node(self.input,self.output,self.indexes[left_split])
        self.right_split = Dtree_Node(self.input,self.output,self.indexes[right_split])

    @property
    def is_leaf(self):
        return (self.score == float('inf'))  # if score isnt changed while finding best attribute to split, node is a leaf node.

    @property
    def split_on_best_attribute(self):
        return self.input.values[
            self.indexes, self.split_attribute_index]  # split_attribute_index is index of attribute that node is split on

    def predict_row(self, data):
        if self.is_leaf: return np.mean(self.output[self.indexes])  #for leaf, predict the avg of all data at this node
        if data[self.split_attribute_index] <= self.split_index:
            node = self.left_split
        else:
            node = self.right_split
        return node.predict_row(data)

    def predict(self, input):
        return np.array([self.predict_row(data) for data in input])
The following libraries need to be installed for the program to run:
1. numpy
2. pandas
3. sklearn
4. matplotlib

The file main.py contains the code for training the tree by performing an 80/20
split on the dataset and running 10 times and printing the average accuracy. This is also
prints the plots of accuracy vs depth and accuracy vs error. The pruning of tree is also done
to find the regression tree which is more accurate. This contains the plot of accuracy vs alpha
which is the statistic that determines the branch to be pruned.

The file decision_tree.py contains the implementation of the decision tree from scratch.
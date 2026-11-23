from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This file is part of the computer assignments for the course DD1418 at KTH.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.5  # The learning rate.


    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.


    MAX_ITERATIONS = 100 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000 # Minibatch size (only for minibatch gradient descent)

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).


            # X kan dela in ord i fyra olika kategorier

            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)



    # ----------------------------------------------------------------------


    def loss(self, x, y):
        """
        Computes the loss function given the input features x and labels y
        
        :param      x:    The input features
        :param      y:    The correct labels
        """

        # Implementering enligt loss formeln
        total_loss = 0

        # Loopar på alla ord (datapoints)
        for i in range(self.DATAPOINTS):

            z = 0

            # Loopar alla features för varje enskilt ord, dotproduct
            for k in range(self.FEATURES):
                z += self.theta[k] * x[i][k]

            sig_prob = self.sigmoid(z)
        
            # Kollar labeln på ordet
            if y[i] == 1:
                cond_prob = sig_prob
            else:
                # Komplementet
                cond_prob = 1 - sig_prob
            
            # Totala lossen, adderar alla pga logaritm sannolikheter
            total_loss += math.log(cond_prob)
        
        # Delar på antal datapoints m, enligt formeln
        return -total_loss/self.DATAPOINTS


    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + np.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """

        # Implementerar enligt formeln
        z = 0

        # Loopar igenom för antal features som finns
        for k in range(self.FEATURES):

            # Theta är vikterna för features
            # datapoints är egenskaper, stor bokstav osv.

            # z = dot produkten av theta-vektorn och datapoint vektor
            z += self.theta[k] * self.x[datapoint][k]

        # Räknar ut sannolikheten att datapunkt i tillhör klass/label 1
        prob = self.sigmoid(z)


        if label == 1:
            return prob
        else:
            return 1 - prob


    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """

        # Implementerar enligt givna formeln 

        # Loopar igenom varje feature för räkna ut dess gradient
        for k in range(self.FEATURES):
            self.gradient[k] = 0

        for i in range(self.DATAPOINTS):
                # Räknar ut sannolikheten från de betingade
                cond_prob = self.conditional_prob(1,i)

                # Tar vår gissning - vårt riktiga värde för att se hur långt från facit vi är
                error = cond_prob - self.y[i]
                # Multiplicerar featurvärdet med med felet och lägger till i summan
                
                for k in range(self.FEATURES):
                   # X [1,1] dvs är [capitalized, first in sentence]
                   self.gradient[k] += self.x[i][k] * error

            # Sparar i gradientvektorn för varje feature
        
        # hmmmm
        for k in range(self.FEATURES):
            self.gradient[k] /= self.DATAPOINTS
     


    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """
        for k in range(self.FEATURES):
            self.gradient[k] = 0

        for i in minibatch:
                # Räknar ut sannolikheten från de betingade
                cond_prob = self.conditional_prob(1,i)

                # Tar vår gissning - vårt riktiga värde för att se hur långt från facit vi är
                error = cond_prob - self.y[i]
                # Multiplicerar featurvärdet med med felet och lägger till i summan
                
                for k in range(self.FEATURES):
                   # X [1,1] dvs är [capitalized, first in sentence]
                   self.gradient[k] += self.x[i][k] * error
        for k in range(self.FEATURES):
            self.gradient[k] /= len(minibatch)
     

        
        # YOUR CODE HERE


    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """
        
        # Räknar ut sannolikheten från de betingade
        cond_prob = self.conditional_prob(1,datapoint)

        # Tar vår gissning - vårt riktiga värde för att se hur långt från facit vi är
        error = cond_prob - self.y[datapoint]
        # Multiplicerar featurvärdet med med felet och lägger till i summan
                
        for k in range(self.FEATURES):
        # X [1,1] dvs är [capitalized, first in sentence]
            self.gradient[k] = self.x[datapoint][k] * error


    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        gradient_square_sum = float("inf")

        for _ in range(self.MAX_ITERATIONS):

            i = random.randrange(self.DATAPOINTS)
            self.compute_gradient(i)

            for k in range(self.FEATURES):
                self.theta[k] = self.theta[k] - self.LEARNING_RATE * self.gradient[k]

            gradient_square_sum = 0
            for i in range(self.FEATURES):

                gradient_square_sum += self.gradient[i] ** 2
            self.update_plot(self.loss(self.x, self.y))
   

    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        
        gradient_square_sum = float("inf")

        while gradient_square_sum > self.CONVERGENCE_MARGIN:

            minibatch = random.sample(range(self.DATAPOINTS), self.MINIBATCH_SIZE)
            self.compute_gradient_minibatch(minibatch)

            for k in range(self.FEATURES):
                self.theta[k] = self.theta[k] - self.LEARNING_RATE * self.gradient[k]

            gradient_square_sum = 0
            for i in range(self.FEATURES):

                gradient_square_sum += self.gradient[i] ** 2
            self.update_plot(self.loss(self.x, self.y))

            print(gradient_square_sum)
      

    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)

        gradient_square_sum = float("inf")

        while gradient_square_sum > self.CONVERGENCE_MARGIN:

            self.compute_gradient_for_all()

            for k in range(self.FEATURES):
                self.theta[k] = self.theta[k] - self.LEARNING_RATE * self.gradient[k]

            gradient_square_sum = 0
            for i in range(self.FEATURES):

                gradient_square_sum += self.gradient[i] ** 2
            self.update_plot(self.loss(self.x, self.y))

            print(gradient_square_sum)
 

    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:');

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, 1+max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    print("startar")
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    print("init")
    b.fit()
    print("fit done")
    b.print_result()


if __name__ == '__main__':
    main()

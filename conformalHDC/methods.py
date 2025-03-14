import numpy as np
import random
import pdb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class ConformalHDC():
    def __init__(self, inputs, labels, calib_size=0.2, sim_measure="euclidean",
                random_state=0, verbose=True, progress=True):

        self.calib_size = calib_size
        self.sim_measure = sim_measure

        self.random_state = random_state
        self.verbose = verbose
        self.progress= progress 

        self.inputs_train, self.inputs_calib, self.labels_train, self.labels_calib =\
        train_test_split(inputs, labels, test_size=self.calib_size, random_state=self.random_state)

        # Train the HD model
        self.oneHvPerClass(self.inputs_train, self.labels_train)
        self.labels = list(self.classHVs.keys())

        
    def _compute_calib_scores(self, score_type=None, **kwargs):
        self.scores_calib = {}
        scores_calib = self._compute_nonconformity_scores(self.inputs_calib, self.labels_calib, score_type = score_type, **kwargs)
    
        for i, label in enumerate(self.labels_calib):
            if (label in self.scores_calib.keys()):
                self.scores_calib[label].append(scores_calib[i])
            else:
                self.scores_calib[label] = [scores_calib[i]]


    def oneHvPerClass(self, HVs, labels, normalize=True):
        #This creates a dict with no duplicates
        classHVs = dict()
        countHVs = dict()
        for i in range(len(labels)):
            label = labels[i]
            if (label in classHVs.keys()):
                classHVs[label] = np.array(classHVs[label]) + np.array(HVs[i])
                countHVs[label] += 1
            else:
                classHVs[label] = np.array(HVs[i])
                countHVs[label] = 1

        if normalize:
            for label in classHVs.keys():
                classHVs[label] /= countHVs[label] 
        
        self.countHVs = countHVs
        self.classHVs = classHVs
    

    # implement the encoding functions and other nonconformity scores later below 
    def _compute_nonconformity_scores(self, HVs, labels, 
                                   score_type=None, **kwargs):
        ''' Computes the nonconformity scores of the HVs and corresponding classHVs
        '''

        # Default confomity score is just euclidean distance
        if score_type == None:
            classHVs_batch =  np.array([self.classHVs[label] for label in labels])

            if self.sim_measure == "euclidean":
                scores = np.linalg.norm(HVs - classHVs_batch, axis=1)
        
        else:
            scores = np.zeros(len(labels))
            for i, (HV, label) in enumerate(zip(HVs, labels)):
                sim_all_classes = 0
                sim_true_class = 0
                for l in self.labels:
                    if self.sim_measure == "euclidean":
                        sim_l = np.linalg.norm(HV - self.classHVs[l])
                    sim_all_classes += sim_l
                    if l == label:
                        sim_true_class = sim_l

                if score_type == "normalized_discount":
                    score = (sim_all_classes-sim_true_class)/(sim_all_classes)*(1/sim_true_class)
                elif score_type == "discount":
                    score = (sim_all_classes-sim_true_class)/(sim_true_class)*(1/sim_true_class)
                elif score_type == "normalized_ratio":
                    score = (sim_all_classes-sim_true_class)/(sim_all_classes)
                elif score_type == "ratio":
                    score = (sim_all_classes-sim_true_class)/(sim_true_class)
                elif score_type == "penalized":
                    penalty = kwargs.get("penalty",1)
                    sim_other_classes = sim_all_classes - sim_true_class
                    score = (1/sim_true_class) - penalty*(1/sim_other_classes)
                # convert the discounted score to noncomformity scores, e.g. small scores <=> more likely to be true labels
                scores[i] = - score 
            
        return scores


    def predict(self, inputs_test):
        scores = []
        n_test = len(inputs_test)
        for label in self.labels:
            tmp_scores = self._compute_nonconformity_scores(inputs_test, [label]*n_test).reshape(-1,1)
            scores.append(tmp_scores)
        
        scores = np.concatenate(scores,axis=1)
        predicted_indices = np.argmin(scores, axis=1)
        predictions = [self.labels[idx] for idx in predicted_indices]

        return predictions


    def conformalPS(self, inputs_test, alpha, 
                    score_type=None, allow_empty=False,
                    **kwargs):
        ''' Computes the conformal prediction sets at significance level alpha
        '''

        self._compute_calib_scores(score_type=score_type, **kwargs)

        n_test = len(inputs_test)
        self.quantiles = {}

        psets = psets = [[] for _ in range(n_test)] 
        for label in self.labels:
            n_calib = len(self.scores_calib[label])
            scores = self._compute_nonconformity_scores(inputs_test, [label]*n_test, score_type=score_type, **kwargs).reshape(-1,1)
            quantile = np.quantile(self.scores_calib[label], (n_calib+1)*(1-alpha)/n_calib)
            self.quantiles[label] = quantile

            for i in range(n_test):
                if scores[i] < quantile:
                    psets[i].append(label)
        
        if not allow_empty:
            pred = self.predict(inputs_test)
            for i, pset in enumerate(psets):
                if len(pset)==0:
                    pset.append(pred[i])

        return psets


    def plot_conformal_boundary(self, inputs_test, labels_test):
        x_min, x_max = np.min(inputs_test[:, 0])-1, np.max(inputs_test[:, 0])+1
        y_min, y_max = np.min(inputs_test[:, 1])-1, np.max(inputs_test[:, 1])+1

        # Create the plot
        fig, ax = plt.subplots()

        # Plot decision boundaries as circles
        circle1 = plt.Circle(self.classHVs[0], self.quantiles[0], color='r', alpha=0.2, fill=True, label='Class 0 Region')
        circle2 = plt.Circle(self.classHVs[1], self.quantiles[1], color='b', alpha=0.2, fill=True, label='Class 1 Region')

        ax.add_artist(circle1)
        ax.add_artist(circle2)

        # Scatter plot of the test points
        colors = ['red' if c == 0 else 'blue' for c in labels_test]
        ax.scatter(inputs_test[:, 0], inputs_test[:, 1], c=colors,  s=20, alpha=0.7, label='Test Points')

        # Plot the centers of the classes
        ax.scatter(self.classHVs[0][0], self.classHVs[0][1], color='r', marker='x', s=100, label='Class 0 Center')
        ax.scatter(self.classHVs[1][0], self.classHVs[1][1], color='b', marker='x', s=100, label='Class 1 Center')

        # Labels and legend
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('ConformalHDC Decision Boundaries')
        ax.legend()

        return plt


    def plot_HDC_boundary(self, inputs_test, labels_test, 
                          xlim=None, ylim=None, alpha_test_point=0.7, alpha_region=0.2, 
                          boundary=True, center=True):
        if xlim is None:
            x_min, x_max = np.min(inputs_test[:, 0])-1, np.max(inputs_test[:, 0])+1
        else:
            x_min, x_max = xlim[0], xlim[1]

        if ylim is None:
            y_min, y_max = np.min(inputs_test[:, 1])-1, np.max(inputs_test[:, 1])+1
        else:
            y_min, y_max = ylim[0], ylim[1]

        midpoint = (self.classHVs[0] + self.classHVs[1]) / 2
        slope = (self.classHVs[1][1] - self.classHVs[0][1]) / (self.classHVs[1][0] - self.classHVs[0][0])
        perp_slope = -1 / slope

        # Equation of the perpendicular bisector: y = perp_slope * (x - midpoint_x) + midpoint_y
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = perp_slope * (x_vals - midpoint[0]) + midpoint[1]

        # Create the plot
        plt.figure()

        if boundary:
            plt.fill_between(x_vals, y_vals, y_min, color='red', alpha=alpha_region, label='Class 0 Region')
            plt.fill_between(x_vals, y_vals, y_max, color='blue', alpha=alpha_region, label='Class 1 Region')

        plt.scatter([], [], color='red', label='Class 0')  # Empty scatter to add red color to the legend
        plt.scatter([], [], color='blue', label='Class 1')  # Empty scatter to add blue color to the legend

        # Scatter plot of the test points
        colors = ['red' if c == 0 else 'blue' for c in labels_test]
        plt.scatter(inputs_test[:, 0], inputs_test[:, 1], c=colors,  s=20, alpha=alpha_test_point)

        if center:
            # Scatter plot of the class centers
            plt.scatter(self.classHVs[0][0], self.classHVs[0][1], color='darkred', marker='x', s=80, label='Class 0 Center')
            plt.scatter(self.classHVs[1][0], self.classHVs[1][1], color='darkblue', marker='x', s=80, label='Class 1 Center')

        # Add labels and other settings
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        #plt.title('Traditional HDC Decision Boundaries')
        plt.legend()

        # Display the plot
        plt.show()

        return plt
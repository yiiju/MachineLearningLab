# Lab3 - Bayesian Learning and Boosting

### µ<sub>k</sub> and Σ<sub>k</sub>

Using the ML-estimates (Maximum Likelihood) for the Gaussian distributed data with 95%-confidence interval.

<div align="center">
    <img src="./image/gaussian.png" width="300" />
</div>

###  Naive Bayes Classifier on the Iris Dataset

<table align="center">
    <tr>
        <td>
            <img src="./image/NaiveBayes_Iris.png" alt="Naive Bayes Classifier on Iris Dataset">
        </td>
        <td>
            <img src="./image/NaiveBayes_Adaboost_Iris.png" alt="Naive Bayes Classifier with Adaboost on Iris Dataset">
        </td>
    </tr>
</table>

### Decision Tree Classifier on the Iris Dataset

<table align="center">
    <tr>
        <td>
            <img src="./image/DecisionTree_Iris.png" alt="Decision Tree Classifier on Iris Dataset">
        </td>
        <td>
            <img src="./image/DecisionTree_Adaboost_Iris.png" alt="Decision Tree Classifier with Adaboost on Iris Dataset">
        </td>
    </tr>
</table>

### Naive Bayes Classifier on the Vowel Dataset

<table align="center">
    <tr>
        <td>
            <img src="./image/NaiveBayes_Vowel.png" alt="Naive Bayes Classifier on Vowel Dataset">
        </td>
        <td>
            <img src="./image/NaiveBayes_Adaboost_Vowel.png" alt="Naive Bayes Classifier with Adaboost on Vowel Dataset">
        </td>
    </tr>
</table>

### Decision Tree Classifier on the Vowel Dataset

<table align="center">
    <tr>
        <td>
            <img src="./image/DecisionTree_Vowel.png" alt="Decision Tree Classifier on Vowel Dataset">
        </td>
        <td>
            <img src="./image/DecisionTree_Adaboost_Vowel.png" alt="Decision Tree Classifier with Adaboost on Vowel Dataset">
        </td>
    </tr>
</table>

> When can a feature independence assumption be reasonable and when not?

> How does the decision boundary look for the Iris dataset? How could one improve the classification results for this scenario by changing classifier or, alternatively, manipulating the data?

> Compute the classification accuracy of the boosted classifier on some data sets and compare it with those of the basic classifier on the vowels and iris data sets:

1. Is there any improvement in classification accuracy? Why/why not?

    Bayes classifier:

    Decision tree classifier:

2. Plot the decision boundary of the boosted classifier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?

    Bayes classifier:

    Decision tree classifier:

3. Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?

    Bayes classifier:

    Decision tree classifier:

> If you had to pick a classifier, naive Bayes or a decision tree or the boosted versions of these, which one would you pick? Motivate from the following criteria:

• Outliers

• Irrelevant inputs: part of the feature space is irrelevant

• Predictive power

• Mixed types of data: binary, categorical or continuous features, etc.

• Scalability: the dimension of the data, D, is large or the number of instances, N, is large, or both.
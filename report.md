# Report ({report_no})

*Date*: {date}
*Time*: {time}

## Semantic Relation Classification
Number of Classes to predict: {classes_no}
Classes: {pred_classes}

Duration: {duration}

### Dataset description:
* Number or folds: {folds_no}
* Training (N={training_no})
	* {training_pairs}
* Validation (N={validation_no}):
	* {validation_pairs}
* Test (N={test_no}):
	* {test_pairs}

### Overall performance ({good_no} of {test_no}):
* Precision: {prec}
* Recall: §{rec}
* F1: {f}1§
* other metrics

### Confusion matrix:

![](confusion_{report_no}.png)

*Figure 1: confusion matrix for...*

### Recall/Precision Curve:

*Figure 2: Precision/recall curve in the overall performance*

### Learning Curve

Figure 3: Learning curve in training and validation/test

### Variance and Bias
https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html

### Recommendations
{recommendations}

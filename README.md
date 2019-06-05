# Interpreting Black Box Models via Hypothesis Testing
This is the code for the paper "Interpreting Black Box Models via Hypothesis Testing" by Collin Burns, Jesse Thomason, and Wesley Tansey. 

> While many methods for interpreting machine learning models have been proposed, they are often ad hoc, difficult to interpret, and come with limited guarantees. This is especially problematic in science and medicine, where model interpretations may be reported as discoveries or guide patient treatments. As a step toward more principled and reliable interpretations, in this paper we reframe black box model interpretability as a multiple hypothesis testing problem. The task is to discover ``important'' features by testing whether the model prediction is significantly different from what would be expected if the features were replaced with uninformative counterfactuals. We propose two testing methods: one that provably controls the false discovery rate but which is not yet feasible for large-scale applications, and an approximate testing method which can be applied to real-world data sets. In simulation, both tests have high power relative to existing interpretability methods. When applied to state-of-the-art vision and language models, the framework selects features that intuitively explain model predictions. The resulting explanations have the additional advantage that they are themselves easy to interpret.

## Instructions

Requires Python 3+, but each type of experiment (synthetic, image, and language) has different particular requirements. See the README in each directory for the specific dependencies and instructions.

## Citation
If you find this work helpful, please consider citing our paper.

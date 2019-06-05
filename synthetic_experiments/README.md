This is the code for replicating the synthetic experiments. 

Requirements
---
- Python 3 with standard packages.
- Keras and Tensorflow.
- [SHAP](https://github.com/slundberg/shap)
- [LIME](https://github.com/marcotcr/lime)
- [DeepExplain](https://github.com/marcoancona/DeepExplain)

Usage
---
1. Run (disc/nn)_experiment.py to compute the feature values for the discontinuous model or neural network respectively.
2. Run (disc/nn)_visualize.py to determine the selections as the FDR/FPR threshold varies and plot the FDR ROC curves.
3. Run gen_table.py (with -d for discontinuous and -i for independent) to generate the empirical FDR and TPR results presented in the appendix (along with additional results). See all_results.txt for the precomputed results.
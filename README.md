# non-CO2-greenhouse-gas-mitigation
## Project Overview


This repository contains the code accompanying the article and is used to analyze non-CO2 greenhouse gas emissions through three main components:

1. national/regional classification based on hierarchical clustering of multidimensional emission characteristics;
2. comparative evaluation and visualization of model performance;
3. SHAP-based interpretation of the drivers of non-CO2 greenhouse gas emissions.

The analysis covers CH₄, N₂O, and F-gases. The code in this repository is mainly intended to reproduce the analytical procedures and figures reported in the article.
## File Description
1. clusters.py

This script is used to classify countries/regions according to their non-CO2 greenhouse gas emission characteristics. It reads the relevant emission feature data, selects the annual mean emissions of CH₄, N₂O, and F-gases together with several categorical features, and then applies standardization, principal component analysis, and Ward’s hierarchical clustering. The script generates the clustering results and outputs the dendrogram, PCA scatter plot, and cluster heatmap.

2. fit.py

This script is used to compare model performance across different gases. It reads observed and predicted values from Excel workbooks, calculates R², MAE, and RMSE, and produces a multi-gas, multi-model comparison figure for assessing the performance of different models.

3. SHAPX.py

This script is used for the interpretation of F-gases emission drivers. It takes the corresponding dataset as input, performs feature engineering, variable screening, parameter optimization, model training, and SHAP analysis, and outputs SHAP summary plots and feature importance results at both the global and regional scales.

## Computational Environment

Python 3.9 or above is recommended.

## Methodological Description
Clustering Analysis

The clustering analysis in this study comprehensively considers three groups of characteristics: emission peaking characteristics (peaked gas type, whether emissions have peaked, and the number of peaked gases), emission level and dynamic characteristics (emission amount, emission intensity, relative ranking, and changing trend), and emission source structure characteristics (the primary emission source of CH₄, N₂O, and F-gases). Based on the similarity of these multidimensional characteristics, hierarchical clustering is conducted to identify differences in non-CO2 greenhouse gas emission patterns across countries/regions. The specific clustering assignment for each country/region is summarized in the appendix of the article.

## Model Evaluation

Model performance is evaluated mainly using R², MAE, and RMSE. A unified scatter-plot framework is employed to compare the fitting ability of different models for different gases.

Driver Interpretation Analysis

To investigate the drivers of non-CO2 greenhouse gas emissions, machine learning models are combined with SHAP analysis to identify key driving factors and their heterogeneous effects. The workflow publicly presented in this repository focuses on the SHAP-based analysis for F-gases.

## Additional Note

In addition to the F-gases SHAP driver analysis presented in this repository, separate driver interpretation analyses were also conducted for CH₄ and N₂O. Their overall technical framework is consistent with that of F-gases, namely, selecting the best-performing model for each gas-specific prediction task and then using SHAP to identify key driving factors and their differential effects. However, because the feature engineering, variable selection, and model specification differ across gases, the corresponding scripts were adjusted specifically for each case. Relevant materials can be obtained from the authors upon reasonable request.

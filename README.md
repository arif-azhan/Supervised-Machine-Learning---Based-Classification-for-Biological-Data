# Supervised Machine Learning - Based Classification for Biological Data

## Project Overview
This project focuses on the classification of **Drosophila (fruit flies) sex** using various supervised machine learning techniques. The dataset used consists of **wing traits and asymmetry measures** of different populations of Drosophila. The objective is to identify key features that differentiate male and female fruit flies while comparing the performance of multiple classifiers.

## Dataset
The dataset used in this study is **84_Loeschcke_et_al_2000_Wing_traits_&_asymmetry_lab_pops.csv**, which contains:
- **1731 entries**
- **16 columns**, including wing traits, asymmetry measures, and demographic information
- **Target variable:** `sex`
- Some columns contain missing values, which were imputed using the **mean imputation** technique.
- Irrelevant columns such as 'Latitude', 'Population', 'Longitude', 'Year_start', 'Year_end', 'Vial', and 'Replicate' were removed for a more focused analysis.

## Exploratory Data Analysis (EDA)
Key insights obtained from the EDA:
- The dataset is **balanced** (50.4% males, 49.6% females).
- **Correlation heatmap** shows that `Wing_area` (-0.45) and `Wing_shape` (-0.25) are the most relevant features for classification.
- **Boxplots** reveal that males tend to have larger wing areas and shapes compared to females.

## Machine Learning Models
### 1. K-Nearest Neighbors (k-NN)
- Normalized data used with **PCA for dimensionality reduction**.
- **Best k-value: 24** (determined using cross-validation).
- **Performance:**
  - Accuracy: **67.72%**
  - Precision: **68.16%**
  - Recall: **68.93%**
  - F1 Score: **68.54%**
- Confusion Matrix analysis showed moderate misclassification.
- PCA visualization confirmed overlapping feature values between sexes, leading to challenges in classification.

### 2. Logistic Regression
- Compared **regularized (L2) vs non-regularized** models.
- **Best regularization parameter (C): 0.001**.
- **Performance:**
  - Without regularization: **Accuracy = 70.89%**
  - With L2 regularization: **Accuracy = 71.47%**
  - Regularized model showed a slight improvement and better generalization to unseen data.

### 3. Random Forest Classifier
- Compared **PCA-transformed features vs Original Features**.
- **Feature importance analysis** showed that `Wing_area` and `Wing_shape` were the most significant predictors.
- **Best Hyperparameters:**
  - **PCA Model:** max_depth=10, max_features=log2, min_samples_split=5, n_estimators=50
  - **Original Features Model:** max_depth=None, max_features=sqrt, min_samples_split=5, n_estimators=100
- **Performance Comparison:**
  - **PCA Model:** Accuracy = **66.86%**
  - **Original Features Model:** Accuracy = **73.49%**
  - **Original Features model outperformed PCA model significantly.**

## Final Model Comparison
| Model                          | Accuracy | Precision | Recall  | F1 Score |
|--------------------------------|----------|------------|----------|----------|
| k-NN (k=24)                     | 67.72%  | 68.16%  | 68.93%  | 68.54%  |
| Logistic Regression             | 70.89%  | 71.11%  | 72.32%  | 71.71%  |
| Logistic Regression (L2)        | 71.47%  | 71.91%  | 72.32%  | 72.11%  |
| Random Forest (PCA)             | 66.86%  | 66.32%  | 71.19%  | 68.66%  |
| Random Forest (Original)        | **73.49%**  | **72.25%**  | **77.97%**  | **75.00%**  |

## Key Takeaways
- **Random Forest with original features performed best**, highlighting the importance of feature retention over dimensionality reduction for small datasets.
- **PCA did not improve results**, likely due to the small number of dimensions (6) already present in the dataset.
- **Wing traits (Wing_area & Wing_shape) are the most significant predictors of sex in Drosophila.**
- **Regularization in logistic regression improved generalization but had minimal impact on accuracy.**

## Future Improvements
- **Try k-NN and Logistic Regression without PCA** to analyze their performance without dimensionality reduction.
- **Use additional feature selection techniques** such as mutual information gain to refine model input features.
- **Explore deep learning models** like neural networks to see if they outperform traditional classifiers.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Supervised-Machine-Learning---Based-Classification-for-Biological-Data.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Supervised-Machine-Learning---Based-Classification-for-Biological-Data
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook to see the analysis and results:
   ```bash
   jupyter notebook
   ```

## Acknowledgements
- Dataset from **Loeschcke et al. (2000)**.
- Machine learning techniques applied with inspiration from various academic research papers.

---
Feel free to reach out for collaborations or suggestions!


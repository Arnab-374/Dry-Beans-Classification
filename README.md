# Dry Beans Classification Using 5 Classifiers  

**Author:** Arnab Bindu Laha   

**Github Link:** [https://github.com/Arnab-374/Dry-Beans-Classification](https://github.com/Arnab-374/Dry-Beans-Classification)  
**Streamlit App Link:** [https://dry-beans-classification.streamlit.app/](https://dry-beans-classification.streamlit.app/)  
  
**Problem Statement:**  
This repository contains a machine learning pipeline and a deployed Streamlit web application designed to classify 7 different varieties of dry beans based on 16 continuous geometric features.  

The project evaluates five distinct classification algorithms:  

- Logistic Regression

- Decision Tree

- K-Nearest Neighbors (KNN)

- Gaussian Naive Bayes

- Support Vector Machine (SVM)
  
**Dataset Description:**  
- **Dataset Link** - [https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset](https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset)  
- **Dataset Name** - Dry Bean Dataset  
- **Dataset Abstract** - Images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera. A total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains.  
- **Attribute Information** -  
  - Area (A): The area of a bean zone and the number of pixels within its boundaries.
  - Perimeter (P): Bean circumference is defined as the length of its border.
  - Major axis length (L): The distance between the ends of the longest line that can be drawn from a bean.
  - Minor axis length (l): The longest line that can be drawn from the bean while standing perpendicular to the main axis.
  - Aspect ratio (K): Defines the relationship between L and l.
  - Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.
  - Convex area (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.
  - Equivalent diameter (Ed): The diameter of a circle having the same area as a bean seed area.
  - Extent (Ex): The ratio of the pixels in the bounding box to the bean area.
  - Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.
  - Roundness (R): Calculated with the following formula: (4piA)/(P^2)
  - Compactness (CO): Measures the roundness of an object: Ed/L
  - ShapeFactor1 (SF1)
  - ShapeFactor2 (SF2)
  - ShapeFactor3 (SF3)
  - ShapeFactor4 (SF4)
  - Class (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira) -> *This is our target, that is what the models will try to predict*  
- **Source** -  
Murat KOKLU  
Faculty of Technology,  
Selcuk University,  
TURKEY.  
ORCID : 0000-0002-2737-2360  
mkoklu@selcuk.edu.tr  
  
  Ilker Ali OZKAN  
  Faculty of Technology,  
  Selcuk University,  
  TURKEY.  
  ORCID : 0000-0002-5715-1040  
  ilkerozkan@selcuk.edu.tr  
  
**Models Used:**  
* Make a Comparison Table with the evaluation metrics calculated for all the 5 models as below:

| Model | Accuracy | ROC AUC Score | Precision | Recall | F1 Score | Matthews Correlation Coefficient |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9207 | 0.9948 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| **Decision Tree** | 0.9045 | 0.9762 | 0.9042 | 0.9045 | 0.9042 | 0.8846 |
| **K-Nearest Neighbors** | 0.9166 | 0.9833 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| **Naive Bayes** | 0.8979 | 0.9916 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| **Support Vector Machine** | 0.9221 | 0.9949 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

* Observations on the performance of each model on the chosen dataset:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Showed strong performance with the second-highest accuracy (0.9207) and F1 score (0.9209). Using the scaled features allowed the gradient descent to converge effectively and separate the classes linearly. |
| **Decision Tree** | Yielded the lowest ROC AUC (0.9762) and second-lowest accuracy (0.9045). Even with depth constraints, it failed to properly classify the test cases. |
| **KNN** | Delivered a solid 0.9166 accuracy. Because the dataset features were properly normalized, the distance-based algorithm could accurately classify the beans based on nearest-neighbor proximity. |
| **Naive Bayes** | Recorded the lowest overall accuracy (0.8979) and MCC (0.8773). Its core assumption of feature independence restricted its performance, as physical dimensions like Area, Perimeter, and Convex Area are highly correlated in dry beans. |
| **Support Vector Machine** | Achieved the highest metrics across the board (Accuracy: 0.9221, ROC AUC: 0.9949). It proved to be the most effective at handling the multi-class overlapping boundaries in this high-dimensional space. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** narrowly outperformed Logistic Regression across all classification metrics, making it the most robust model for this dataset. |
  

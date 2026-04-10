# Problem Set 01: Chest X-Ray Pneumonia Detection using CNN

## 📌 Project Overview
This project aims to develop a Convolutional Neural Network (CNN) to classify chest X-ray images of pediatric patients into two categories: **Pneumonia** and **Normal**. The goal is to assist in the medical diagnosis process by accurately identifying lung infections from X-ray scans.

## 🛠️ Approach
Given the visual nature of the dataset (JPEG images), a Deep Learning approach using a **Convolutional Neural Network (CNN)** was selected. CNNs are the industry standard for computer vision tasks because they can automatically learn spatial hierarchies of features (like edges, textures, and lung opacities) directly from the pixel data. 

## ⚙️ Methodology
1. **Data Preprocessing & Augmentation:**
   * Images were resized to `150x150` pixels and pixel values were normalized to a `[0, 1]` range to help the neural network converge faster.
   * `ImageDataGenerator` was used to apply **Data Augmentation** to the training set. Techniques like shearing, zooming, and horizontal flipping were applied to artificially expand the dataset, introduce variability, and prevent the model from overfitting.
2. **Model Architecture:**
   * **Convolutional Layers:** 3 consecutive blocks of `Conv2D` and `MaxPooling2D` layers were used to extract spatial features and reduce dimensionality.
   * **Fully Connected Layers:** The output was flattened and passed into a `Dense` layer with 128 neurons. 
   * **Regularization:** A `Dropout` layer (0.5) was added to randomly deactivate neurons during training, which further prevents overfitting.
   * **Output Layer:** A single `Dense` node with a `sigmoid` activation function was used for the binary classification output.
3. **Compilation:** The model was compiled using the `adam` optimizer and `binary_crossentropy` loss function.

## 📊 Findings
* **Model Performance:** After training, the model achieved a final Test Accuracy of **[Insert Final Accuracy percentage here, e.g., 88.50%]**.
* **Loss & Accuracy Curves:** The plotted graphs for training and validation show the model's learning progression over the epochs. The validation loss remained relatively stable compared to the training loss, indicating that the dropout and augmentation strategies successfully mitigated severe overfitting.
* **Conclusion:** The CNN successfully learned the distinguishing features between normal and pneumonia-infected lungs, proving to be a viable foundational model for medical image classification.


# Problem Set 02: Bank Term Deposit Prediction

## 📌 Project Overview
This project involves building a predictive model for a banking institution to determine whether a customer will subscribe to a term deposit based on their demographic and banking behavior. The algorithm used for this binary classification problem is **Logistic Regression**.

## 🛠️ Approach
The objective is to predict a binary outcome ("yes" or "no"). Logistic Regression is a highly interpretable and efficient baseline model for binary classification tasks. Since the dataset contains a mix of numerical (age, balance) and categorical (job, marital status) attributes, strict data preprocessing was required before feeding it to the mathematical model.

## ⚙️ Methodology
1. **Data Preprocessing:**
   * **Target Variable Encoding:** The target column `y` ("yes"/"no") was transformed into binary values (`1` and `0`) using `LabelEncoder`.
   * **Handling Categorical Features:** Logistic Regression cannot process raw text. All categorical variables (e.g., job, education, contact type) were converted into numerical formats using **One-Hot Encoding** (`pd.get_dummies()`). The `drop_first=True` parameter was used to avoid the dummy variable trap (multicollinearity).
2. **Data Splitting:** The dataset was split into an 80% training set and a 20% testing set to ensure the model could be evaluated on unseen data.
3. **Feature Scaling:** Logistic Regression is sensitive to the scale of input features. A `StandardScaler` was applied to standardize the training and testing sets, ensuring features with larger magnitudes (like bank balance) did not disproportionately dominate the model's learning process.
4. **Model Training:** The `LogisticRegression` model was trained on the scaled dataset. Iterative loss tracking (Log Loss) was utilized to visualize the model's convergence.

## 📊 Findings
* **Overall Accuracy:** The Logistic Regression model achieved an accuracy of **[Insert Final Accuracy here, e.g., 89.20%]** on the test dataset.
* **Model Convergence:** The Log Loss curve demonstrates that the model's error decreased steadily over the iterations and converged successfully.
* **Classification Report & Confusion Matrix:** * Looking at the Confusion Matrix, the model is highly accurate at predicting the majority class (customers who did *not* subscribe).
   * **[Look at your classification report. If precision/recall for "1" is low, add this sentence:]** However, because bank datasets are typically imbalanced (more "no"s than "yes"s), the recall for actual subscribers is relatively lower, suggesting that while the overall accuracy is high, identifying positive leads remains challenging without techniques like SMOTE or class weighting.
 
   * 

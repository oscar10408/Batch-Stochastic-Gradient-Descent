# Polynomial Linear Regression

This project implements polynomial linear regression from scratch in Python using NumPy. It includes:

- Polynomial feature expansion
- Least squares objective calculation
- Batch gradient descent (BGD)
- Stochastic gradient descent (SGD)
- Visualizations and analysis in a Jupyter notebook

## 📁 Files

- `linear_regression.py`: Core implementation of data loading, feature generation, training objective, and optimization methods.
- `linear_regression.ipynb`: Demonstration and analysis of training using both BGD and SGD.

## 🧠 Algorithms Implemented

- **Polynomial Feature Generation**  
  Converts scalar input features into polynomial feature vectors.

- **Least Squares Objective**  
  Calculates the training loss based on squared error.

- **Gradient Descent Algorithms**  
  - **Batch Gradient Descent**: Uses the full dataset for each gradient update.
  - **Stochastic Gradient Descent**: Updates weights using one example at a time.

## 📊 Sample Use Case

This is a simple learning exercise for understanding how regression models work under the hood. You can adjust the degree of polynomial and learning rate to see how they affect training and convergence.

## 📌 Requirements

- Python 3.x
- NumPy
- Jupyter (for `.ipynb` usage)

## 🧑‍💻 Author

Created by Hao-Chun Shih (Oscar) as part of a regression algorithm study.

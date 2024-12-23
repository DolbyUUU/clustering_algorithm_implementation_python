# Clustering Algorithm Implementation and Visualization from Scratch with Python

## Overview

This project implements four popular clustering algorithms from scratch in Python, designed to work for datasets with `d >= 2` dimensions and `k >= 2` clusters. The implementations are tested on 2D datasets and compared visually with scikit-learn's implementations to evaluate correctness and performance.

### **Implemented Clustering Algorithms**
1. **K-Means Clustering**  
2. **Gaussian Mixture Model (GMM) using Expectation-Maximization (EM)**  
3. **Mean-Shift Clustering**  
4. **Agglomerative Clustering**

---

## Python Implementations

- **`KMeans.py`**: Implementation of the K-Means clustering algorithm.  
- **`KMeans_Ver0.py`**: A functional version of K-Means with modular methods.  
- **`GaussianMM.py`**: Implementation of GMM using EM.  
- **`GaussianMM_Ver0.py`**: An extended version with AIC, BIC, and prediction functions.  
- **`MeanShift.py`**: Implementation of Mean-Shift clustering.  
- **`Agglomerative.py`**: Implementation of Agglomerative clustering.

---

## Evaluations and Tests

- **`test_2d_visualization.py`**:  
  Tests each implementation on 2D datasets with visualization, comparing the results to scikit-learn's equivalent algorithms.  
- **`data_2d_test/`**:  
  Contains the datasets used for testing.  
- **`test_2d_visualization_results/`**:  
  Stores the output images of the clustering results.

---

## Visualization Results

### **Blobs Dataset**

| **Algorithm**       | **My Implementation**                                                                                     | **Scikit-learn**                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Agglomerative**    | <img src="test_2d_visualization_results/blobs_agglomerative_my_implementation.png" width="300">           | <img src="test_2d_visualization_results/blobs_agglomerative_scikit-learn.png" width="300">           |
| **EM-GMM**           | <img src="test_2d_visualization_results/blobs_em-gmm_my_implementation.png" width="300">                 | <img src="test_2d_visualization_results/blobs_em-gmm_scikit-learn.png" width="300">                 |
| **K-Means**          | <img src="test_2d_visualization_results/blobs_k-means_my_implementation.png" width="300">                | <img src="test_2d_visualization_results/blobs_k-means_scikit-learn.png" width="300">                |
| **Mean-Shift**       | <img src="test_2d_visualization_results/blobs_mean-shift_my_implementation.png" width="300">             | <img src="test_2d_visualization_results/blobs_mean-shift_bandwidth__4_58_scikit-learn.png" width="300"> |

---

### **Moons and Stars Dataset**

| **Algorithm**       | **My Implementation**                                                                                     | **Scikit-learn**                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Agglomerative**    | <img src="test_2d_visualization_results/moons_and_stars_agglomerative_my_implementation.png" width="300"> | <img src="test_2d_visualization_results/moons_and_stars_agglomerative_scikit-learn.png" width="300"> |
| **EM-GMM**           | <img src="test_2d_visualization_results/moons_and_stars_em-gmm_my_implementation.png" width="300">       | <img src="test_2d_visualization_results/moons_and_stars_em-gmm_scikit-learn.png" width="300">       |
| **K-Means**          | <img src="test_2d_visualization_results/moons_and_stars_k-means_my_implementation.png" width="300">      | <img src="test_2d_visualization_results/moons_and_stars_k-means_scikit-learn.png" width="300">      |
| **Mean-Shift**       | <img src="test_2d_visualization_results/moons_and_stars_mean-shift_my_implementation.png" width="300">   | <img src="test_2d_visualization_results/moons_and_stars_mean-shift_bandwidth__5_66_scikit-learn.png" width="300"> |

---

### **Sticks Dataset**

| **Algorithm**       | **My Implementation**                                                                                     | **Scikit-learn**                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Agglomerative**    | <img src="test_2d_visualization_results/sticks_agglomerative_my_implementation.png" width="300">          | <img src="test_2d_visualization_results/sticks_agglomerative_scikit-learn.png" width="300">          |
| **EM-GMM**           | <img src="test_2d_visualization_results/sticks_em-gmm_my_implementation.png" width="300">                | <img src="test_2d_visualization_results/sticks_em-gmm_scikit-learn.png" width="300">                |
| **K-Means**          | <img src="test_2d_visualization_results/sticks_k-means_my_implementation.png" width="300">               | <img src="test_2d_visualization_results/sticks_k-means_scikit-learn.png" width="300">               |
| **Mean-Shift**       | <img src="test_2d_visualization_results/sticks_mean-shift_my_implementation.png" width="300">            | <img src="test_2d_visualization_results/sticks_mean-shift_bandwidth__4_37_scikit-learn.png" width="300"> |

# Clustering algorithm implementaion and visualization from scratch with python

## Four popular clustering algorithms (for d >= 2 dimensions, k >= 2 clusters):  
- (1) k-means clustering  
- (2) Gaussian mixture model - expectation maximization algorithm (EM-GMM)  
- (3) mean-shift clustering   
- (4) agglomerative clustering 

## Python implementations:  
- `KMeans.py`: k-means clustering  
- `KMeans_Ver0.py`: second version of k-means implementation as a function  
- `GaussianMM.py`: EM-GMM
- `GaussianMM_Ver0.py`: second version of EM-GMM implementation with functions of AIC, BIC and predict  
- `MeanShift.py`: mean-shift clustering
- `Agglomerative`: agglomerative clustering

## Evaluations and tests:  
- `test_2d_visualization.py`: tests on 2D datasets with visualization, compared with sklearn implementation  
- `data_2d_test folder`: datasets for tests  
- `test_2d_visualization_results folder`: output images of tests  


## Visualization Results

The following figures compare the clustering results of my own implementations with those of scikit-learn's implementations. Each dataset is processed using different clustering algorithms.

---

### Blobs Dataset

| **Algorithm**       | **My Implementation**                                                                 | **Scikit-learn**                                                                 |
|----------------------|---------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Agglomerative**    | ![Blobs - Agglomerative (My)](test_2d_visualization_results/blobs_agglomerative_my_implementation.png) | ![Blobs - Agglomerative (Sklearn)](test_2d_visualization_results/blobs_agglomerative_scikit-learn.png) |
| **EM-GMM**           | ![Blobs - EM-GMM (My)](test_2d_visualization_results/blobs_em-gmm_my_implementation.png)               | ![Blobs - EM-GMM (Sklearn)](test_2d_visualization_results/blobs_em-gmm_scikit-learn.png)               |
| **K-Means**          | ![Blobs - K-Means (My)](test_2d_visualization_results/blobs_k-means_my_implementation.png)             | ![Blobs - K-Means (Sklearn)](test_2d_visualization_results/blobs_k-means_scikit-learn.png)             |
| **Mean-Shift**       | ![Blobs - Mean-Shift (My)](test_2d_visualization_results/blobs_mean-shift_my_implementation.png)       | ![Blobs - Mean-Shift (Sklearn)](test_2d_visualization_results/blobs_mean-shift_bandwidth__4_58_scikit-learn.png) |

---

### Moons and Stars Dataset

| **Algorithm**       | **My Implementation**                                                                 | **Scikit-learn**                                                                 |
|----------------------|---------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Agglomerative**    | ![Moons and Stars - Agglomerative (My)](test_2d_visualization_results/moons_and_stars_agglomerative_my_implementation.png) | ![Moons and Stars - Agglomerative (Sklearn)](test_2d_visualization_results/moons_and_stars_agglomerative_scikit-learn.png) |
| **EM-GMM**           | ![Moons and Stars - EM-GMM (My)](test_2d_visualization_results/moons_and_stars_em-gmm_my_implementation.png)               | ![Moons and Stars - EM-GMM (Sklearn)](test_2d_visualization_results/moons_and_stars_em-gmm_scikit-learn.png)               |
| **K-Means**          | ![Moons and Stars - K-Means (My)](test_2d_visualization_results/moons_and_stars_k-means_my_implementation.png)             | ![Moons and Stars - K-Means (Sklearn)](test_2d_visualization_results/moons_and_stars_k-means_scikit-learn.png)             |
| **Mean-Shift**       | ![Moons and Stars - Mean-Shift (My)](test_2d_visualization_results/moons_and_stars_mean-shift_my_implementation.png)       | ![Moons and Stars - Mean-Shift (Sklearn)](test_2d_visualization_results/moons_and_stars_mean-shift_bandwidth__5_66_scikit-learn.png) |

---

### Sticks Dataset

| **Algorithm**       | **My Implementation**                                                                 | **Scikit-learn**                                                                 |
|----------------------|---------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Agglomerative**    | ![Sticks - Agglomerative (My)](test_2d_visualization_results/sticks_agglomerative_my_implementation.png) | ![Sticks - Agglomerative (Sklearn)](test_2d_visualization_results/sticks_agglomerative_scikit-learn.png) |
| **EM-GMM**           | ![Sticks - EM-GMM (My)](test_2d_visualization_results/sticks_em-gmm_my_implementation.png)               | ![Sticks - EM-GMM (Sklearn)](test_2d_visualization_results/sticks_em-gmm_scikit-learn.png)               |
| **K-Means**          | ![Sticks - K-Means (My)](test_2d_visualization_results/sticks_k-means_my_implementation.png)             | ![Sticks - K-Means (Sklearn)](test_2d_visualization_results/sticks_k-means_scikit-learn.png)             |
| **Mean-Shift**       | ![Sticks - Mean-Shift (My)](test_2d_visualization_results/sticks_mean-shift_my_implementation.png)       | ![Sticks - Mean-Shift (Sklearn)](test_2d_visualization_results/sticks_mean-shift_bandwidth__4_37_scikit-learn.png) |

---

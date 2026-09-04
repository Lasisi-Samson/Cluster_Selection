# Cluster_Selection
A Bayesian Framework that selects optimal cluster count in a dataset with attached level of uncertainty.



Determining the appropriate number of clusters ($K$) is a critical challenge in unsupervised learning. Heuristic methods like the Elbow plot or Silhouette analysis often lack a formal statistical foundation and fail to quantify uncertainty. 

Combining a statistical background with machine learning principles, this repository implements a **Bayesian Model Selection Framework** that reformulates distance-based clustering evaluation into a probabilistic log-posterior scoring system.

---

## Key Methodology

### 1. Generative Error Model
The framework assumes that observed data points $x_i \in \mathbb{R}^p$ assigned to cluster $k$ are generated relative to their cluster center $\mu_k$ plus a random error term:

$$x_i = \mu_k + \varepsilon_i \quad \text{for } z_i = k$$


Assuming isotropic multivariate Gaussian errors $\varepsilon_i \sim \mathcal{N}(0, \sigma^2 I_p)$, within-cluster dispersion ($W_K$) maps directly to a standard **Gaussian Log-Likelihood**:

$$\log L(X \mid Z, \mu, \sigma^2) = N \left( - \frac{p}{2} \log(2\pi \sigma^2) \right) - \frac{1}{2\sigma^2} \sum_{i=1}^{N} \Vert{} x_i - \mu_{z_i} \Vert{}^2$$


### 2. Penalty via Bayesian Information Criterion (BIC)
Because standard likelihood monotonically increases as $K \to N$ (causing severe overfitting), the framework incorporates a Bayesian penalty to control model complexity:

$$\text{Log-Prior (BIC)} = -\frac{1}{2} (K \cdot p) \log(N)$$


Combining these yields a normalized **Log-Posterior Probability** score across candidate values of $K$:

$$\text{Log-Posterior} = \text{Log-Likelihood} + \text{Log-Prior}$$

---

## Experimental Validation

The evaluation framework was benchmarked across standard machine learning datasets:
* **Iris Dataset**
* **Wine Dataset**
* **Wholesale Customers Dataset**

Across evaluations on multiple underlying clustering algorithms—including **$K$-Means**, **Agglomerative Hierarchical Clustering**, and **Dirichlet Process Gaussian Mixture Models (DP-GMM)**—the framework provided robust, consistent cluster selection while quantifying model uncertainty via normalized posterior probabilities.

---

## Quickstart

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# 1. Load sample dataset
X, _ = load_iris(return_X_y=True)

# 2. Evaluate K-Means models from K=2 to K=6 using the Bayesian framework
from cluster_selection import evaluate_kmeans_models

results_df = evaluate_kmeans_models(X, k_range=range(2, 7), sigma2=1.0, prior_type="bic")
print(results_df.sort_values(by="posterior_prob", ascending=False))

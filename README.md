# Projects for Applied Computer Vision

## 1. MNIST Dataset Curation

**Notebooks:**
- `MNIST_CURATION_basic.ipynb`
- `MNIST_CURATION_Active_Learning.ipynb`

The **basic notebook** performs a one-time, fully automatic curation of the MNIST training set using predefined thresholds on model- and dataset-level metrics (e.g., hardness, mistakenness, uniqueness).  

The **active-learning notebook** extends this by adding an iterative loop in which the model is repeatedly retrained and then used to propose high-value samples (likely mislabels or non-digits/artifacts) for **manual review and correction**, leading to a progressively cleaner and more informative training set.

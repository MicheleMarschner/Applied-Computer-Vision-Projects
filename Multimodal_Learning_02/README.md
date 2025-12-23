# Multimodal Learning with RGB–LiDAR Fusion - Project 02
This project extends the NVIDIA DLI Multimodality Workshop by implementing and analyzing a complete multimodal learning pipeline for RGB–LiDAR data. It covers dataset exploration, fusion architecture design, ablation studies, contrastive pretraining, and final classifier evaluation.

The goal of this project is to systematically study how different RGB–LiDAR fusion strategies affect
classification performance on a controlled multimodal dataset (cubes vs. spheres).

---

## 1. Method Overview

The project follows a structured multimodal learning pipeline:

1. Dataset exploration and sanity checks (RGB, LiDAR, class balance)
2. Independent modality encoders for RGB and LiDAR
3. Fusion at different stages (early / intermediate / late)
4. Optional contrastive pretraining of modality encoders
5. Supervised classifier training
6. Quantitative evaluation and qualitative analysis

## 2. Setup Instructions (Colab or Local)

## Setup

It is recommended to run this project in **Google Colab**, as this ensures a consistent environment and reproducible evaluation.

### 1. Requirements

- Python 3.11 or higher
- PyTorch 2.0 or higher with CUDA support (or use Google Colab GPU runtime)
- Git
- Weights & Biases account (for experiment tracking)

---

### 2. Running in Google Colab (recommended)

1. Open Colab and enable GPU

   - `Runtime` → `Change runtime type` → `Hardware accelerator: GPU`.

2. Clone the repository

   ```python
   %cd "/content/drive/MyDrive" 
   !git clone https://github.com/MicheleMarschner/Applied-Computer-Vision-Projects.git
   %cd "/content/drive/MyDrive/Applied-Computer-Vision-Projects/Multimodal_Learning_02"                                
   ```

   **Important:** Don’t change the project location. The repository **must** be cloned into: `/content/drive/MyDrive`
   All notebooks assume this as the project root.

   The repo should have the following structure:
   ```text
   Applied-Computer-Vision-Projects/Multimodal_Learning_02/
   ├── notebooks/
   │   ├── 01_dataset_exploration.ipynb    # Task 2
   │   ├── 02_fusion_comparison.ipynb      # Task 3
   │   ├── 03_strided_conv_ablation.ipynb  # Task 4
   │   └── 04_final_assessment.ipynb       # Task 5
   │
   ├── src/
   │   ├── __init__.py
   │   ├── models.py          # All model architectures
   │   ├── datasets.py        # Dataset classes
   │   ├── training.py        # Training loops
   │   ├── utility.py         # Helper functions
   │   └── visualization.py   # Plotting utilities
   │
   ├── checkpoints/           # Saved model weights
   ├── results/               # Figures and tables
   ├── requirements.txt       # Dependencies
   └── README.md              # Setup and usage instructions
   ```
3. Install dependencies

   All necessary dependecies will be downloaded once you run the notebooks. 

4. Prepare the dataset                                                                             
   
   This project uses the assessment dataset provided by the course.
   To make the notebooks fully reproducible, please access it using the public link:
   [Dataset Download Link](https://drive.google.com/drive/folders/1sPoBLVY-ho4IolgCzszGU6xnz4uPW6Mu?usp=drive_link)
   (accessible to anyone with the link)

   Access the dataset and create a shortcut directly inside your repository folder: Multimodal_Learning_02/data
   `Right-click` → `Organize` → `Create shortcut`

   Creating a shortcut avoids duplicating the dataset while allowing the notebooks to access it via a fixed relative path.

   The dataset should have the following structure:
   ```text
   data/assessment/
      ├── cubes/
      │   ├── rgb/*.png
      │   └── lidar/*.npy
      └── spheres/
            ├── rgb/*.png
            └── lidar/*.npy
   ```
6. Set your Weights & Biases API key
   This project logs metrics to W&B. You must create an account and a project before running the notebooks.

   Store the secret in Colab Secrets by opening the left sidebar → “Secrets” → “Add new secret”
   Name: WANDB_API_KEY
   Value: your key from https://wandb.ai/authorize

   Load the key inside your notebook:
   ```python
   import os
   os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY")
   ```
   
   W&B will now authenticate automatically without exposing your key in the notebook.

7. Open and run the notebooks in the respective order and execute each cell


### 3. Run Locally
If you run this locally, adapt all Colab-specific paths and commands in the notebooks and config file (e.g. `/content/drive/MyDrive`, `drive.mount`, and `%cd`).

1. Run the following commands:
```bash
# Clone repository
git clone https://github.com/MicheleMarschner/Applied-Computer-Vision-Projects.git
cd Applied-Computer-Vision-Projects/Multimodal_Learning_02

# Create environment with conda (or another environment of your choice)
conda create -n [PROJECT] python=3.10
conda activate [PROJECT]
```

The repo should have the following structure:
   ```text
   Applied-Computer-Vision-Projects/Multimodal_Learning_02/
   ├── notebooks/
   │   ├── 01_dataset_exploration.ipynb    # Task 2
   │   ├── 02_fusion_comparison.ipynb      # Task 3
   │   ├── 03_strided_conv_ablation.ipynb  # Task 4
   │   └── 04_final_assessment.ipynb       # Task 5
   │
   ├── src/
   │   ├── __init__.py
   │   ├── models.py          # All model architectures
   │   ├── datasets.py        # Dataset classes
   │   ├── training.py        # Training loops
   │   ├── utility.py         # Helper functions
   │   └── visualization.py   # Plotting utilities
   │
   ├── checkpoints/           # Saved model weights
   ├── results/               # Figures and tables
   ├── requirements.txt       # Dependencies
   └── README.md              # Setup and usage instructions
   ```
2. Prepare the dataset                                                              
   
   This project uses the assessment dataset provided by the course.
   To make the notebooks fully reproducible, please access it using the public link:
   👉 [Dataset Download Link](https://drive.google.com/drive/folders/1sPoBLVY-ho4IolgCzszGU6xnz4uPW6Mu?usp=drive_link)
   (accessible to anyone with the link)

   Access the dataset, download it and place it inside the repository under: Multimodal_Learning_02/data

   The dataset should have the following structure:
   ```text
   data/assessment/
      ├── cubes/
      │   ├── rgb/*.png
      │   └── lidar/*.npy
      └── spheres/
            ├── rgb/*.png
            └── lidar/*.npy

   ```
4. Set your Weights & Biases API key                                                            
This project logs metrics to W&B. You must create an account and a project before running the notebooks.

macOS / Linux:
```bash
export WANDB_API_KEY="your-key-here"
```

Windows (PowerShell):
```bash
setx WANDB_API_KEY "your-key-here"
```

In Python, load it with:
```python
import os
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
```

4. Start Jupyter and open the notebooks
```bash
jupyter lab
```

5. Open and run the notebooks in the respective order and execute each cell

---

## 3. Weights & Biases Project Link
You can view all experiment runs, metrics, and logged artifacts in the public W&B project:

- **Project name:** `cilp-extended-assessment`
- **W&B username:** `michele-marschner-university-of-potsdam`
- 🔗 [Project Link](https://wandb.ai/michele-marschner-university-of-potsdam/cilp-extended-assessment)

---

## 4. Instructions to Reproduce Results
To reproduce all results shown in this project, first complete the steps in the Setup Instructions (which include installing dependencies, preparing the dataset, and setting the W&B API key).
Once the environment is ready, proceed as follows:

1. Run the notebooks in order
Execute each notebook from top to bottom without skipping cells:
notebooks/01_dataset_exploration.ipynb
notebooks/02_fusion_comparison.ipynb
notebooks/03_strided_conv_ablation.ipynb
notebooks/04_final_assessment.ipynb

Each notebook automatically:
- loads the dataset
- sets random seeds for reproducibility
- trains the corresponding model (unless you have pretrained models in the checkpoints folder) 
- logs metrics to Weights & Biases
- saves results to the checkpoints/ folder

2. Loading pretrained checkpoints (optional)
If you want to reproduce results quickly without retraining, you may load the saved models from this Drive Folder: [Drive Folder Link](https://drive.google.com/drive/folders/1c60n458cce9aY4K__lm68uMntOoPdCHH?usp=sharing)
Furthermore all final models are available as a W&B artifact ([Link](https://wandb.ai/michele-marschner-university-of-potsdam/cilp-extended-assessment/artifacts/model/multimodal_learning02-checkpoints/v0/files)) 


This allows you to skip training and directly run evaluation or visualization cells.

3. Normalization statistics (mean and standard deviation) are computed using 2,000 samples from the training split. These statistics are then applied to all training, validation, and test samples to avoid data leakage. For the experiments all training data and validation data has been used. There is a separate test set which could be used for further hyperparameter search. 

## 5. Limitations
- The dataset is small and synthetically generated, limiting real-world generalization
- Only simple CNN-based encoders are explored
- LiDAR is represented as dense projections rather than raw point clouds

## 6. Results
All notebooks contain the results (tables, observations and interpretation) in the Evaluation section of the respective notebook. 

Final results:
| Component | Metric | Requirement | Achieved |
|----------|--------|-------------|----------|
| CILP | Val loss | < 3.2 | 2.5598  | 
| Projector | Val MSE | < 2.5 | 1.2246 | 
| RGB→LiDAR | Val accuracy | > 95% | 96.25% |

## 7. Acknowledgements
All code in this repository was written by the author unless explicitly stated otherwise.

External resources were used for reference and conceptual guidance, including:
- NVIDIA DLI Multimodality Workshop materials (dataset structure and baseline ideas)
- PyTorch, FiftyOne and Weights & Biases official documentation

Coding assistance was provided by ChatGPT (OpenAI) for debugging support, code refactoring suggestions, and clarification of PyTorch and training concepts. All generated suggestions were reviewed, adapted, and integrated manually.

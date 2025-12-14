# Multimodal Learning with RGB–LiDAR Fusion - Project 02
This project extends the NVIDIA DLI Multimodality Workshop by implementing and analyzing a complete multimodal learning pipeline for RGB–LiDAR data. It covers dataset exploration, fusion architecture design, ablation studies, contrastive pretraining, and final classifier evaluation.

---

## 1. Setup Instructions (Colab or Local)

## Setup

You can run this project either in **Google Colab** (recommended) or **locally**.

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
   !git clone https://github.com/MicheleMarschner/Applied-Computer-Vision-Projects.git
   %cd ./Applied-Computer-Vision-Projects/Multimodal_Learning_02                                  
   ```

   The repo should have the following structure:
   
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

3. Install dependencies

   ```python
   !pip install -r requirements.txt
   ```

4. Prepare the dataset                                                                             
   
   This project uses the assessment dataset provided by the course.
   To make the notebooks fully reproducible, please access it using the public link:
   [Dataset Download Link](https://drive.google.com/drive/folders/1sPoBLVY-ho4IolgCzszGU6xnz4uPW6Mu?usp=drive_link)
   (accessible to anyone with the link)

   Access the dataset and create a shortcut directly inside your repository folder: Multimodal_Learning_02/data
   `Right-click` → `Organize` → `Create shortcut`

   The dataset should have the following structure:
   data/assessment/
      ├── cubes/
      │   ├── rgb/*.png
      │   └── lidar/*.npy
      └── spheres/
            ├── rgb/*.png
            └── lidar/*.npy

5. Set your Weights & Biases API key
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

6. Open and run the notebooks in the respective order and execute each cell


### 3. Run Locally

1. Run the following commands:
```bash
# Clone repository
git clone https://github.com/MicheleMarschner/Applied-Computer-Vision-Projects.git
cd Applied-Computer-Vision-Projects/Multimodal_Learning_02

# Create environment with conda (or another environment of your choice)
conda create -n [PROJECT] python=3.10
conda activate [PROJECT]

# Install dependencies
pip install -r requirements.txt
```

The repo should have the following structure:
   
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

2. Prepare the dataset                                                              
   
   This project uses the assessment dataset provided by the course.
   To make the notebooks fully reproducible, please access it using the public link:
   👉 [Dataset Download Link](https://drive.google.com/drive/folders/1sPoBLVY-ho4IolgCzszGU6xnz4uPW6Mu?usp=drive_link)
   (accessible to anyone with the link)

   Access the dataset, download it and place it inside the repository under: Multimodal_Learning_02/data

   The dataset should have the following structure:
   data/assessment/
      ├── cubes/
      │   ├── rgb/*.png
      │   └── lidar/*.npy
      └── spheres/
            ├── rgb/*.png
            └── lidar/*.npy


3. Set your Weights & Biases API key                                                            
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

## 2. Weights & Biases Project Link
You can view all experiment runs, metrics, and logged artifacts in the public W&B project:

- **Project name:** `cilp-extended-assessment`
- **W&B username:** `michele-marschner-university-of-potsdam`
- 🔗 **https://wandb.ai/michele-marschner-university-of-potsdam/cilp-extended-assessment**

---

## 3. Summary of Results

|index|Fusion Strategy|Avg Valid Loss|Best Valid Loss|Num of params|Avg time per epoch \(min:s\)|GPU Memory \(MB, max\)|
|---|---|---|---|---|---|---|
|0|Early Fusion|0\.0047|1\.2874e-06|8387990|10\.0998|497\.7852|
|1|Intermediate \(Concat\)|0\.0057|5\.0663e-07|16672374|15\.8507|672\.7407|
|2|Intermediate \(Multiplicative\)|0\.0069|1\.3186e-06|8480374|13\.6595|643\.0933|
|3|Intermediate \(Hadamard\)|0\.0023|1\.4230e-06|8480374|13\.2025|675\.4458|
|4|Intermediate \(Add\)|0\.0029|1\.0505e-07|8480374|13\.1532|707\.7983|
|5|Late Fusion|0\.0070|1\.5497e-07|16672374|15\.6503|833\.4009|



**Early Fusion:**
|index|Metric|MaxPool2d|Strided Conv|Difference \(Strided - MaxPool\)|
|---|---|---|---|---|
|0|Validation Loss \(best\)|5\.9752e-07|7\.1077e-07|1\.1325e-07|
|1|Parameters|8387990\.0|8387990\.0|0\.0|
|2|Training Time \(s\)|136\.3252|129\.5581|-6\.7671|
|3|Final Accuracy|1\.0|1\.0|0\.0|

**Intermediate Fusion (Concat):**
|index|Metric|MaxPool2d|Strided Conv|Difference \(Strided - MaxPool\)|
|---|---|---|---|---|
|0|Validation Loss \(best\)|1\.3411e-08|4\.7867e-06|4\.7733e-06|
|1|Parameters|16672374\.0|16672374\.0|0\.0|
|2|Training Time \(s\)|218\.2184|168\.5268|-49\.6916|
|3|Final Accuracy|1\.0|1\.0|0\.0|

**Late Fusion:**
|index|Metric|MaxPool2d|Strided Conv|Difference \(Strided - MaxPool\)|
|---|---|---|---|---|
|0|Validation Loss \(best\)|1\.2889e-07|5\.1322e-06|5\.0033e-06|
|1|Parameters|16672374\.0|16672374\.0|0\.0|
|2|Training Time \(s\)|217\.3971|169\.4651|-47\.9319|
|3|Final Accuracy|1\.0|1\.0|0\.0|

---

## 4. Instructions to Reproduce Results
To reproduce all results shown in this project, first complete the steps in
👉 Setup Instructions (which include installing dependencies, preparing the dataset, and setting the W&B API key).
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
- trains the corresponding model (unless you choose to load checkpoints)
- logs metrics to Weights & Biases
- saves results to the checkpoints/ folder

2. Loading pretrained checkpoints (optional)
If you want to reproduce results quickly without retraining, you may load the saved models from checkpoints/.

Add this in your Colab cell before training:
```python
model = YourModelClass(...)
model.load_state_dict(torch.load("checkpoints/model_name.pth"))
model.to(device)
model.eval()
```
This allows you to skip training and directly run evaluation or visualization cells.

# Denoising Probabilistic Diffusion Models - Project 03
This project evaluates a pretrained text-to-image diffusion model by generating images from text prompts and analyzing output quality as well as internal U-Net representations. The focus is on understanding how classifier-free guidance strength influences semantic alignment, realism, and diversity.

**Goal**
The goal is to:
* Generate images conditioned on text prompts using a diffusion model
* Evaluate generations with CLIP Score and FID
* Analyze intermediate U-Net embeddings using FiftyOne
* Track experiments with Weights & Biases
* (Bonus) Demonstrate uncertainty-aware prediction with an IDK classifier

---

## 1. Setup Instructions (Colab or Local)

## Setup

Running the project in **Google Colab** is recommended for reproducibility.

### 1. Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.0 (CUDA recommended)
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
   %cd "/content/drive/MyDrive/Applied-Computer-Vision-Projects/Diffusion_Model_03"                                
   ```

   **Important:** Don’t change the project location. The repository **must** be cloned into: `/content/drive/MyDrive`
   All notebooks assume this as the project root.

   The repo should have the following structure:

   Applied-Computer-Vision-Projects/Diffusion_Model_03/
    ├── checkpoints/                    # Pretrained models
    ├── results/                        # Saved figures, tables, and export
    ├── utils/
    │   ├── __init__.py
    │   ├── config.py                   # Constants and Paths
    │   ├── UNet_utils.py               # U-Net architecture definition
    │   ├── ddpm_utils.py               # DDPM forward & sampling utilities
    │   └── other_utils.py              # Helper functions (seeding, timestamps, etc.)
    │
    ├── Assignment_3.ipynb              # Main assignment notebook
    ├── Bonus.ipynb                     # Bonus: MNIST classifier with IDK option
    ├── README.md                       # Project documentation
    └── requirements.txt                # Python dependencies

3. Install dependencies

   All necessary dependencies will be downloaded once you run the notebooks. 

4. Prepare the dataset                                                                             
   The cropped flowers dataset provided by the course is used as the real-image reference for FID.
   To make the notebooks fully reproducible, please access it using the public link:
   [Dataset Download Link](https://drive.google.com/drive/folders/1vtKfOJgDGoSO8JCVes3wHAm9RXy_aQtU?usp=drive_link)
   (accessible to anyone with the link)

   Access the dataset and create a shortcut directly inside your repository folder: Diffusion_Model_03/data
   `Right-click` → `Organize` → `Create shortcut`

   Creating a shortcut avoids duplicating the dataset while allowing the notebooks to access it via a fixed relative path.

   The dataset should have the following structure:
   data/cropped_flowers/
    ├── daisy/
    ├── roses/
    └── sunflowers/

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
If you run this locally, adapt all Colab-specific paths and commands in the notebooks and config file (e.g. `/content/drive/MyDrive`, `drive.mount`, and `%cd`).

1. Run the following commands:
```bash
# Clone repository
git clone https://github.com/MicheleMarschner/Applied-Computer-Vision-Projects.git
cd Applied-Computer-Vision-Projects/Diffusion_Model_03

# Create environment with conda (or another environment of your choice)
conda create -n [PROJECT] python=3.10
conda activate [PROJECT]
```

The repo should have the following structure:
   
   Applied-Computer-Vision-Projects/Diffusion_Model_03/
    ├── checkpoints/                    # Pretrained models
    ├── results/                        # Saved figures, tables, and export
    ├── utils/
    │   ├── __init__.py
    │   ├── config.py                   # Constants and Paths
    │   ├── UNet_utils.py               # U-Net architecture definition
    │   ├── ddpm_utils.py               # DDPM forward & sampling utilities
    │   └── other_utils.py              # Helper functions (seeding, timestamps, etc.)
    │
    ├── Assignment_3.ipynb              # Main assignment notebook
    ├── Bonus.ipynb                     # Bonus: MNIST classifier with IDK option
    ├── README.md                       # Project documentation
    └── requirements.txt                # Python dependencies

2. Prepare the dataset                                                              
   The cropped flowers dataset provided by the course is used as the real-image reference for FID.
   To make the notebooks fully reproducible, please access it using the public link:
   👉 [Dataset Download Link](https://drive.google.com/drive/folders/1vtKfOJgDGoSO8JCVes3wHAm9RXy_aQtU?usp=drive_link)
   (accessible to anyone with the link)

   Access the dataset, download it and place it inside the repository under: Diffusion_Model_03/data

   The dataset should have the following structure:
   data/cropped_flowers/
    ├── daisy/
    ├── roses/
    └── sunflowers/


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

- **Project name:** `diffusion_model_assessment_v2`
- **W&B username:** `michele-marschner-university-of-potsdam`
- 🔗 [Project Link](https://wandb.ai/michele-marschner-university-of-potsdam/diffusion_model_assessment_v2)

---

## 3. Instructions to Reproduce Results
To reproduce all results shown in this project, first complete the steps in the Setup Instructions (which include installing dependencies, preparing the dataset, and setting the W&B API key).
Once the environment is ready, proceed as follows:

1. Execute each notebook from top to bottom without skipping cells:
    Assignment_3.ipynb
    Bonus.ipynb

Each notebook automatically:
- loads the dataset
- sets random seeds for reproducibility
- trains the corresponding model (unless you have pretrained models in the checkpoints folder) 
- logs metrics to Weights & Biases
- saves results to the checkpoints/ folder

2. Loading pretrained checkpoints (optional)
All pre-trained models are available as a W&B artifact ([Link](https://wandb.ai/michele-marschner-university-of-potsdam/diffusion_model_assessment_v2/runs/ps80y8lb)) 

## 4. Limitations
* Low image resolution limits realism
* Small sample size affects FID stability
* No cross-attention conditioning
* Results focus on analysis rather than photorealism

## 5. Results
All notebooks contain the results (tables, observations and interpretation) in the Evaluation section of the respective notebook. 

Overview of final results for Assignment 3:

| Metric                         | Result                         |
|--------------------------------|--------------------------------|
| Avg. CLIP score                | 0.212                          |
| Best guidance                  | w ≈ 2.0                        |
| FID                            | 320.5                          |


## 6. Acknowledgements
All code in this repository was written by the author unless explicitly stated otherwise.

External resources were used for reference and conceptual guidance, including:
- NVIDIA DLI Multimodality Workshop materials (dataset structure and baseline ideas)
- PyTorch, FiftyOne and Weights & Biases official documentation

Coding assistance was provided by ChatGPT (OpenAI) for debugging support, code refactoring suggestions, and clarification of PyTorch and training concepts. All generated suggestions were reviewed, adapted, and integrated manually.
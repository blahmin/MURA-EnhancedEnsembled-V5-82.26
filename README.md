# MURA-EnhancedEnsemble-V5-82.26

## Overview  
MURA-EnhancedEnsemble-V1 represents the culmination of extensive experimentation, training, and fine-tuning for fracture classification in musculoskeletal radiographs. This model is an **ensemble of three deep CNNs**, each using a **residual-enhanced architecture with attention mechanisms**. The ensemble dynamically adjusts its decision-making process based on the body part being classified, leveraging a specialized attention mechanism for areas that historically perform worse in classification tasks.

The model was trained over multiple days with **extensive data augmentation strategies**, including transformations such as **random rotations, horizontal flips, brightness and contrast adjustments, and affine distortions**. Different **loss functions, optimizers, learning rate schedules, and dropout rates** were tested to improve generalization.

## Model Performance  
- **Validation Accuracy:** 82.26%  
- **Cohen’s Kappa Score:** 0.6415  
- **Radiologist Benchmark (Reference):** 0.76  

## Model Architecture  
The **EnhancedEnsemble** consists of three **BaseModel** networks, each trained separately but later combined through a weighted decision-making process.  

### **1. BaseModel (Residual CNN with Attention)**
- Uses **EnhancedBlock** for **residual learning**, allowing deep feature extraction while preserving gradient flow.  
- Three main convolutional layers:
  - Each layer consists of residual blocks with **batch normalization**, **ReLU activations**, and **dropout regularization**.  
- **Attention mechanism** to highlight relevant spatial regions before classification.  
- **Adaptive average pooling**, ensuring robustness across variable input sizes.  
- **Final classification layer** for binary fracture detection.  

### **2. Ensemble Mechanism**
- Three **BaseModels** generate independent predictions.  
- Predictions are combined using **learnable weighting mechanisms** based on body part information.  
- Specialized **attention-based weighting layers** for complex body parts such as:
  - **Shoulders**
  - **Elbows**
- Other body parts use **default softmax-weighted averaging** for ensemble prediction.  

## Key Features  
- **Residual learning with skip connections** to improve training stability and convergence.  
- **Adaptive attention mechanism** to focus on important regions in feature maps.  
- **Dropout regularization** applied at multiple layers to enhance generalization.  
- **Body part-specific weighting mechanism** for ensemble predictions, improving classification on historically difficult regions.  
- **Extensive data augmentation** strategies to simulate real-world X-ray variability and mitigate overfitting.  

## Training Process  
Training included extensive experimentation with different hyperparameters and optimization techniques:  
- **Loss Function:** CrossEntropyLoss  
- **Optimizers Tested:** Adam, SGD with momentum, RMSprop  
- **Learning Rate Strategies:** Cosine Annealing, Step Decay, Cyclical Learning Rates  
- **Batch Sizes Tested:** 16, 32, 64  
- **Augmentation Techniques:**  
  - Random horizontal flips  
  - Random rotations  
  - Brightness and contrast adjustments  
  - Affine transformations  
  - Grid distortions  

## Installation 
Must download the Stanford MURA dataset and set up the train and val folder paths. 
Install dependencies with:  
pip install torch torchvision albumentations pillow

python
Copy
Edit

## Future Improvements
Further refine body-part specific weighting mechanisms for ensemble decision-making.
Investigate pretrained medical imaging backbones (DenseNet, EfficientNet) for transfer learning.
Apply class rebalancing techniques to further improve Cohen’s Kappa Score.
Implement Grad-CAM visualizations to interpret model predictions and validate feature importance.
Explore semi-supervised learning techniques to improve performance on ambiguous X-rays.
This model represents the most refined version of the work done on automated X-ray analysis for musculoskeletal fracture classification. Future iterations will focus on integrating advanced transfer learning and interpretability techniques to enhance reliability in clinical applications.

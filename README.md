# Phishing Website Detector

Developed by: Reginald Hingano
Based on work originally produced in CS412 – Artificial Intelligence (USP, 2025)

---

## Overview

This repository presents a complete phishing website detection system designed and developed to handle imbalanced datasets using generative and ensemble techniques.

The system uses a Markov Chain Monte Carlo-based Generative Adversarial Network (MCMC-GAN) to generate high-quality synthetic phishing data, which is then used to train a robust Stacking Ensemble model composed of multiple base classifiers.

While the idea originated as part of a group project for CS412 at the University of the South Pacific, the full system — including architecture, implementation, training logic, and augmentation method — was solely developed by Kokanati for independent research and further optimization.

---

## Features

- Custom-built MCMC-GAN (TensorFlow/Keras)
- Feature selection with SelectKBest (mutual information)
- Multi-model stacking with Logistic Regression, Random Forest, XGBoost, CatBoost
- Extensive evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- Organized, modular architecture for extensibility

---

## Project Structure

PhishingWebsite_Detector/
├── data/                  # Raw dataset (CSV format)  
├── logs/                  # Optional log outputs  
├── outputs/               # Balanced datasets, models, evaluations  
├── mcmc_gan/              # GAN modules  
│   ├── generator.py  
│   ├── discriminator.py  
│   └── mcmc_sampler.py  
├── utils/  
│   └── feature_selector.py  
├── train.py               # GAN training + data generation  
├── test.py                # Model training and evaluation  
├── config.py              # Hyperparameters and paths  
├── requirements.txt       # Project dependencies  
└── README.md              # This file  

---

## How to Use

1. Clone the Repository

```
git clone https://github.com/Kokanati/PhishingWebsite_Detector.git
cd PhishingWebsite_Detector
```

2. Create and Activate a Virtual Environment

```
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies

```
pip install -r requirements.txt
```

4. Add Your Dataset

Place your CSV dataset inside the `data/` folder and name it:

```
data/phishing_dataset.csv
```

This CSV must include a column named `status`, where:
- 0 = legitimate website
- 1 = phishing website

---

## Run the System

Step 1: Train the GAN and Balance the Dataset

```
python train.py
```

Step 2: Train Classifiers and Evaluate

```
python test.py
```

Results are saved in the `outputs/` directory.

---

## Evaluation Metrics

The system outputs:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

These are printed in the terminal and saved in `outputs/`.

---

## Academic Acknowledgement

This project evolved from coursework submitted in the CS412 Artificial Intelligence course at the University of the South Pacific (2025).  
However, the codebase, GAN training logic, data augmentation design, and model architecture were independently developed by Kokanati.

---

## License

This repository is intended for academic and research use only.  
© 2025 Kokanati. All rights reserved.

---

## Contributing

Contributions and feedback are welcome. Please fork this repository or open an issue to discuss improvements or enhancements.

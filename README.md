X-Ray Disease Detection using Deep Learning
This project uses DenseNet121 to classify chest X-ray images into three categories:

-Normal
-COVID-19
-Viral Pneumonia

It includes a trained model, a clean FastAPI-based web app, i deployed this model using Render.

Project Structure:
xray-deploy/
├── model/
│   └── best_densenet_model.pth      # Trained model
├── app/
│   ├── main.py                      # FastAPI backend
│   ├── predict.py                   # Prediction logic
│   └── templates/
│       └── index.html               # Frontend UI
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md

Features
Deep learning with transfer learning (DenseNet121)
Preprocessing and data augmentation
Trained with class weights for handling class imbalance
FastAPI backend for predictions
User-friendly frontend for image upload

result:
https://diesease-detection.onrender.com

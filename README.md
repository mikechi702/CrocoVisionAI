# CrocoVisionAI

## Project Description
Croco Vision AI is a simple image classification tool that classifies pictures of crocodilians into four categories: Alligator, Crocodile, Caiman, and Gharial. This was made as part of a class project for CSC 4444, Artificial Intelligence, at Louisiana State University.

## How to Run
1. Clone the GitHub repository on your local machine.
2. Set up your python environment
- (Optional): Create a virtual environment with venv.
3. Run <pip install -r requirements.txt> in your python environment to download libraries and dependencies
4. In the root project directory, run python app.py
5. In your browser, visit localhost:5000
6. Upload a file from your machine (.jpg or .png format) from the interface

## Technologies
- Python: Used as the main programming language to build the LLM.
- Flask: Framework and library for serving the frontend and handling file upload from the user. 
- TensorFlow: Framework for building and fine-tuning the LLM.
- HTML/CSS/Javascript: Used to create the UI/UX

## APIs
- Flickr: Used to download the training, validation, and testing set for the LLM . All images are from public domain.
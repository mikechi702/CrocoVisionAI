# CrocoVisionAI

## Project Description
Croco Vision AI is a simple image classification Convolutional Neural Network (CNN) that classifies pictures of crocodilians into four categories: Alligator, Crocodile, Caiman, and Gharial. This was made as part of a class project for CSC 4444, Artificial Intelligence, at Louisiana State University.

## How to Run
1. Clone the GitHub repository on your local machine.
2. Set up your python environment
- (Optional): Create a virtual environment with venv.
3. Run <pip install -r requirements.txt> in your python environment to download libraries and dependencies
4. In the root project directory, run python neuralnet.py
5. Wait for the neural network to load
6. When prompted to enter a file path, enter a file path of an image for the CNN (.jpg and .png formats supported) (example in <>'s: <**inputs/alligator-1.jpg**>)
 - A small popup window containing your input image and a bar graph of the class labels will appear.
 - Sample images can be found in the 'inputs' folder.
 - To exit the program, type 'exit' in the input loop

## Technologies
- Python: Used as the main programming language to build the Neural Net.
- Flask: Framework and library for serving the frontend and handling file upload from the user. 
- TensorFlow: Framework for building and fine-tuning the CNN.
- HTML/CSS/Javascript: Used to create the UI/UX

## APIs and External Resources
- Flickr: Used to download the training, validation, and testing set for the CNN. All images are from public domain.

## Notes
- CNN pipeline to the frontend is incomplete. Currently, it is only accessible by running <python neuralnet.py> in the terminal.
- The 'inputs' folder contains sample images you can use to manually test the CNN. These are not used anywhere in the training, validation, or testing sets within the CNN, and is meant to be a folder for the user to store their own images to test the Neural Network.
# Handwritten Text Recognition- Python Project (Team 33)

![](https://badgen.net/badge/Tools/Python-3.7&above/blue?icon=https://simpleicons.org/icons/python.svg&labelColor=cyan&label)
![](https://badgen.net/badge/Library/Pytorch/blue?icon=https://simpleicons.org/icons/pytorch.svg&labelColor=cyan&label)
![](https://badgen.net/badge/Library/PyQt5/blue?icon=&labelColor=cyan&label) ![](https://badgen.net/badge/Tools/numpy/blue?icon=https://upload.wikimedia.org/wikipedia/commons/1/1a/NumPy_logo.svg&labelColor=cyan&label)

## About

<p align="center">
  <img src="https://user-images.githubusercontent.com/73872595/115130888-699dbf00-a047-11eb-9d59-d6774f77f4df.png" alt="Team Logo"
	title="Logo" width="100" height="100" />
</p>

The aim of this project was to design and implement a tool that recognises handwritten digits. We have used Convolutional Neural Network(CNN) and MNIST datset.
The overall goal of this project is to get familiar with software design, GUI program and usage of basic machine learning skills.
This project was a part of the curriculum of “COMPSYS 302” offered by the University of Auckland.

## Installation

Use [MiniConda](https://docs.conda.io/en/latest/miniconda.html) to create a new environment.

```bash
conda create –n environment-name python=3.8
```

Activate the environment

```bash
conda activate environment-name
```

Download the Tools/Libraries needed

```bash
pip install PyQt5 torch torchvision matplotlib
```

The next steps need to be executed in the environment created.

## Usage

To use the Handwritten Digit recognition run the _"main.py"_ in scripts folder.
An interface similar to this image should pop up.

<p align="center">
  <img src="https://user-images.githubusercontent.com/73872595/115680587-ce3e7e00-a3a7-11eb-91d0-04319667ff62.png" alt="Main Window"
	title="Main Window" width="350" height="400" />
</p>

<hr>

You can draw any digit from 0-9 on the white canvas and click on _Recognize_ to see the predicted digit. Click on _Clear_ to clear the canvas.
The overall accuracy of predicting the digit is 97%.

<p align="center">
  <img src="https://user-images.githubusercontent.com/73872595/115680803-03e36700-a3a8-11eb-8587-a18b4a7d9ed3.png" alt="Main Window Prediction"
	title="Main Window Prediction" width="350" height="400" />
</p>

<hr>

To download MNIST dataset and train the model on the dataset, go to **File >> Train Model**.
_Download MNIST and Model Training_ dialog box should pop up.

<p align="center">
  <img src="https://user-images.githubusercontent.com/73872595/115681818-fc708d80-a3a8-11eb-8b76-fe308b767fb8.png"
	title="Training and Dataset Download Dialog" width="350" height="400" />
</p>
<hr>

Optional: Click on _Download MNIST_ to download the dataset and _Train_ to train the model.
Training the model can take upto 2-3mins.

<p align="center">
  <img src="https://user-images.githubusercontent.com/73872595/115685978-da790a00-a3ac-11eb-98e5-fa63d611c286.png" alt="Trained Model"
	title="Trained Model" width="350" height="400" />
</p>
<hr>

To view the training dataset the model was trained on, got ot **View >> View Training Images**. 
<p align="center">
  <img src="https://user-images.githubusercontent.com/73872595/115682336-799c0280-a3a9-11eb-9a9a-b22fe7777aea.png" alt="Training Images"
	title="Training Images" width="350" height="400" />
</p>
<hr>

To view the testing dataset that was used for testing and calculating the accuracy of the model, go to **View >> View Testing Images**.
<p align="center">
  <img src="https://user-images.githubusercontent.com/73872595/115683198-4f971000-a3aa-11eb-8ed7-48c9ec677a4d.png" alt="Testing Images"
	title="Testing Images" width="350" height="400" />
</p>
<hr>

## Conclusion and Future Works

Overall, this project gave us invaluable experience with Python, Pytorch, PyQt, and basics of Machine Learning.
In the future, we will expand the dataset to letters as well, so that the interface recognizes letters as well. And build a better model that can be more accurate.

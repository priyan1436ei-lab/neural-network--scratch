# Handwritten Digit Recognition AI

A handwritten-digit recognition project built with **Python, PyTorch, the MNIST dataset, and Gradio**. The project trains a neural network to classify grayscale images of handwritten digits from **0 to 9**, then uses the trained model in an interactive drawing app.

> **Project status:** The PyTorch training notebook has been run in Google Colab. The training log reported **468,202 model parameters** and used the CPU. Add your actual final test accuracy below after checking the completed training output. The interactive app works only when `app.py`, the matching model architecture, and the trained checkpoint are correctly connected.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Model and Results](#model-and-results)
- [Repository Structure](#repository-structure)
- [Run in Google Colab](#run-in-google-colab)
- [Run Locally on Windows](#run-locally-on-windows)
- [Interactive App Workflow](#interactive-app-workflow)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [GitHub Publishing](#github-publishing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

The goal of this project is to demonstrate an end-to-end machine-learning workflow:

1. Load the MNIST handwritten-digit dataset.
2. Prepare the images and labels for PyTorch.
3. Train a neural network to classify digits.
4. Evaluate the trained model on test data.
5. Save the best model checkpoint as `best_model.pt`.
6. Connect the saved model to a Gradio interface so a user can draw a digit and receive a prediction.

This project is suitable for learning the fundamentals of neural networks, image classification, model evaluation, and deploying a small interactive AI demo.

## Features

- **Digit classification:** Predicts one of ten classes, `0` through `9`.
- **PyTorch training:** Uses a neural network implemented with PyTorch.
- **Checkpoint saving:** Saves the model weights when test accuracy improves, according to the training script.
- **Interactive drawing interface:** The planned Gradio app lets users draw a digit with a mouse or touch input.
- **Prediction scores:** Shows class scores from the model's softmax output. These are model confidence scores, not guaranteed probabilities of correctness.
- **Colab support:** Training can be performed in Google Colab without setting up a local Python environment.

## How It Works

```text
MNIST dataset
     |
     v
Image preprocessing
     |
     v
PyTorch neural network
     |
     v
Training and evaluation
     |
     v
Save best_model.pt
     |
     v
Gradio drawing interface
     |
     v
Draw a digit -> preprocess image -> model prediction -> scores for 0-9
```

### Prediction pipeline

1. A user draws a digit on a canvas.
2. The drawing is converted to grayscale.
3. The digit is cropped and resized to the format expected by the model, commonly `28 × 28` pixels for MNIST.
4. Pixel values are scaled and normalized exactly as they were during training.
5. The image is converted into a PyTorch tensor and passed to the model.
6. The model produces class logits; softmax can convert these logits into a score for each digit.
7. The app displays the highest-scoring class and the scores for the other classes.

**Important:** Inference preprocessing must match the preprocessing used during training. If the training pipeline applies MNIST normalization, the app must apply the same normalization. The model class and layer dimensions in `model.py` must also match the model that created `best_model.pt`.

## Dataset

This project is designed for the **MNIST handwritten-digit dataset**, a standard image-classification dataset containing grayscale images of handwritten digits.

| Property | Description |
|---|---|
| Task | Multi-class image classification |
| Classes | 10 classes: digits `0`–`9` |
| Image format | Grayscale |
| Image size | `28 × 28` pixels in the standard MNIST representation |
| Training split | 60,000 images |
| Test split | 10,000 images |
| Typical label | Integer digit from `0` to `9` |

The training script can download the dataset through `torchvision.datasets.MNIST` when the relevant download option is enabled and the runtime has internet access.

## Technology Stack

- **Python** — programming language
- **PyTorch** — neural-network implementation, training, and inference
- **Torchvision** — MNIST dataset utilities and image transforms
- **NumPy** — numerical array operations, if used by the app
- **Pillow (PIL)** — image processing for the drawing input, if used by the app
- **Gradio** — interactive browser-based user interface
- **Google Colab / Jupyter Notebook** — development and training environment
- **Git and GitHub** — version control and project sharing

## Model and Results

The training output shared during development reported:

- **Reported parameter count:** `468,202`
- **Training device in that run:** CPU
- **Checkpoint filename:** `best_model.pt`
- **Dataset indicated by the download output:** MNIST, to be confirmed against the notebook's dataset-loading code

Complete this table using the output from your own finished training run:

| Metric | Result |
|---|---|
| Best test accuracy | `ADD_YOUR_ACTUAL_VALUE_HERE` |
| Final training loss | `ADD_YOUR_ACTUAL_VALUE_HERE` |
| Number of epochs | `ADD_YOUR_ACTUAL_VALUE_HERE` |
| Device used | CPU or GPU, as shown by the notebook |

Do not report an expected or illustrative accuracy as the measured result. Use the values printed by your completed notebook.

## Repository Structure

A clean project repository can use the following structure. Keep only files that actually exist in your project, and update the names if your notebook uses a different structure.

```text
mnist-digit-recognizer/
├── app.py                 # Gradio interface and inference function
├── model.py               # The exact neural-network architecture
├── best_model.pt          # Trained state_dict checkpoint (generated by training)
├── notebook.ipynb         # Colab/Jupyter training notebook
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Excludes caches and temporary files
```

### File details

- `app.py`: Loads the model, preprocesses the drawing, runs inference, and renders the Gradio interface.
- `model.py`: Contains the neural-network class. Its architecture must match the one used to train `best_model.pt`.
- `best_model.pt`: Contains saved model weights when the training code uses `torch.save(model.state_dict(), "best_model.pt")`.
- `notebook.ipynb`: Contains dataset loading, model definition, training loop, evaluation, and checkpoint saving.
- `requirements.txt`: Lists libraries needed to run the application locally.
- `.gitignore`: Prevents cache files and other unnecessary files from being committed.

If the model checkpoint is large or you prefer not to store model binaries in Git, consider GitHub Releases or a model-hosting service. Make sure the app has a documented way to obtain the checkpoint.

## Run in Google Colab

### 1. Open the notebook

Open your `.ipynb` file in Google Colab, or upload the notebook through **File → Upload notebook**.

### 2. Run the notebook cells

Run the cells in order so that the imports, dataset loaders, model definition, and training functions are available before training begins.

### 3. Train and save the checkpoint

The training loop should save the best weights. For example, this pattern saves weights when the measured accuracy improves:

```python
if test_acc > best_acc:
    best_acc = test_acc
    torch.save(model.state_dict(), "best_model.pt")
```

This snippet assumes `best_acc`, `test_acc`, and `model` have already been defined in the training loop.

### 4. Verify that the checkpoint exists

```python
import os
print("Checkpoint exists:", os.path.exists("best_model.pt"))
```

The expected output is `True`. If it prints `False`, check whether the training cell completed successfully and whether the file was saved to a different working directory.

### 5. Install Gradio if needed

```python
%pip install -q gradio
```

Continue with the app code only after confirming that the model definition, checkpoint, and preprocessing are compatible.

## Run Locally on Windows

These steps assume that Python is installed and available from Command Prompt or PowerShell.

### 1. Create or open the project folder

```powershell
mkdir mnist-digit-recognizer
cd mnist-digit-recognizer
```

If the folder already exists, open a terminal in that folder instead of creating it again.

### 2. Create a virtual environment

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation due to its execution policy, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

Create `requirements.txt` with the packages your code imports. A starting point is:

```text
torch
torchvision
gradio
numpy
Pillow
```

Then run:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

For GPU-enabled PyTorch, select the installation command appropriate for your operating system and CUDA setup from the official [PyTorch Get Started page](https://pytorch.org/get-started/locally/). CPU-only execution is also supported, although training may take longer.

### 4. Add the project files

Copy `app.py`, `model.py`, and `best_model.pt` into the project folder. Also include your training notebook if you want the training workflow to be reproducible.

### 5. Launch the app

```powershell
py app.py
```

Gradio normally prints a local URL in the terminal. Open that URL in a browser. A local run does not require a public share link.

## Interactive App Workflow

When `app.py` is implemented and connected to the trained checkpoint, users should be able to:

1. Open the Gradio page.
2. Draw one digit between `0` and `9` on the canvas.
3. Submit the drawing to the model.
4. View the top prediction and the scores for the ten classes.
5. Clear the canvas and test another digit.

For a public demo, the app can later be deployed to a suitable hosting platform. A Gradio share URL created from a temporary Colab runtime is intended for sharing during that running session; it is not a permanent hosting solution.

## Training the Model

A typical training workflow is:

1. Load the MNIST training and test datasets.
2. Apply the same image transformations consistently.
3. Create the neural-network model and move it to the selected device.
4. Choose a loss function and optimizer that are compatible with the model output and labels.
5. Iterate over training batches, compute loss, backpropagate gradients, and update model weights.
6. Evaluate on the test loader as implemented in the current notebook.
7. Save the best checkpoint according to the chosen metric.

The exact architecture, optimizer, learning rate, batch size, number of epochs, and transforms should be documented from the actual notebook rather than guessed. For a more rigorous experiment, use a separate validation set to select the checkpoint and reserve the test set for final evaluation.

## Evaluation

At minimum, evaluate the model using test accuracy and loss. Additional useful checks include:

- **Confusion matrix:** Shows which digits are frequently confused with one another.
- **Per-class precision and recall:** Helps examine performance separately for each digit.
- **Correct and incorrect examples:** Visually inspect predictions to identify common failure cases.
- **Training curves:** Plot loss and accuracy against epoch to check learning behaviour and possible overfitting.

Remember that a high score on MNIST does not guarantee equally strong performance on drawings from every user. The input canvas, handwriting style, stroke width, centering, and normalization can differ from the training images.

## Troubleshooting

### `Checkpoint exists: False`

The checkpoint may not have been created, the training cell may not have completed, or the notebook may be running in a different directory. Check the current directory with `os.getcwd()` and list files with `os.listdir()`.

### `FileNotFoundError: best_model.pt`

Place the checkpoint in the app's working directory, or update the checkpoint path in `app.py`. If using Colab, rerun the training/checkpoint cell or upload the checkpoint again after a runtime reset.

### `size mismatch` while loading the checkpoint

The model architecture in `model.py` does not match the architecture used when saving the weights. Use the same class definition and layer dimensions as the training notebook.

### Input shape error

A CNN typically expects a tensor shaped like `[batch, channels, height, width]`, for example `[1, 1, 28, 28]`. A fully connected model may expect a flattened vector. Adapt preprocessing to the actual architecture rather than assuming one input shape.

### Predictions are consistently wrong

Check that:

- The grayscale convention is correct (background and digit contrast match training).
- The image is resized and centered appropriately.
- Pixel scaling and normalization are identical to training.
- The model is in evaluation mode using `model.eval()`.
- Predictions run without gradient tracking, such as inside `torch.inference_mode()`.
- The output layer has ten classes and the label order is `0` through `9`.

### Gradio component or argument error

Gradio APIs can change between versions. Check the installed version with `python -c "import gradio; print(gradio.__version__)"` and adapt the component arguments to that version's official documentation.

### App link stops working

A Colab runtime may disconnect or stop. Run the app in an active runtime, or deploy it to a persistent hosting service for a longer-lived demo.

## Limitations and Future Improvements

Possible future improvements include:

- Show the top prediction more prominently and display all ten class scores.
- Add a button to clear the canvas and a preview of the processed `28 × 28` input.
- Add confusion-matrix and training-curve visualizations.
- Compare a fully connected network with a convolutional neural network (CNN).
- Use a validation split for checkpoint selection and only use the test set for final evaluation.
- Add unit tests for image preprocessing and model output shape.
- Package the app for deployment and document the hosting steps.
- Test with varied handwriting styles and record known failure cases.

## GitHub Publishing

1. Create a new GitHub repository, for example `mnist-digit-recognizer`.
2. Add `README.md`, the notebook, application code, model definition, dependency file, and other necessary assets.
3. Verify that no private credentials, API tokens, or unrelated personal files are included.
4. Add screenshots or a short GIF of the working app if available.
5. Confirm that setup instructions are reproducible on a fresh environment.

Example Git commands, run from inside the project folder:

```bash
git init
git add README.md app.py model.py requirements.txt notebook.ipynb .gitignore
git commit -m "Add handwritten digit recognition project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/mnist-digit-recognizer.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and change the filenames in `git add` to match the files that actually exist. If Git has already been initialized or `origin` already exists, do not repeat those commands blindly; inspect `git status` and `git remote -v` first.

### Suggested repository description

> Handwritten Digit Recognition using PyTorch and MNIST, with a Gradio drawing interface for interactive digit predictions.

## License

Choose a license before publishing. If you want others to reuse and modify your project, an OSI-approved license such as the MIT License is a common option. Add a `LICENSE` file only after deciding which license you want to apply. Confirm any dataset and dependency terms as well.

## Acknowledgements

- The MNIST handwritten-digit dataset and its maintainers.
- The PyTorch and Torchvision communities.
- The Gradio project for its Python-based interface framework.

---

**Maintainer:** Add your name or GitHub username here.  
**Project type:** Educational machine-learning project.  
**Last updated:** Update this date when you revise the project.

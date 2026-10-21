# 🧠 Neural Network from Scratch — XOR with NumPy

Build and understand a small artificial neural network **from scratch using NumPy**. This repository walks through the core mechanics of a feed-forward neural network and uses the classic **XOR problem** to demonstrate why a hidden layer with nonlinear activation can learn a relationship that a single linear neuron cannot.

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/priyan1436ei-lab/neural-network--scratch/blob/main/neural_network_from_scratch.ipynb)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [The XOR Problem](#-the-xor-problem)
- [Network Architecture](#-network-architecture)
- [How Learning Works](#-how-learning-works)
- [Technology Stack](#-technology-stack)
- [Getting Started](#-getting-started)
- [Run in Google Colab](#-run-in-google-colab)
- [Run Locally](#-run-locally)
- [Interactive Network Visualizer](#-interactive-network-visualizer)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Experiments](#-experiments-to-try)
- [Limitations and Future Improvements](#-limitations-and-future-improvements)
- [Learning Resources](#-learning-resources)
- [Author](#-author)
- [License](#-license)

## 🚀 Overview

Neural networks are made from layers of interconnected neurons. Each connection has a weight, each neuron can have a bias, and an activation function introduces nonlinearity. During training, the network changes its weights and biases to reduce prediction error.

This project implements these ideas with basic NumPy operations rather than relying on an automatic-differentiation framework. It is designed as a hands-on learning project for students who want to see what happens inside a neural network.

The main notebook includes:

1. An XOR dataset containing four binary input combinations.
2. A NumPy implementation with a forward pass.
3. Sigmoid activation and its derivative.
4. Mean squared error loss.
5. Backpropagation using the chain rule.
6. Gradient-descent parameter updates.
7. Predictions for all four XOR combinations.
8. Loss-curve plotting examples.
9. A slider-based interactive network visualization in Google Colab.
10. An additional reusable `NeuralNetwork` class with ReLU hidden activations and a sigmoid output.

## ✨ Features

- **NumPy-only core model:** Manually computes the forward and backward passes.
- **XOR classification:** Learns the mapping from two binary inputs to one binary output.
- **Manual backpropagation:** Calculates gradients without PyTorch or TensorFlow autograd.
- **Gradient descent:** Updates the weights and biases using the computed gradients.
- **Sigmoid activation:** Converts neuron outputs to values between 0 and 1.
- **Training-loss visualization:** Plots loss values over training epochs.
- **Interactive sliders:** Change input A and input B to observe the prediction.
- **Network diagram:** Displays the 2-input, 4-hidden-neuron, 1-output architecture, weights, and hidden activations.
- **Beginner-friendly notebook:** Combines code, experiments, and visual results in Google Colab.

## 🔢 The XOR Problem

XOR means **exclusive OR**. Its output is 1 when the two input bits are different and 0 when they are the same.

| Input A | Input B | Expected XOR |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A single linear neuron cannot correctly represent this pattern. A network with a hidden layer and a nonlinear activation function can learn it.

The project trains on these four input examples and then evaluates the network on the same truth table. Because this is a tiny teaching dataset, successfully predicting all four cases is a demonstration of learning the XOR function, not evidence of performance on a large real-world dataset.

## 🏗️ Network Architecture

The primary NumPy implementation uses this architecture:

```text
Input layer             Hidden layer            Output layer

   Input A ───────┐     ┌────────────┐
                  ├────▶│ Hidden 1   │───┐
   Input B ───────┤     │ Hidden 2   │   │
                  ├────▶│ Hidden 3   │───┼────▶ XOR prediction
                  └────▶│ Hidden 4   │───┘
                        └────────────┘

                   2 → 4 → 1
```

- **Input layer:** 2 values, each either 0 or 1.
- **Hidden layer:** 4 neurons with sigmoid activation in the primary implementation.
- **Output layer:** 1 sigmoid neuron representing the XOR output probability.
- **Weights and biases:** Initialized in NumPy and adjusted during training.

The notebook also contains another `NeuralNetwork` class that supports configurable layer sizes, ReLU hidden activations, sigmoid output, and cross-entropy loss. In that example, the architecture is `[2, 8, 1]`. These are separate demonstrations in the notebook; run the relevant cells in order and keep their model variables distinct if extending the code.

## ⚙️ How Learning Works

### 1. Forward propagation

Each layer calculates a weighted sum plus a bias:

```text
z = X @ W + b
a = activation(z)
```

Here, `X` is the input or previous layer's activation, `W` is a weight matrix, `b` is a bias, and `a` is the layer's activation.

### 2. Loss calculation

The primary implementation calculates mean squared error:

```python
loss = np.mean((y - a2) ** 2)
```

This measures the difference between expected labels `y` and output activations `a2`.

### 3. Backpropagation

The implementation computes gradients from the output layer toward the input layer using the chain rule. It calculates the impact of weights and biases on the loss.

For sigmoid activation, when the activation is already known as `a`, the derivative is:

```python
def sigmoid_derivative(a):
    return a * (1 - a)
```

### 4. Gradient descent

Parameters are updated in the direction that reduces the loss:

```python
W -= learning_rate * dW
b -= learning_rate * db
```

The main example uses a learning rate of `0.5` and runs for `10,000` epochs. These are configuration choices, and you can experiment with them.

### 5. Prediction

After training, the network computes a continuous output. A threshold of `0.5` can convert this output into a binary class:

```python
prediction = int(output[0, 0] >= 0.5)
```

## 🛠️ Technology Stack

| Technology | Use |
|---|---|
| Python | Main programming language |
| NumPy | Matrix operations and numerical computations |
| Matplotlib | Loss graph and network diagram |
| ipywidgets | Interactive input sliders |
| Google Colab / Jupyter Notebook | Notebook execution and visualization |
| Git and GitHub | Version control and publishing |

The primary network implementation is built with NumPy. PyTorch is not required for the XOR model itself.

## ✅ Getting Started

### Requirements

- Python 3.9+ recommended for a modern Python environment.
- NumPy.
- Matplotlib for plotting.
- ipywidgets for interactive sliders in notebook environments.
- Google Colab is the easiest option because the notebook already includes Colab-specific widget setup.

Install the packages locally:

```bash
python -m pip install numpy matplotlib ipywidgets
```

## ☁️ Run in Google Colab

1. Open the notebook using the badge at the top of this README.
2. In Google Colab, select **Runtime → Run all** or run the code cells in order.
3. The main training cell creates the XOR input data, initializes weights, trains the network, and prints losses during training.
4. The prediction cell prints the model outputs for all four input combinations.
5. Run the visualization cell after the relevant trained weights have been created.
6. Use the **Input A** and **Input B** sliders to change the values from 0 to 1. The visualization should update the activations, output probability, binary prediction, expected result, and connection weights.

### Important notebook execution note

The notebook contains multiple versions of the neural network. The slider visualizer requires the variables `W1`, `b1`, `W2`, and `b2` created by the primary 2→4→1 training cell. If you see an error saying those variables are missing, run the first training cell before running the visualizer.

The notebook also includes a separate PyTorch CNN-on-MNIST experiment. That is a different project workflow from the NumPy XOR implementation and uses a different architecture and dataset.

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/priyan1436ei-lab/neural-network--scratch.git
cd neural-network--scratch
```

### 2. Install dependencies

```bash
python -m pip install numpy matplotlib ipywidgets jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `neural_network_from_scratch.ipynb` in the Jupyter interface and run the cells in order.

> **Note:** The notebook contains imports from `google.colab` for its interactive Colab visualizer. Those Colab-specific lines will not work unchanged in a local Jupyter session. The core XOR training and prediction cells can run locally; adapt the widget setup for a local environment if you want the same interactive visualization.

## 🎛️ Interactive Network Visualizer

The Colab notebook includes sliders for the two XOR inputs and a diagram of the network.

The visualization shows:

- The values of input A and input B.
- The activation of each of the four hidden neurons.
- The output probability generated by the network.
- The thresholded XOR prediction.
- The expected XOR value.
- The current weights on connections between layers.
- A status indicating whether the prediction matches the truth table.

### How to use it

1. Run the primary XOR training cell first.
2. Run the interactive visualization cell.
3. Move either slider between 0 and 1.
4. Observe how hidden activations and the output change.
5. Test all four combinations: (0,0), (0,1), (1,0), and (1,1).

The diagram displays the trained model's weights. Its colors and line widths are visual cues for the sign and magnitude of weights; they are not a proof that a particular connection independently causes the final answer.

## 📊 Results

The notebook contains sample loss values used for a training-loss plot:

| Epoch | Recorded loss value |
|---:|---:|
| 0 | 0.283190 |
| 1,000 | 0.183325 |
| 2,000 | 0.020997 |
| 3,000 | 0.005533 |
| 4,000 | 0.002786 |
| 5,000 | 0.001768 |
| 6,000 | 0.001262 |
| 7,000 | 0.000966 |
| 8,000 | 0.000774 |
| 9,000 | 0.000642 |

These values are stored as a plotting example inside the notebook. For a fresh experiment, capture loss values from the actual training loop and plot those values rather than assuming these example values will be reproduced exactly.

### XOR evaluation checklist

After training, confirm that the rounded predictions match:

| Input | Expected class |
|---|---:|
| (0, 0) | 0 |
| (0, 1) | 1 |
| (1, 0) | 1 |
| (1, 1) | 0 |

If you report an accuracy percentage in a project presentation, calculate it from the four actual predictions and state that it is measured on the XOR truth table.

## 📁 Project Structure

The repository currently contains the notebook:

```text
neural-network--scratch/
├── README.md
└── neural_network_from_scratch.ipynb
```

The notebook includes multiple examples, training loops, graphing code, predictions, and an interactive network visualizer. You can split the notebook into separate Python modules later if you want a more structured project.

## 🐛 Troubleshooting

### Error: Missing trained model variables

**Message:** `Missing trained model variables: ['W1', 'b1', 'W2', 'b2']`

**Solution:** Run the primary XOR training cell that defines and trains `W1`, `b1`, `W2`, and `b2` before running the visualizer. If the runtime was reset, rerun the required setup and training cells.

### Error: `NameError: loss_history is not defined`

**Solution:** Define a list such as `loss_history = []` before training and append the calculated loss during each iteration, or run the plotting example cell that defines sample values.

For live loss tracking, add these lines to the main training cell:

```python
loss_history = []

for epoch in range(epochs):
    # Forward pass and loss calculation go here.
    # After calculating loss, store it:
    loss_history.append(float(loss))
    # Backpropagation and parameter updates follow.
```

Place the loss append inside your existing loop; do not replace the forward pass, backpropagation, or update logic with the comments above.

### Interactive controls do not appear

In Google Colab, run the widget setup cell and allow the custom widget manager when prompted. Then execute the visualization cell after training. In local Jupyter, remove or adapt imports from `google.colab` and configure ipywidgets for that environment.

### Predictions are incorrect

- Verify the training loop has completed.
- Confirm the labels follow the XOR truth table.
- Use the correct trained weights and biases.
- Ensure the hidden layer has a nonlinear activation.
- Try a different random seed, learning rate, or number of epochs.
- Test all four input combinations after training.

### Loss-plot label mismatch

The primary implementation uses **Mean Squared Error (MSE)** in its first training loop. The notebook's later plot should label the y-axis according to the loss values that were actually collected. The separate `NeuralNetwork` class uses binary cross-entropy loss.

## 🧪 Experiments to Try

- Change the hidden-layer size from 4 to 2, 8, or 16 neurons.
- Experiment with different learning rates.
- Change the number of training epochs.
- Compare sigmoid and ReLU hidden-layer activation functions.
- Plot actual loss values from the training loop.
- Display the hidden-layer activation values for every XOR input.
- Save weights and biases after training and load them in a separate prediction script.
- Compare the manual NumPy implementation against a small PyTorch equivalent.
- Add tests that assert the four rounded predictions match the XOR truth table.
- Improve the visualizer with a legend for connection weights and neuron activations.

Change one parameter at a time and record the effect so that results remain understandable.

## 🔭 Limitations and Future Improvements

This is a small educational example, not a general-purpose production model. It trains on only four XOR examples. The current notebook also contains several versions of a neural network, so variable names and cell execution order matter.

Useful improvements include:

- Refactor the notebook into clear modules for data, model, training, prediction, and visualization.
- Add automated tests for activation functions, gradients, and XOR predictions.
- Add a chart that plots the live loss history collected during training.
- Add a button or control to reset the visualizer.
- Add numerical gradient checks to verify backpropagation.
- Compare optimizers and initialization strategies.
- Write a separate local-Jupyter-compatible visualizer.
- Document the environment and dependencies with a `requirements.txt` file.

## 📚 Learning Resources

- [Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)
- [IPython Widgets Documentation](https://ipywidgets.readthedocs.io/)
- [Google Colab](https://colab.research.google.com/)

## 👨‍💻 Author

**Priyan A**

- GitHub: [@priyan1436ei-lab](https://github.com/priyan1436ei-lab)
- Repository: [neural-network--scratch](https://github.com/priyan1436ei-lab/neural-network--scratch)

## 📄 License

No license file is currently documented in this README. Add a `LICENSE` file after choosing the license you want to apply. Until then, do not assume others have permission to redistribute the code.

---

If you found this useful for learning neural networks, consider starring the repository ⭐ and trying the experiments above.

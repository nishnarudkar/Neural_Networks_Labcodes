# Neural Networks and Fuzzy Systems Lab Codes

This repository contains lab assignments for the **Neural Networks and Fuzzy Systems** course. The experiments cover fundamental neural network concepts, from basic models like McCulloch-Pitts neurons to multi-layer backpropagation and TensorFlow implementations.

---

## 🧪 Experiments Overview

A summary of the experiments included in this repository:

### **Experiment 1: McCulloch-Pitts (MP) Neuron Model**

- **Description:** Implements basic logic gates using the McCulloch-Pitts neuron model, demonstrating the foundational concept of a computational neuron.
- **Files:**
  - `Exp_1_Mcculloch Pitts/2-input_binary_AND_MP.ipynb`
  - `Exp_1_Mcculloch Pitts/2-input_binary_OR_MP.ipynb`
  - `Exp_1_Mcculloch Pitts/2-input_binary_NAND_MP.ipynb`
  - `Exp_1_Mcculloch Pitts/EXP 1 Bipolar.ipynb`

### **Experiment 2: Single-Layer Perceptron Learning**

- **Description:** Implements the Perceptron learning algorithm to train a single-layer neuron for linearly separable problems (logic gates).
- **Files:**
  - `Exp2_.../single_layer_perceprton_AND_Gate.ipynb`
  - `Exp2_.../single_layer_perceptron_OR_Gate.ipynb`
  - `Exp2_.../single_layer_perceptron_NAND.ipynb`
  - `Exp2_.../single_layer_perceptron_NOR_Gate.ipynb`

### **Experiment 3: Single-Layer Feedforward Network**

- **Description:** Demonstrates the forward pass computation in a single-layer network, including calculation of weighted sums and application of various activation functions (Linear, ReLU, Sigmoid, Bipolar Sigmoid).

### **Experiment 4: Multi-Layer Perceptron (Forward Pass)**

- **Description:** Implements the forward pass of a multi-layer perceptron, calculating activations from input through hidden layers to output.

### **Experiment 5: Single-Layer Backpropagation**

- **Description:** Implements the backpropagation algorithm (Delta rule) to train a single-layer network.

### **Experiment 6: Multi-Layer Backpropagation**

- **Description:** Generalizes the Delta rule with backpropagation for a multi-layer network, implemented from scratch using NumPy.

### **Experiment 7: Activation Functions & Derivatives**

- **Description:** Explores various activation functions (Sigmoid, ReLU, Leaky ReLU) and visualizes their derivatives using TensorFlow's `GradientTape` and Matplotlib.

### **Experiment 8: Backpropagation with TensorFlow**

- **Description:** Implements a 2-layer neural network for the AND gate using TensorFlow and automatic differentiation with `tf.GradientTape`.

---

## 🛠️ Technologies Used

- **Python 3**
- **Jupyter Notebook**
- **NumPy:** For numerical operations and custom network logic.
- **TensorFlow:** For automatic differentiation (`tf.GradientTape`) and modern neural network components.
- **Matplotlib:** For visualization of activation functions and derivatives.

---

## 🚀 How to Use

1. **Clone the repository:**
    ```bash
    git clone https://github.com/nishnarudkar/neural_networks_labcodes.git
    cd neural_networks_labcodes
    ```

2. **Install dependencies:**  
   *(It is recommended to use a virtual environment)*
    ```bash
    pip install -r requirements.txt
    ```
   *Note: If `requirements.txt` is missing, create one containing:*
   ```
   numpy
   tensorflow
   matplotlib
   ```

3. **Run Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

4. **Open experiments:**  
   Navigate to the desired experiment folder and open `.ipynb` files to view and run code.

---

## 📑 License

This repository is for educational purposes. Please cite or reference if used for academic work.

---

## 🙌 Contributing

Pull requests are welcome! For any suggestions, please open an issue or submit a PR.

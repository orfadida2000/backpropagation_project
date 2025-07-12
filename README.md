# Exercise 2 – General-Purpose Neural Network Framework

This project provides a reusable neural network framework implemented for Exercise 2 in the "Computational Models of Learning" course.       
The framework supports arbitrary input/output sizes, multiple layers, any activation function, and both scalar and vector-valued outputs.

---

## Project Structure

- `backpropagation_project/`  # Project root  
  - `neuralnet/`              # General-purpose neural network framework  
    - `__init__.py`  
    - `layers.py`  
    - `network.py`  
    - `training.py`  
    - `utilities.py`  
    - `visualization.py`  
    - `assertions/`  
      - `__init__.py`  
      - `layer_assertion.py`  
      - `network_assertion.py`  
      - `training_assertion.py`  
  - `scripts/` – Exercise-specific scripts  
    - `part1.py`  
    - `part2.py`  
  - `.gitignore`  
  - `requirements.txt`  
  - `setup.cfg`
  - `README.md`
  - `LICENSE`

---

## Features

- Supports arbitrary input and output dimensions  
- Custom activation functions per layer  
- Layer-wise modular assertions  
- Visualizations for model behavior  
- Separation of general-purpose code and exercise logic  

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

Always run from the project root (`backpropagation_project/`) using module mode:

```bash
python -m scripts.part1
python -m scripts.part2
```

This ensures all relative imports and packages are resolved correctly.

---

## Requirements

- Python ≥ 3.4  
- numpy  
- matplotlib

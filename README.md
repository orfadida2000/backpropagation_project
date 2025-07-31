# General-Purpose Neural Network Framework

This project provides a reusable neural network framework implemented for Exercise 2 in the **["Computational Models of Learning"](https://ims.tau.ac.il/Tal/Syllabus/Syllabus_L.aspx?lang=EN&course=1071871201&year=2023&req=4fc7d50f1d0af396d10b3c2ba3ca280781fb29f1ad6cc4943abf9dc495ab6c6a&caller=)** course.       
The framework supports arbitrary input/output sizes, multiple layers, any activation function, and both scalar and vector-valued outputs.

---

## 📁 Project Structure

```
backpropagation_project/           # Project root
├── neuralnet/                     # General-purpose neural network framework
│   ├── __init__.py
│   ├── layers.py
│   ├── network.py
│   ├── training.py
│   ├── utilities.py
│   ├── visualization.py
│   └── assertions/
│       ├── __init__.py
│       ├── layer_assertion.py
│       ├── network_assertion.py
│       └── training_assertion.py
├── scripts/
│   ├── part1.py
│   └── part2.py
├── .gitignore
├── LICENSE                         # MIT License
├── README.md   
├── requirements.txt                # Python dependencies
└── setup.cfg
```

---

## 📚 Features

- Supports arbitrary input and output dimensions  
- Custom activation functions per layer  
- Layer-wise modular assertions  
- Visualizations for model behavior  
- Separation of general-purpose code and exercise logic  

---

## 🛠 Setup

### 1. Create a virtual environment

#### Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ How to Run

Always run from the project root (**[backpropagation_project/](.)**) using module mode:

```bash
python -m scripts.part1
python -m scripts.part2
```

This ensures all relative imports and packages are resolved correctly.

---

### 📝 Important Notes

- Python ≥ 3.4  
- numpy  
- matplotlib


## 📄 License

MIT License.
See **[LICENSE](LICENSE)** for details.

## 👤 Author

- **Name:** Or Fadida
- **Email:** [orfadida@mail.tau.ac.il](mailto:orfadida@mail.tau.ac.il)
- **GitHub:** [orfadida2000](https://github.com/orfadida2000)

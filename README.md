# 🔥 Dynamic & Static Asymptotic Analysis Package 🔥

A powerful package for **Teacher-Student model** analysis using **statistical mechanics**! 📊

⚠️ *Work in Progress... Stay tuned!* 🚀

---

## 📌 Requirements

This package was implemented using **Python 3.11.11**.
To install dependencies, run:

```bash
pip install -r requirements.txt
```

### 📦 Dependencies:
✅ **DGL** → `2.1.0`  
✅ **Torch** → `2.4.0`  
✅ **NumPy** → `1.26.4`  
✅ **Pandas** → `2.2.2`  
✅ **Matplotlib** → `3.10.0`  
✅ **Seaborn** → `0.13.2`  
✅ **Scikit-Learn** → `1.6.1`  
✅ **NetworkX** → `3.4.2`  
✅ **TQDM** → `4.67.1`  

---

## 🚀 How to Use

### **Step 1: Define a Dataset Class**

```python
class Dataset:

    def __init__(self, ..., device="cpu"):
        # Define dataset attributes such as input dimensions, sparsity, and device

    def generate_sample(self):
        # Method to generate a single data sample
        return x, y

    def generate_dataset(self, n_samples):
        # Create a dataset using multiple samples
        X_dataset = []
        y_dataset = []

        for _ in range(n_samples):
            X, y, c = self.generate_sample()
            X_dataset.append(X.T)
            y_dataset.append(y)

        X_dataset = torch.vstack(X_dataset)
        y_dataset = torch.tensor(y_dataset, device=self.device)

        return X_dataset, y_dataset
```

---

### **Step 2: Define the Model**

Example: **Linear Regression**

```python
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, d=100):
        super(LinearRegression, self).__init__()
        
        # Define attributes
        self.d = d
        self.W = nn.Parameter(torch.randn(d, 1))

    def forward(self, X):
        # Model prediction: (n, d) -> (n,)
        return (self.W.T @ X.T / torch.sqrt(torch.tensor(self.d, dtype=torch.float32))).flatten()
```

---

### **Step 3: Compute Order Parameters**

This function calculates order parameters using the dataset and model objects.

```python
def calc_orderparams(data_generator, model):
    d = data_generator.d
    W0 = data_generator.W0
    h = data_generator.h
    rho = data_generator.rho
    eta = data_generator.eta
    
    w = model.W
    
    m = (1 / d) * (w.T @ W0).item()
    q = (1 / d) * (w.T @ w).item()
    
    eg = 0.5 * ((torch.sqrt(torch.tensor(rho)) * m - h.T @ h).item() ** 2 + eta * q)
    
    return [m, q, eg]
```

---

### **Step 4: Define the Loss Function**

Example: **Ridge Regression** (for both Online SGD and Static Analysis)

```python
def loss_function(y_pred, y, model, reg_param=1.0, online=True):
    mse_loss = torch.sum((y_pred - y) ** 2)
    
    if online:
        reg = torch.trace(model.W.T @ model.W) / model.W.size(0)
    else:
        reg = torch.trace(model.W.T @ model.W)
    
    return mse_loss + reg_param * reg
```

---

### **Step 5: Visualizing Results** 📊

Below is an example of visualizing **static analysis learning limits**.

```python
model_class = RidgeRegression
data_generator = DataSet(..., device=device)

alpha_values = [0.25 * i for i in range(1, 20)]

all_trials_orderparams = main.run_experiments_alphas_seeds(
    data_generator,
    model_class,
    loss_function,
    calc_orderparams,
    alpha_values,
    reg_param=0.01,
    lr=0.01,
    tol=1e-6,
    max_iter=50000,
    patience=100,
    verbose_interval=5000,
    num_trials=5,
    seed_list=None,
    visualize=True
)
```

---

🚀 **Happy Researching!** 🚀


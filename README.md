#Introduction to Neural Networks

## Course Overview

Welcome to the **Introduction to Neural Networks** course. This comprehensive course provides a solid foundation in deep learning and neural networks. You will learn the underlying mathematics, implement key algorithms like gradient descent and backpropagation, and build your own image classifier using Python, NumPy, and TensorFlow/PyTorch.

**Instructor**: Luis (former Machine Learning Engineer at Google, PhD in Mathematics from University of Michigan)

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Tools & Environment](#tools--environment)
3. [Course Outline](#course-outline)
4. [Core Concepts & Mathematics](#core-concepts--mathematics)
5. [Quiz Questions & Answers](#quiz-questions--answers)
6. [Code Implementations](#code-implementations)
7. [Key Formulas Reference](#key-formulas-reference)
8. [Learning Outcomes](#learning-outcomes)

---

## Prerequisites

To ensure success in this course, you should ideally have the following knowledge:

### Python Programming (Intermediate Level)
- At least 40 hours of programming experience
- Familiarity with data structures: dictionaries, lists, tuples, sets
- Experience with **NumPy** (array operations, broadcasting)
- Experience with **pandas** (DataFrames, data manipulation)

### Mathematics Requirements

#### Probability & Statistics
- Calculating probability of events
- Conditional probability
- Mean, variance, and standard deviation of distributions
- Random variables (discrete and continuous)

#### Linear Algebra
- Vectors and vector operations
- Matrices and matrix multiplication
- Dot products
- Linear transformations

#### Calculus
- Derivatives and partial derivatives
- Chain rule
- Multivariable calculus basics
- Gradient of a function

### Resource Recommendations

If you need a refresher, use these resources:

| Topic | Resource |
|-------|----------|
| Linear Algebra | [Khan Academy Linear Algebra](https://www.khanacademy.org/math/linear-algebra) |
| Multivariable Calculus | [Khan Academy Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus) |
| Matrices | [Khan Academy Introduction to Matrices](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices) |
| Python Programming | [Udacity Intro to Python](https://www.udacity.com/course/introduction-to-python-programming--ud1110) |

---

## Tools & Environment

### Jupyter Notebooks

Jupyter Notebooks are web applications that combine:
- **Code cells**: Write and execute Python code (run with `Shift + Enter`)
- **Markdown cells**: Formatted text, images, equations using Markdown syntax
- **Visualizations**: Inline plots and charts

### Workspaces

All notebooks run inside **Udacity Workspaces** - no local installation required.

> ⚠️ **Critical Notes**:
> - Workspaces shut down after **30 minutes of inactivity**
> - Running processes stop when workspace shuts down
> - Workspaces may take up to 5 minutes to start
> - Use **Menu button > Reset Data** to restore original content (⚠️ deletes unsaved code)
> - Save work locally: `File > Download as > Notebook (.ipynb)`

---

## Course Outline

### Lesson 1: Introduction to Neural Networks

| Topic | Description |
|-------|-------------|
| Classification Problems | Spam detection, stock forecasting, object recognition, medical diagnosis |
| Linear Classification | Finding boundaries that separate data classes |
| Perceptrons | Building blocks of neural networks (nodes + weighted edges) |
| Logical Operators | AND, OR, NOT, XOR implemented as perceptrons |
| Perceptron Algorithm | Iterative algorithm to find optimal separating line |
| Error Functions | Measuring model performance (discrete vs. continuous) |
| Gradient Descent | Optimization algorithm to minimize error |
| Sigmoid Function | Converting discrete predictions to continuous probabilities |
| Softmax | Multi-class probability distribution |
| One-Hot Encoding | Converting categorical data to numerical |
| Maximum Likelihood | Probability-based model evaluation |
| Cross-Entropy | Error function for classification |
| Logistic Regression | Complete classification algorithm |
| Backpropagation | Training neural networks via chain rule |

### Lesson 2: Implementing Gradient Descent

- Different error functions (Mean Squared Error, Cross-Entropy)
- Gradient descent implementation using NumPy
- Matrix multiplication for efficient computation

### Lesson 3: Training Neural Networks

- Optimization techniques for better training
- Handling overfitting and underfitting
- Hyperparameter tuning
- Validation strategies

### Lesson 4: Deep Learning with TensorFlow

- TensorFlow basics
- Building deep learning models
- Image classification pipelines

### Lesson 5: Project - Create Your Own Image Classifier

- Train on CIFAR-10 dataset
- Evaluate against benchmark models
- Build complete image classification application

---

## Core Concepts & Mathematics

### 1. The Linear Model

For a student admission problem with test scores (`x₁`) and grades (`x₂`):

```
2·x₁ + x₂ - 18 = score
```

**Decision Rule**:
- If `score ≥ 0` → Predict **Accepted** (ŷ = 1)
- If `score < 0` → Predict **Rejected** (ŷ = 0)

### 2. General Linear Equation

```
Wx + b = 0
```

Where:
- **W** = weight vector `(w₁, w₂, ..., wₙ)`
- **x** = input feature vector `(x₁, x₂, ..., xₙ)`
- **b** = bias (scalar)

**Dimension Requirements**:
- `W`: `(1 × n)` row vector
- `x`: `(n × 1)` column vector
- `b`: `(1 × 1)` scalar

### 3. Perceptron Structure

```
Inputs: x₁, x₂, ..., xₙ
Weights: w₁, w₂, ..., wₙ
Bias: b

Output = step_function(∑(wᵢ·xᵢ) + b)

Where step_function(x) = 1 if x ≥ 0, else 0
```

### 4. Sigmoid Function (Logistic Function)

```
σ(x) = 1 / (1 + e⁻ˣ)
```

**Properties**:
- Output range: (0, 1)
- σ(0) = 0.5 (50% probability)
- σ(∞) → 1
- σ(-∞) → 0

**Derivative** (critical for backpropagation):
```
σ'(x) = σ(x) · (1 - σ(x))
```

### 5. Softmax Function

For scores `Z₁, Z₂, ..., Zₙ`:

```
P(class i) = e^(Zᵢ) / (e^(Z₁) + e^(Z₂) + ... + e^(Zₙ))
```

**Properties**:
- All probabilities between 0 and 1
- Sum of all probabilities = 1
- Uses exponential to ensure positivity

### 6. Cross-Entropy Error Function

**Binary Classification** (two classes):
```
E = -[y·ln(p) + (1-y)·ln(1-p)]
```

Where:
- `y` = true label (0 or 1)
- `p` = predicted probability

**Multi-Class Classification**:
```
E = -∑∑ yᵢⱼ·ln(pᵢⱼ)
```

### 7. Gradient Descent Update Rules

For a point with coordinates `(x₁, ..., xₙ)`, label `y`, prediction `ŷ`:

```
wᵢ ← wᵢ + α·(y - ŷ)·xᵢ
b  ← b  + α·(y - ŷ)
```

Where `α` = learning rate (small positive number, e.g., 0.1)

**Gradient of Error Function**:
```
∇E = -(y - ŷ)·(x₁, ..., xₙ, 1)
```

### 8. Feedforward for Multi-Layer Network

For a 2-layer network:

```
h = σ(W⁽¹⁾·x + b⁽¹⁾)
ŷ = σ(W⁽²⁾·h + b⁽²⁾)
```

Where:
- `W⁽¹⁾` = first layer weights
- `b⁽¹⁾` = first layer bias
- `W⁽²⁾` = second layer weights
- `b⁽²⁾` = second layer bias

---

## Quiz Questions & Answers

### Quiz Set 1: Student Admission Prediction

**Question**: Given score from `2x₁ + x₂ - 18` for student with test=7, grades=6:
- Calculation: `2(7) + 6 - 18 = 14 + 6 - 18 = 2`
- Since `2 ≥ 0`, prediction is **Accepted** (ŷ = 1)

**Answer**: `Wx + b ≥ 0, so ŷ = 1 and we predict the student will be accepted`

---

### Quiz Set 2: Dimension Requirements

**Question**: Dimensions for `Wx + b` from the data table?

**Correct Answer**: `W: (1×n), x: (n×1), b: (1×1)`

**Explanation**: 
- `x` must be column vector for matrix multiplication
- `W` must be row vector with matching inner dimension
- `b` is scalar bias term

---

### Quiz Set 3: Modified Weights

**Question**: If `w₁` changes from 2 to 1.5, with test=7, grades=6:
- New score: `1.5(7) + 6 - 18 = 10.5 + 6 - 18 = -1.5`
- Since `-1.5 < 0`, prediction is **Rejected**

**Answer**: Rejected

---

### Quiz Set 4: AND → OR Perceptron

**Question**: Two ways to convert AND to OR perceptron?

**Answers**:
- ✅ Increase the weights
- ✅ Decrease the magnitude of the bias

**Explanation**: OR requires a larger positive region, achieved by increasing weights (making line steeper) OR decreasing bias magnitude (shifting line down).

---

### Quiz Set 5: XOR Neural Network

**Question**: Set operations for perceptrons A, B, C in XOR network?

**Correct Matches**:
- **A**: AND
- **B**: OR
- **C**: NOT

**Explanation**: XOR = (A AND NOT B) OR (NOT A AND B). The two-layer network combines NAND, AND, and OR.

---

### Quiz Set 6: Gradient Descent Requirements

**Question**: Conditions to apply gradient descent?

**Answers** (check all that apply):
- ✅ The error function should be differentiable
- ✅ The error function should be continuous

**Why not discrete?** A discrete error function (like counting misclassifications) doesn't provide gradient information - small changes may not change the error value.

---

### Quiz Set 7: Sigmoid 50% Probability

**Question**: Which points have exactly 50% probability for `score = 4x₁ + 5x₂ - 9`?

**Solution**: 50% probability occurs when `score = 0` (sigmoid(0) = 0.5)

Test each point:
- `(1, 1)`: `4(1) + 5(1) - 9 = 4 + 5 - 9 = 0` ✅
- `(2, 4)`: `8 + 20 - 9 = 19` ❌
- `(5, -5)`: `20 - 25 - 9 = -14` ❌
- `(-4, 5)`: `-16 + 25 - 9 = 0` ✅

**Answers**: `(1, 1)` and `(-4, 5)`

---

### Quiz Set 8: Positive Number Function

**Question**: What function turns every number into a positive number?

**Answer**: **exp** (exponential function)

**Explanation**: `eˣ > 0` for all real x. Used in softmax to ensure non-negative scores.

---

### Quiz Set 9: Maximum Likelihood Interpretation

**Question**: True statement for a very high value of `P(all)`?

**Answer**: The model classifies all points correctly.

**Explanation**: `P(all)` is product of individual probabilities. Maximum value (1.0) occurs only when every point is classified with 100% certainty.

---

### Quiz Set 10: Product to Sum Function

**Question**: What function turns products into sums?

**Answer**: **log** (logarithm)

**Property**: `log(a·b) = log(a) + log(b)`

This is why cross-entropy uses logarithms - converting product of probabilities into sum of log-probabilities.

---

### Quiz Set 11: Cross-Entropy Interpretation

**Question**: Relationship between cross-entropy and probability?

**Answer**: A higher cross-entropy implies a lower probability for an event.

**Explanation**: `CrossEntropy = -ln(Probability)`. Higher cross-entropy = more negative log = smaller probability.

---

### Quiz Set 12: Gradient Scalar Interpretation

**Question**: What does the scalar `(y - ŷ)` signify?

**Answers**:
- ✅ Closer the label to the prediction, smaller the gradient
- ✅ Farther the label from the prediction, larger the gradient

**Explanation**: Gradient magnitude = `|y - ŷ|·||x||`. When prediction is correct (y = ŷ), gradient is 0 (no update needed).

---

### Quiz Set 13: Combining Perceptrons

**Question**: For `w₁·0.4 + w₂·0.6 + b`, which weights/bias give final probability 0.88?

**Solution**: Need `score` such that `σ(score) = 0.88`

`σ(score) = 1/(1+e⁻ˢᶜᵒʳᵉ) = 0.88`
`1 + e⁻ˢᶜᵒʳᵉ = 1/0.88 ≈ 1.136`
`e⁻ˢᶜᵒʳᵉ = 0.136`
`-score = ln(0.136) ≈ -2`
`score ≈ 2`

Test options:
- Option 1: `2(0.4) + 6(0.6) - 2 = 0.8 + 3.6 - 2 = 2.4` ❌
- Option 2: `3(0.4) + 5(0.6) - 2.2 = 1.2 + 3 - 2.2 = 2.0` ✅
- Option 3: `5(0.4) + 4(0.6) - 3 = 2 + 2.4 - 3 = 1.4` ❌

**Answer**: `w₁: 3, w₂: 5, b: -2.2`

---

## Code Implementations

### AND Perceptron

```python
# AND Perceptron Implementation
and_weight1 = 0.2
and_weight2 = 0.3
and_bias = -0.4

def AND_perceptron(x1, x2):
    score = and_weight1 * x1 + and_weight2 * x2 + and_bias
    return 1 if score >= 0 else 0

# Truth table verification
print(AND_perceptron(0, 0))  # 0
print(AND_perceptron(0, 1))  # 0
print(AND_perceptron(1, 0))  # 0
print(AND_perceptron(1, 1))  # 1
```

### NOT Perceptron

```python
# NOT Perceptron Implementation
not_weight1 = 0.0      # Ignore first input
not_weight2 = -0.5     # Negative weight for negation
not_bias = 0.1

def NOT_perceptron(x1, x2):  # x2 is the input to negate
    score = not_weight1 * x1 + not_weight2 * x2 + not_bias
    return 1 if score >= 0 else 0

# NOT gate (using x2 as input)
print(NOT_perceptron(0, 0))  # 1
print(NOT_perceptron(0, 1))  # 0
```

### Perceptron Algorithm

```python
import numpy as np

def perceptron_algorithm(X, y, learning_rate=0.1, epochs=100):
    """
    X: input features (n_samples, n_features)
    y: labels (n_samples,)
    learning_rate: α
    epochs: number of passes over the data
    """
    n_samples, n_features = X.shape
    w = np.random.randn(n_features)
    b = np.random.randn()
    
    for epoch in range(epochs):
        for i in range(n_samples):
            # Calculate prediction
            score = np.dot(w, X[i]) + b
            y_pred = 1 if score >= 0 else 0
            
            # Update if misclassified
            if y[i] != y_pred:
                if y[i] == 1:  # Positive label, predicted negative
                    w += learning_rate * X[i]
                    b += learning_rate
                else:  # Negative label, predicted positive
                    w -= learning_rate * X[i]
                    b -= learning_rate
    
    return w, b
```

### Sigmoid Function

```python
import numpy as np

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid (for backpropagation)"""
    s = sigmoid(x)
    return s * (1 - s)
```

### Softmax Function

```python
def softmax(scores):
    """
    Convert scores to probabilities
    scores: array of shape (n_classes,)
    Returns: probabilities summing to 1
    """
    exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
    return exp_scores / np.sum(exp_scores)

# Example usage
scores = np.array([2.0, 1.0, 0.1])
probabilities = softmax(scores)
print(probabilities)  # [0.659, 0.242, 0.099]
print(sum(probabilities))  # 1.0
```

### Cross-Entropy Function

```python
def cross_entropy_binary(y_true, y_pred):
    """
    Binary cross-entropy loss
    y_true: true label (0 or 1)
    y_pred: predicted probability (0 to 1)
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def cross_entropy_multiclass(y_true, y_pred):
    """
    Multi-class cross-entropy loss
    y_true: one-hot encoded true labels (n_samples, n_classes)
    y_pred: predicted probabilities (n_samples, n_classes)
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)
```

### Logistic Regression with Gradient Descent

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for epoch in range(self.epochs):
            # Forward pass
            linear_output = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(linear_output)
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # Optional: print loss every 100 epochs
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(y_pred + 1e-15) + 
                               (1-y) * np.log(1 - y_pred + 1e-15))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict_proba(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return self.sigmoid(linear_output)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
```

### One-Hot Encoding

```python
def one_hot_encode(labels):
    """
    Convert categorical labels to one-hot encoded vectors
    labels: array of categories (e.g., ['duck', 'cat', 'dog', 'duck'])
    Returns: 
        encoded: 2D array of one-hot vectors
        classes: unique class labels
    """
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    
    n_samples = len(labels)
    encoded = np.zeros((n_samples, n_classes))
    
    for i, label in enumerate(labels):
        encoded[i, class_to_idx[label]] = 1
    
    return encoded, unique_classes

# Example
animals = ['duck', 'cat', 'dog', 'duck', 'cat']
one_hot, classes = one_hot_encode(animals)
print("Classes:", classes)  # ['cat', 'dog', 'duck']
print("One-hot:\n", one_hot)
# [[0. 0. 1.]  (duck)
#  [1. 0. 0.]  (cat)
#  [0. 1. 0.]  (dog)
#  [0. 0. 1.]  (duck)
#  [1. 0. 0.]] (cat)
```

### Backpropagation for 2-Layer Network

```python
class TwoLayerNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.lr = learning_rate
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        n_samples = X.shape[0]
        
        # Output layer gradients
        dZ2 = output - y
        dW2 = (1/n_samples) * np.dot(self.a1.T, dZ2)
        db2 = (1/n_samples) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.a1)
        dW1 = (1/n_samples) * np.dot(X.T, dZ1)
        db1 = (1/n_samples) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update parameters
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(output + 1e-15) + 
                               (1-y) * np.log(1 - output + 1e-15))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        output = self.forward(X)
        return (output >= 0.5).astype(int)
```

---

## Key Formulas Reference

### 1. Linear Model
```
score = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
ŷ = 1 if score ≥ 0 else 0
```

### 2. Sigmoid Activation
```
σ(z) = 1/(1 + e⁻ᶻ)
σ'(z) = σ(z)(1 - σ(z))
```

### 3. Softmax Activation
```
softmax(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ
```

### 4. Binary Cross-Entropy Loss
```
L = -[y·ln(ŷ) + (1-y)·ln(1-ŷ)]
```

### 5. Multi-Class Cross-Entropy
```
L = - Σᵢ Σⱼ yᵢⱼ·ln(ŷᵢⱼ)
```

### 6. Gradient Descent Update
```
θ ← θ - α·∇L(θ)
```

### 7. Perceptron Update (Misclassified)
```
If y=1, ŷ=0: w ← w + αx, b ← b + α
If y=0, ŷ=1: w ← w - αx, b ← b - α
```

### 8. Logistic Regression Gradient
```
∇L = (ŷ - y)·x
w ← w - α·(ŷ - y)·x
b ← b - α·(ŷ - y)
```

### 9. Chain Rule for Backpropagation
```
∂L/∂w = (∂L/∂ŷ)·(∂ŷ/∂z)·(∂z/∂w)
∂L/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))
∂ŷ/∂z = ŷ(1-ŷ)
∂z/∂w = x
```

### 10. Maximum Likelihood
```
L(θ) = Πᵢ P(yᵢ|xᵢ; θ)
l(θ) = Σᵢ ln(P(yᵢ|xᵢ; θ))
```

---

## Learning Outcomes

After completing this course, you will be able to:

### Foundational Knowledge
- ✅ Identify linear vs. non-linear classification problems
- ✅ Distinguish classification problems by dimensions and classes
- ✅ Identify perceptron components and their role in neural networks

### Implementation Skills
- ✅ Translate logical operators into perceptrons (AND, OR, NOT, XOR)
- ✅ Implement perceptron algorithm to find linear boundaries
- ✅ Adapt perceptron for non-linear boundaries
- ✅ Implement error functions and gradient descent
- ✅ Apply sigmoid for continuous predictions
- ✅ Use softmax for multi-class classification
- ✅ Implement one-hot encoding for categorical data

### Model Evaluation
- ✅ Use maximum likelihood and cross-entropy for performance measurement
- ✅ Apply logistic regression to minimize model error

### Advanced Concepts
- ✅ Identify neural network components and architectures
- ✅ Implement backpropagation for weight optimization
- ✅ Build multi-layer perceptrons for complex problems

### Project Completion
- ✅ Train deep learning models on image datasets (CIFAR-10)
- ✅ Create image classifiers using TensorFlow/PyTorch
- ✅ Evaluate model performance against benchmarks

---

## Workspace & Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Workspace won't start | Wait up to 5 minutes, refresh page |
| Notebook kernel dies | Reset workspace via Menu > Reset Data |
| Code changes lost | Download notebook before resetting: File > Download as |
| Import errors | Check that all required libraries are installed |
| Memory errors | Reduce batch size or simplify model |

### Best Practices

1. **Save work frequently**: Download notebooks after significant changes
2. **Use clear variable names**: Follow Python PEP 8 conventions
3. **Comment your code**: Explain complex logic
4. **Test incrementally**: Verify each component before combining
5. **Monitor loss values**: Ensure they decrease during training
6. **Use validation data**: Detect overfitting early

---

## Additional Resources

### Online Courses
- [Deep Learning Specialization (Andrew Ng)](https://www.deeplearning.ai/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### Books
- "Neural Networks and Deep Learning" by Michael Nielsen
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaron Courville

### Libraries Documentation
- [NumPy Documentation](https://numpy.org/doc/)
- [TensorFlow Documentation](https://www.tensorflow.org/docs)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Practice Platforms
- [Kaggle](https://www.kaggle.com/) - Real-world datasets and competitions
- [Google Colab](https://colab.research.google.com/) - Free GPU notebooks

---

## Course Completion Checklist

- [ ] Completed all video lessons
- [ ] Passed all quiz questions
- [ ] Implemented AND/OR/NOT perceptrons
- [ ] Coded perceptron algorithm from scratch
- [ ] Implemented sigmoid and softmax functions
- [ ] Coded cross-entropy loss function
- [ ] Built logistic regression with gradient descent
- [ ] Implemented one-hot encoding
- [ ] Coded backpropagation for 2-layer network
- [ ] Completed the Student Admissions notebook
- [ ] Built image classifier with TensorFlow
- [ ] Submitted final CIFAR-10 classifier project

---

## Acknowledgments

- **Luis Serrano** - Lead instructor and content creator
- **Udacity** - Platform and course infrastructure
- **Google** - Support and real-world insights

---

*Happy Learning! 🧠🤖*

*"Neural networks are not magic - they're just mathematics, and you've got this!"*
```

This complete README includes:
- ✅ All course prerequisites and requirements
- ✅ Complete tool setup instructions
- ✅ Detailed course outline
- ✅ **Every quiz question with correct answers and explanations**
- ✅ All mathematical formulas and derivations
- ✅ Full code implementations for every algorithm
- ✅ Learning outcomes checklist
- ✅ Troubleshooting guide
- ✅ Additional resource recommendations

# dolese_ai

A modern Python framework for artificial intelligence development with focus on machine learning, natural language processing, and data analysis.

## Features

- 🤖 **AI Model Framework**: Base classes for machine learning and deep learning models
- 📊 **Data Processing**: Utilities for data cleaning, preprocessing, and transformation  
- 🔧 **Configuration Management**: Flexible configuration system with environment variable support
- 📝 **Logging**: Structured logging with customizable formats and outputs
- 🎯 **Pre-built Models**: Ready-to-use classification and regression models
- 🧪 **Testing**: Comprehensive test suite with pytest

## Installation

### From Source

```bash
git clone https://github.com/seludoto/dolese_ai.git
cd dolese_ai
pip install -e .
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Classification Example

```python
from dolese_ai import DataProcessor
from dolese_ai.models import ClassificationModel
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)

# Initialize data processor
processor = DataProcessor()
X_train, X_test, y_train, y_test = processor.split_data(X, y)

# Create and train model
model = ClassificationModel(algorithm="random_forest")
model.train(X_train, y_train)

# Make predictions and evaluate
predictions = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Basic Regression Example

```python
from dolese_ai.models import RegressionModel
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=10)

# Create and train model
model = RegressionModel(algorithm="random_forest")
model.train(X, y)

# Make predictions
predictions = model.predict(X[:10])
```

### Data Processing

```python
from dolese_ai import DataProcessor
import pandas as pd

# Create processor
processor = DataProcessor()

# Clean data (handles missing values, duplicates)
cleaned_data = processor.clean_data(your_dataframe)

# Normalize features
normalized_X = processor.normalize_features(X)

# Split data
X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.2)
```

## Project Structure

```
dolese_ai/
├── src/dolese_ai/
│   ├── __init__.py          # Main package exports
│   ├── core.py              # Core AI model classes
│   ├── models/              # Pre-built model implementations
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   └── regression.py
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── config.py        # Configuration management
│       └── logger.py        # Logging utilities
├── tests/                   # Test suite
├── examples/                # Example scripts
├── docs/                    # Documentation
├── requirements.txt         # Dependencies
└── pyproject.toml          # Project configuration
```

## Configuration

dolese_ai uses a flexible configuration system:

```python
from dolese_ai.utils import Config

# Load default configuration
config = Config()

# Get values
model_type = config.get("model.type", "default")

# Set values
config.set("training.batch_size", 64)

# Environment variables (optional)
# DOLESE_AI_LOG_LEVEL=DEBUG
# DOLESE_AI_MODEL_TYPE=classification
```

## Available Models

### Classification Models
- **Random Forest**: `ClassificationModel(algorithm="random_forest")`
- **Logistic Regression**: `ClassificationModel(algorithm="logistic_regression")`
- **SVM**: `ClassificationModel(algorithm="svm")`

### Regression Models
- **Random Forest**: `RegressionModel(algorithm="random_forest")`
- **Linear Regression**: `RegressionModel(algorithm="linear_regression")`
- **SVR**: `RegressionModel(algorithm="svr")`

## Examples

Run the example script to see dolese_ai in action:

```bash
cd examples
python basic_example.py
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/seludoto/dolese_ai.git
cd dolese_ai

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Run tests
pytest

# Format code
black src/ tests/ examples/

# Type checking
mypy src/dolese_ai/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass (`pytest`)
6. Format your code (`black .`)
7. Commit your changes (`git commit -m 'Add some amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Deep learning model support with PyTorch/TensorFlow integration
- [ ] Natural Language Processing utilities
- [ ] Computer Vision model templates
- [ ] Model deployment utilities
- [ ] Hyperparameter optimization
- [ ] Model versioning and experiment tracking
- [ ] REST API generation from models
- [ ] Cloud deployment integration

## Support

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/seludoto/dolese_ai).

---

**Happy coding with dolese_ai! 🚀**

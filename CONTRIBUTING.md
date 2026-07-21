# Contributing to CPDiT

Thank you for your interest in contributing to the Latent Diffusion Transformer project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/CPDiT.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Set up development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install pytest black flake8
   ```

## Development Workflow

### Code Style

We follow PEP 8 with these guidelines:

- Format code with `black`:
  ```bash
  black src/ scripts/
  ```

- Lint with `flake8`:
  ```bash
  flake8 src/ scripts/
  ```

- Type hints for function signatures

### Writing Tests

Create tests in a `tests/` directory:

```python
import unittest
from src.models import LatentDiffusionTransformer

class TestModel(unittest.TestCase):
    def test_forward_pass(self):
        model = LatentDiffusionTransformer()
        # Add assertions
        self.assertTrue(...)
```

Run tests:
```bash
python -m pytest tests/
```

### Documentation

- Add docstrings to all classes and functions
- Update README.md for new features
- Include code examples for new functionality

### Commit Messages

Use clear, descriptive commit messages:

```
Add VAE loss weight scaling
- Implement beta parameter for KL divergence
- Update config to include vae_beta
- Tests pass for new parameter range
```

## Areas for Contribution

### High Priority

1. **Data loading**: Support for more satellite datasets (GOES, Sentinel, etc.)
2. **Model improvements**: Different architectures, attention mechanisms
3. **Training efficiency**: Mixed precision, gradient accumulation, distributed training
4. **Evaluation metrics**: More sophisticated forecasting metrics (SSIM, etc.)

### Medium Priority

1. **Documentation**: Tutorials, API docs
2. **Visualization**: Better prediction visualization tools
3. **Configuration**: More flexible config management
4. **Logging**: Better experiment tracking and reporting

### Low Priority

1. **Optimization**: Performance improvements
2. **Testing**: Expand test coverage
3. **Examples**: More usage examples

## Submitting Changes

1. Ensure all tests pass
2. Update documentation
3. Submit a pull request with:
   - Clear description of changes
   - Reference to any related issues
   - Examples of new functionality

## Report Issues

Use GitHub issues to report bugs or suggest features. Include:

- Clear description of the problem
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (Python version, CUDA, etc.)

## Questions?

Open a discussion or reach out to the maintainers.

Thank you for contributing!

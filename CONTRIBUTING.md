# Contributing to MLOps-Forge

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

Thank you for your interest in contributing to MLOps-Forge! We value the contributions of each developer and appreciate your effort to make this project better.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue using the bug report template and include:

- A clear title and description
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)

### Suggesting Enhancements

For feature requests, please create an issue using the feature request template and include:

- A clear title and description
- The motivation behind the enhancement
- How the enhancement would work
- Any alternative solutions you've considered

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Format code (`black .` and `isort .`)
6. Commit your changes (`git commit -m 'Add your feature'`)
7. Push to the branch (`git push origin feature/your-feature`)
8. Open a Pull Request

### Development Setup

1. Clone the repository
   ```bash
   git clone https://github.com/YourUsername/MLOps-Production-System.git
   cd MLOps-Production-System
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks
   ```bash
   pre-commit install
   ```

### Coding Standards

- Follow PEP 8 guidelines
- Write docstrings for all functions, classes, and modules
- Include type hints
- Add unit tests for new features

### Commit Message Guidelines

We follow conventional commits:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Changes to the build process or tools

Example: `feat: Add model drift detection`

## License

By contributing, you agree that your contributions will be licensed under the project's license.

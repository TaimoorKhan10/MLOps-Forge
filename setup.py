from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="mlops_forge",
    version="1.0.0",
    author="Taimoor Khan",
    author_email="contact@taimoorkhan.dev",
    description="A complete production-ready MLOps framework with built-in distributed training, monitoring, and CI/CD.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TaimoorKhan10/MLOps-Forge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mlops-forge-train=mlops_forge.cli.train:main",
            "mlops-forge-serve=mlops_forge.cli.serve:main",
            "mlops-forge-monitor=mlops_forge.cli.monitor:main",
        ],
    },
)

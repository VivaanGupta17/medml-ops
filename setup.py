"""
MedML-Ops Package Setup
========================
Installs the medml-ops package as an editable install.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]

setup(
    name="medml-ops",
    version="1.0.0",
    author="MedML-Ops Contributors",
    description="FDA-Compliant MLOps Pipeline for Medical AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/medml-ops",
    packages=find_packages(where=".", include=["src*"]),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.9.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pre-commit>=3.4.0",
            "httpx>=0.24.0",
        ],
        "dicom": [
            "pydicom>=2.4.0",
            "SimpleITK>=2.3.0",
        ],
        "deep-learning": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "xgboost": [
            "xgboost>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medml-pipeline=scripts.run_pipeline:main",
            "medml-report=scripts.generate_report:main",
            "medml-validate=src.data_validation.schema_validator:main",
            "medml-bias=src.data_validation.bias_detector:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "mlops", "medical-ai", "fda", "gmlp", "pccp", "510k",
        "bias-detection", "fairness", "drift-monitoring", "model-card",
        "samd", "healthcare-ai", "regulatory-compliance",
    ],
    project_urls={
        "Documentation": "https://github.com/yourusername/medml-ops/docs",
        "FDA GMLP Guidance": (
            "https://www.fda.gov/medical-devices/software-medical-device-samd/"
            "good-machine-learning-practice-medical-device-development-guiding-principles"
        ),
        "Issue Tracker": "https://github.com/yourusername/medml-ops/issues",
    },
)

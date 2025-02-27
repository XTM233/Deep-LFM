from setuptools import setup

requirements = [
    "matplotlib==3.5.1",
    "numpy==1.22.3",
    "scikit-learn",
    "torch==1.11.0",
    "gpytorch==1.9.1",
    "pandas==2.0.0",
    "openpyxl==3.1.2",
]

setup(
    name="dlfm",
    author="Thomas Baldwin-McDonald",
    packages=["dlfm"],
    description="Pathwise deep latent force models with variational inducing points.",
    long_description=open("README.md").read(),
    install_requires=requirements,
    python_requires=">=3.9",
)

from setuptools import find_packages, setup

REQUIRED = [
    #"tensorflow==2.15.0",
    #"hydra-core",
    "optax",
    #"tensorflow_datasets",
    "invoke",
]


setup(
    name="voltage_fitting",
    python_requires=">=3.8.0",
    packages=find_packages(),
    install_requires=REQUIRED,
)

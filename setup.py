from setuptools import setup, find_packages

setup(
    name="jusThink",
    version="0.1.0",
    packages=find_packages(),  
    install_requires=[
        "openai",
        "tiktoken",
        "networkx",
        "matplotlib",
        "scikit-learn",
        "tenacity",
        "tqdm",
        "flask"
    ],
    description="My Python library using OpenAI and other dependencies",
   
)

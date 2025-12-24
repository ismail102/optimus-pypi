from setuptools import setup, find_packages

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Operating System :: OS Independent"
]

setup(
    name="optimus-jbscorer",
    version="0.0.1",
    description="JBScore++: A semantic and harmfulness-based metric for evaluating LLM jailbreak prompts",
    long_description=open("README.txt", encoding="utf-8").read() + '\n\n' + open('CHANGELOG.txt').read(),
    long_description_content_type="text/plain",
    url="",
    author="Ismail Hossain",
    author_email="ihossain@miners.utep.edu",
    license="MIT",
    classifiers=classifiers,
    keywords=[
        "LLM",
        "jailbreak",
        "AI safety",
        "adversarial prompts",
        "robust evaluation",
        "semantic similarity"
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "transformers",
        "sentence-transformers",
        "numpy"
    ],
)

JBScore++: Semantic–Harmfulness-Based Jailbreak Scoring
OVERVIEW

This repository provides an implementation of JBScore++, a continuous metric for evaluating jailbreak prompts in large language models. The metric jointly considers semantic similarity to a harmful target intent and the estimated harmfulness of the prompt content.

Unlike binary jailbreak success metrics such as Attack Success Rate (ASR), JBScore++ produces a real-valued score in the range [0, 1]. This allows finer-grained evaluation by penalizing trivial paraphrases, benign rewrites, and low-risk prompts while highlighting prompts that are both semantically aligned with harmful intent and likely to induce unsafe behavior.

The core implementation is provided through the JBScoreCalculator class.

KEY FEATURES

Semantic similarity computation using Sentence-BERT embeddings

Harmfulness estimation using an NLI-style sequence classification model

Continuous jailbreak scoring metric (JBScore++)

Compatible with CPU and GPU execution via PyTorch

Modular design enabling replacement of encoders or classifiers

DEPENDENCIES

The following libraries are required:

Python version 3.9 or higher

PyTorch

HuggingFace Transformers

Sentence-Transformers

NumPy

To install all dependencies using pip:

pip install torch transformers sentence-transformers numpy

For GPU support, install the CUDA-enabled version of PyTorch following the official instructions at:
https://pytorch.org/get-started/locally/

REQUIRED MODELS

Two pretrained models are required.

Semantic Similarity Encoder
A Sentence-BERT model, for example:
sentence-transformers/all-mpnet-base-v2

Harmfulness Classifier
An NLI-style sequence classification model with an entailment label, for example:
roberta-large-mnli

CLASS DESCRIPTION: JBScoreCalculator

The JBScoreCalculator class computes semantic similarity, harmfulness probability, and the final JBScore++ metric.

Initialization parameters:

sim_encoder: SentenceTransformer instance for similarity encoding

tokenizer: AutoTokenizer corresponding to the harmfulness classifier

harm_classifier: AutoModelForSequenceClassification (NLI-style)

device: torch.device("cpu") or torch.device("cuda")

METHOD DETAILS

compute_similarity(prompts1, prompts2)

Description:
Computes cosine similarity between two lists of prompts using Sentence-BERT embeddings.

Details:

Embeddings are L2-normalized before similarity computation

Cosine similarity values are clipped to the range [0, 1]

Input:

prompts1: list of strings

prompts2: list of strings

Output:

NumPy array of similarity scores

compute_harmfulness(prompts, hypothesis)

Description:
Computes the probability that each prompt entails a harmful hypothesis using an NLI classifier.

Process:

Tokenizes prompt-hypothesis pairs

Performs a forward pass through the classifier

Extracts the entailment probability

Input:

prompts: list of strings

hypothesis: string describing harmful intent

Output:

NumPy array of harmfulness probabilities in [0, 1]

jbscore(similarity, harmfulness, s_upper, h_lower, alpha, beta)

Description:
Computes the final JBScore++ value by combining similarity and harmfulness with penalty terms.

Parameters:

similarity: semantic similarity scores

harmfulness: harmfulness probabilities

s_upper: similarity threshold for trivial-copy penalty

h_lower: harmfulness threshold

alpha: similarity penalty steepness

beta: harmfulness penalty steepness

Output:

NumPy array of JBScore++ values in [0, 1]
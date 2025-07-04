# Qwen3 1.7B Quantisation Analysis

This repository provides the code and resources for analysing the impact of quantisation on the Qwen3 1.7B Large Language Model (LLM), particularly its performance on the Massive Multitask Language Understanding (MMLU) benchmark, VRAM usage, and inference speed.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Results structure](#results-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project is a companion to the article "The impact of quantising a small open source LLM", delving into the practical implications of applying quantisation techniques to small open source LLMs. This repository aims to provide a clear, reproducible methodology for assessing these trade-offs using the Qwen3 1.7B model (or any other Open Source LLM available on Huggingface) and the MMLU benchmark.

## Features

This project offers a specialised framework for evaluating the Qwen3 1.7B LLM with different quantisation configurations on the Massive Multitask Language Understanding (MMLU) dataset. It is designed to:

* Facilitate the testing of various quantisation schemes (e.g., 4-bit, 8-bit) on the Qwen3 1.7B model.
* Measure the model's accuracy on MMLU across diverse subjects for each quantisation level.
* Monitor and log key metrics such as inference speed (execution time) and GPU memory consumption (VRAM usage) during evaluation.
* Provide a standardised workflow to ensure experiments can be easily replicated and compared.

## Results structure

Evaluation results for different quantisation levels will be saved in the `results/` folder. The corresponding JSON files will store the accuracy, number of examples, execution time, used VRAM, and total VRAM for each evaluated MMLU subject for a given quantisation configuration.

The evaluation results will be stored in a JSON file with a structure similar to the following, with additional keys to denote the quantisation level:

```json
{
    "abstract_algebra": {
        "accuracy": 0.8235294117647058,
        "number_examples": 17,
        "execution_time": 364.404,
        "used_VRAM": 7934,
        "total_VRAM": 8188
    },
    "high_school_physics": {
        "accuracy": 0.625,
        "number_examples": 16,
        "execution_time": 840.058,
        "used_VRAM": 2224,
        "total_VRAM": 8188
    }
}
```

## License
MIT License. Copyright holder: Renaud Gaudron

## Acknowledgements
- HuggingFace Transformers for providing easy access to models and tokenizers.
- HuggingFace Datasets for the MMLU dataset.
- Qwen Team for releasing the Qwen3 1.7B model.
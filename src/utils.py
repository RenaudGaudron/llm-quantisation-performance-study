"""Collection of utility functions for benchmarking open-source LLMs with the MMLU dataset.

This handles common tasks such as safely loading JSON data, managing Hugging Face models
and datasets, and providing helpers for displaying and selecting MMLU subjects and examples.

Functions:
    - load_config: Loads configuration settings from a YAML file.
    - load_json_safely: Safely loads JSON data from a specified file path.
    - load_model_and_tokenizer: Loads a pre-trained Hugging Face model and tokenizer.
    - load_and_prepare_dataset: Loads and optionally samples from a Hugging Face dataset.
    - list_subjects: Extracts and optionally displays unique subjects from a dataset.
    - display_results: Displays previously evaluated subjects and their performance metrics.
    - random_subject: Selects a random, unevaluated subject from a list.
    - print_random_examples: Prints random examples from a filtered dataset for inspection.
    - get_nvidia_smi_output: Fetches the used and total VRAM using nvidia-smi
"""

import json
import random as rd
import subprocess
from pathlib import Path

import torch
import yaml
from datasets import Dataset, disable_progress_bar, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_config(config_path: str = "config.yaml") -> tuple:
    """Load configuration from a YAML file into a dict.

    Args:
        config_path (str): The path to the YAML configuration file. Defaults to "config.yaml".

    Returns:
        dict: A dictionary containing the model name, max new tokens, dataset name,
               quantization type, batch_status, batch_size,
               number of examples, number of subjects, results path, and log path.

    """
    # Check if the config file exists directly in the current working directory
    with Path(config_path).open() as f:
        # Load the YAML configuration file
        config_dict = yaml.safe_load(f)

        print(f"Model Name: {config_dict['model']['name']}")
        print(f"Max New Tokens: {config_dict['model']['max_new_tokens']}")
        print(f"Quantization: {config_dict['model']['quantization']}")
        print(f"Batch status: {config_dict['model']['batch_status']}")
        print(f"Batch size: {config_dict['model']['batch_size']}")

        print(f"Dataset Name: {config_dict['dataset']['name']}")
        print(f"Number Examples: {config_dict['dataset']['number_examples']}")
        print(f"Number Subjects: {config_dict['dataset']['number_subjects']}")

        print(f"Results Path: {config_dict['paths']['results']}")
        print(f"Log Path: {config_dict['paths']['logs']}")

        return config_dict


def load_json_safely(config_dict: dict) -> tuple[str, dict]:
    """Load data from a JSON file, handling cases where the file is empty, malformed, or missing.

    Args:
        config_dict (dict): The dictionary containing the configuration settings.

    Returns:
        tuple[str, dict]: A tuple containing:
                          - The full path of the file that was attempted to be loaded.
                          - The data loaded from the JSON file as a dictionary.
                            Returns an empty dictionary `{}` if the file is empty,
                            not valid JSON, or not found.

    """
    result_path = (
        config_dict["paths"]["results"]
        + config_dict["model"]["quantization"]
        + "_"
        + str(config_dict["dataset"]["number_examples"])
        + "_examples"
        + ".json"
    )

    try:
        # Check if the file exists
        if not Path(result_path).exists():
            print(
                f"Warning: The file '{result_path}' was not found. Returning an empty dictionary.",
            )
            return result_path, {}

        # Check if the file is empty
        if Path(result_path).stat().st_size == 0:
            print(f"Warning: The file '{result_path}' is empty. Returning an empty dictionary.")
            return result_path, {}

        with Path(result_path).open(encoding="utf-8") as f:
            return result_path, json.load(f)
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from the file '{result_path}'. "
            "It might be malformed or not contain valid JSON. Returning an empty dictionary.",
        )
        return result_path, {}


def load_model_and_tokenizer(config_dict: dict) -> dict:
    """Load a pre-trained model and tokenizer from the Hugging Face Transformers library.

    Also loads the BitsAndBytesConfig for quantization.

    Args:
        config_dict (dict): The dictionary containing the configuration settings.

    Returns:
        dict: A dictionary containing the loaded model and tokenizer.

    """
    model_dict = {}

    print(
        f"Loading model '{config_dict['model']['name']}' with quantization: "
        f"{config_dict['model']['quantization']} "
        f"and batching: {config_dict['model']['batch_status']}",
    )

    quantization_config = None

    if config_dict["model"]["quantization"] == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif config_dict["model"]["quantization"] == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Load the model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config_dict["model"]["name"],
            quantization_config=quantization_config,
            torch_dtype="auto",  # "auto" lets transformers decide based on config/model
            device_map="auto",
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have `accelerate` and `bitsandbytes` installed if using quantization.")
        raise

    tokenizer = AutoTokenizer.from_pretrained(config_dict["model"]["name"])

    # Configure tokenizer for batching if enabled
    if config_dict["model"]["batch_status"]:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # Store the model and tokenizer
    model_dict["model"] = model
    model_dict["tokenizer"] = tokenizer

    return model_dict


def load_and_prepare_dataset(config_dict: dict) -> Dataset:
    """Load a dataset from the Hugging Face datasets library and prepares it for use.

    If not all examples are loaded, a seed insures that the same subset is selected every time.

    Args:
        config_dict (dict): The dictionary containing the configuration settings.

    Returns:
        Dataset: A prepared dataset object.

    """
    print(f"Loading dataset '{config_dict['dataset']['name']}'... \n")

    # Load the dataset
    if config_dict["dataset"]["number_examples"] == "all":
        dataset = load_dataset(config_dict["dataset"]["name"], "all")
        dataset = dataset["test"][:]
        dataset = Dataset.from_dict(dataset)
    else:
        dataset = load_dataset(config_dict["dataset"]["name"], "all")
        dataset = dataset.shuffle(seed=5155)["test"][
            : int(config_dict["dataset"]["number_examples"])
        ]
        dataset = Dataset.from_dict(dataset)

    print(f"Number of questions loaded: {len(dataset['question'])}")

    return dataset


def list_subjects(dataset: Dataset) -> list:
    """List all unique subjects in a dataset.

    Args:
        dataset (Dataset): The huggingface dataset from which to extract subjects.

    Returns:
        list: A list of unique subjects in the dataset.

    """
    unique_subject = list(set(dataset["subject"]))

    print("Unique subjects in the dataset: \n")
    for subject in unique_subject:
        print(f"\u2022 {subject} \n")

    return unique_subject


def display_results(results: dict) -> list:
    """Display the topics already evaluated and their results.

    Args:
        results (dict): A dictionary containing the evaluation results for each topic.

    Returns:
        list: A list of subjects that have been evaluated.

    """
    subjects_evaluated = list(results.keys()) if results else []

    print(f"Subjects already evaluated: {subjects_evaluated} \n")
    for topic in results:
        print(f"Topic: {topic}")
        print(f"\t \u2022 Number of examples: {results[topic]['number_examples']}")
        print(f"\t \u2022 Accuracy: {results[topic]['accuracy']}")

    return subjects_evaluated


def random_subject(unique_subject: list, subjects_evaluated: list, config_dict: dict) -> list[str]:
    """Select random subjects from the list of unique subjects not yet evaluated.

    The number of random subjects is provided by the configuration file.

    Args:
        unique_subject (list): A list of unique subjects in the dataset.
        subjects_evaluated (list): A list of subjects that have already been evaluated.
        config_dict (dict): The dictionary containing the configuration settings.

    Returns:
        list[str]: A list of randomly selected subjects that have not been evaluated yet.
                   If n_sub is greater than the number of available subjects, all remaining
                   subjects will be returned.

    """
    remaining_subjects = list(set(unique_subject) - set(subjects_evaluated))

    if not remaining_subjects:
        print("No subjects remaining to be evaluated.")
        return []

    # Ensure that the number of subjects does not exceed the number of available subjects
    num_to_select = min(config_dict["dataset"]["number_subjects"], len(remaining_subjects))

    # Use random.sample to select unique random subjects
    subjects_of_interest = rd.sample(remaining_subjects, num_to_select)

    if num_to_select == 1:
        print(f"The random subject of interest is: {subjects_of_interest[0]}")
    else:
        print(f"The random subjects of interest are: {subjects_of_interest}")

    return subjects_of_interest


def print_random_examples(dataset: Dataset, subject_of_interest: list, n_ex: int = 1) -> None:
    """Print n_ex random examples from the dataset for the given subject.

    Args:
        dataset (Dataset): The dataset to sample from.
        subject_of_interest (list): The subject to filter examples by.
        n_ex (int): The number of random examples to print. Defaults to 1.

    """
    print(f"Random examples from the subjects of interest: {subject_of_interest} \n")
    print("--" * 50 + "\n")

    # Disable the progress bar for all subsequent dataset operations
    disable_progress_bar()

    questions = dataset.filter(lambda x: x["subject"] in subject_of_interest).shuffle()

    number_examples = min(n_ex, len(questions))

    questions = questions.select(range(number_examples))

    for ind, _ in enumerate(questions):
        print(f"The subject is {questions['subject'][ind]} \n")
        print(f"The question is: {questions['question'][ind]} \n")

        print("The options are:\n")
        for choice in questions["choices"][ind]:
            print(f"\t \u2022 {choice} \n")

        sol_index = questions["answer"][ind]
        print(f"The answer is: {questions['choices'][ind][sol_index]} \n")
        print("--" * 50 + "\n")


def get_nvidia_smi_output() -> tuple[float, float]:
    """Execute nvidia-smi command and returns its JSON output.

    Returns:
        tuple[float, float]: A tuple containing the used and total VRAM in MB.

    """
    # Query for memory usage in JSON format
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd).decode("utf-8").strip()  # noqa: S603
    # Parse the output
    used_memory_mb, total_memory_mb = map(int, output.split(","))
    return used_memory_mb, total_memory_mb

"""Provides the core logic for evaluating Language Models (LLMs).

This is done on specific subjects from a given dataset, particularly tailored for
multiple-choice question answering. It handles the prompting of the LLM,
parsing of its responses, and logging of the evaluation process and results.

The module is designed to work with Hugging Face Transformers models and
datasets, and it incorporates logging to track the evaluation progress
and outcomes for each subject.

It supports quantization and batching, which can both be turned on or off.

Functions:
    - llm_eval: The primary entry point for evaluation, determining whether to
            use batching or single-item evaluation based on configuration.
    - single_llm_eval: The main evaluation function that orchestrates the
      prompting, response generation, prediction extraction, and accuracy
      calculation for a given subject when batching is off.
    - batch_llm_eval: The main evaluation function that orchestrates the
      prompting, response generation, prediction extraction, and accuracy
      calculation for a given subject when batching is on.
    - llm_eval_init: Initializes evaluation parameters and log file paths.
    - log_initialisation: Writes initial setup messages to the log.
    - log_subject_initialisation: Logs the start of evaluation for a new
                                  subject.
    - log_processing_item: Logs the progress of processing individual
                           items in the dataset.
    - generate_and_log_llm_prompt: Constructs and logs the prompt sent
                                   to the LLM.
    - batch_generate_llm_response: Generates a response from the LLM
                                   based on provided messages.
    - log_response: Logs the LLM's raw response and the correct answer.
    - extract_and_log_prediction: Parses the LLM's response to extract
                                  a prediction and logs it.
    - cal_and_store_subject_acc: Calculates, logs, and stores the accuracy
                                 for a subject.
    - batch_init: Initializes the batching process for a subset of the dataset.
    - batch_log_processing_item: Logs the progress of processing a batch of items.
    - batch_generate_llm_prompt: Generates a formatted prompt for the LLM for
                                 a single item in a batch.
    - single_iteration: Iterates through a subject's dataset without batching.
    - batch_iteration: Iterates through a subject's dataset using batching.
    - single_generate_llm_prompt: Initializes the processing and generates
                                  a prompt for a single item.
    - single_generate_llm_response: Generates a text response from the LLM
                                    for a single set of messages.
    - format_message: Formats the message list for the LLM.
"""

# Importing necessary libraries
import datetime
import json
import re
import time
from pathlib import Path
from typing import TextIO

import evaluate
import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

from src.utils import get_nvidia_smi_output


def llm_eval(
    dataset: Dataset,
    result_path: str,
    model_dict: dict,
    subject_of_interest: list,
    results_dict: dict,
    config_dict: dict,
) -> None:
    """Check whether batching is on or off and call appropriate function.

    Args:
        dataset (Dataset): The Huggingface dataset containing the questions and answers.
        result_path (str): The path of the JSON file containing the result for this quantization.
        model_dict (dict): The dictionary containing the model and tokenizer.
        subject_of_interest (list): A list containing the subjects to be evaluated.
        results_dict (dict): A dictionary to store the evaluation results for each subject.
        config_dict (dict): The dictionary containing the configuration settings.

    """
    if config_dict["model"]["batch_status"]:
        print("Batch status is True. Calling batch_llm_eval...")
        batch_llm_eval(
            dataset,
            result_path,
            model_dict,
            subject_of_interest,
            results_dict,
            config_dict,
        )
    else:
        print("Batch status is False. Calling single_llm_eval...")
        single_llm_eval(
            dataset,
            result_path,
            model_dict,
            subject_of_interest,
            results_dict,
            config_dict,
        )


def batch_llm_eval(
    dataset: Dataset,
    result_path: str,
    model_dict: dict,
    subject_of_interest: list,
    results_dict: dict,
    config_dict: dict,
) -> None:
    """Evaluate a Language Model (LLM) on a specific subject from a dataset using batching.

    Args:
        dataset (Dataset): The Huggingface dataset containing the questions and answers.
        result_path (str): The path of the JSON file containing the result for this quantization.
        model_dict (dict): The dictionary containing the model and tokenizer.
        subject_of_interest (list): A list containing the subjects to be evaluated.
        results_dict (dict): A dictionary to store the evaluation results for each subject.
        config_dict (dict): The dictionary containing the configuration settings.

    """
    log_name = llm_eval_init(config_dict["paths"]["logs"])

    with Path(log_name).open("w") as f_log:
        log_initialisation(f_log, config_dict)

        for subject in subject_of_interest:
            start_time = log_subject_initialisation(f_log, subject)

            # Reset the lists containing the answers, predictions, and accuracy
            references, predictions, acc_metric = ([] for _ in range(3))

            # Filtering the dataset to the subject of interest
            subject_dataset = dataset.filter(lambda x, s=subject: x["subject"] == s)

            # Iterate through a subject's dataset in batches
            batch_iteration(
                f_log,
                model_dict,
                config_dict,
                subject_dataset,
                subject,
                references,
                predictions,
            )

            # Calculate and store the accuracy for the subject
            cal_and_store_subject_acc(
                f_log,
                result_path,
                start_time,
                subject,
                predictions,
                references,
                acc_metric,
                results_dict,
                subject_dataset,
            )


def single_llm_eval(
    dataset: Dataset,
    result_path: str,
    model_dict: dict,
    subject_of_interest: list,
    results_dict: dict,
    config_dict: dict,
) -> None:
    """Evaluate a Language Model (LLM) on a specific subject from a dataset without batching.

    Args:
        dataset (Dataset): The Huggingface dataset containing the questions and answers.
        result_path (str): The path of the JSON file containing the result for this quantization.
        model_dict (dict): The dictionary containing the model and tokenizer.
        subject_of_interest (list): A list containing the subjects to be evaluated.
        results_dict (dict): A dictionary to store the evaluation results for each subject.
        config_dict (dict): The dictionary containing the configuration settings.

    """
    log_name = llm_eval_init(config_dict["paths"]["logs"])

    with Path(log_name).open("w") as f_log:
        log_initialisation(f_log, config_dict)

        for subject in subject_of_interest:
            start_time = log_subject_initialisation(f_log, subject)

            # Reset the lists containing the answers, predictions, and accuracy
            references, predictions, acc_metric = ([] for _ in range(3))

            # Filtering the dataset to the subject of interest
            subject_dataset = dataset.filter(lambda x, s=subject: x["subject"] == s)

            # Iterate through a subject's dataset without batching
            single_iteration(
                f_log,
                model_dict,
                config_dict,
                subject_dataset,
                subject,
                references,
                predictions,
            )

            # Calculate and store the accuracy for the subject
            cal_and_store_subject_acc(
                f_log,
                result_path,
                start_time,
                subject,
                predictions,
                references,
                acc_metric,
                results_dict,
                subject_dataset,
            )


def llm_eval_init(log_path: str) -> str:
    """Create the log file path based on the current datetime.

    This function generates a unique log file name using the current timestamp,
    ensuring that each evaluation run has its own log.

    Args:
        log_path (str): The directory path where the log file will be saved.

    Returns:
        str: The full path for the log file, named with a datetime stamp.

    """
    current_time = datetime.datetime.now(tz=datetime.timezone.utc)
    datetime_str = current_time.strftime("%Y%m%d_%H%M%S")

    return log_path + f"evaluation_log_{datetime_str}.txt"


def log_initialisation(f_log: TextIO, config_dict: dict) -> None:
    """Write the first few initialisation lines in the log file.

    Args:
        f_log (TextIO): The log file object to write the initialisation message to.
        config_dict (dict): Contains the configuration settings, to be written in the log.

    """
    f_log.write("--" * 50 + "\n")
    f_log.write("Initialisation\n")
    f_log.write("--" * 50 + "\n")

    f_log.write(f"Model Name: {config_dict['model']['name']}" + "\n")
    f_log.write(f"Max New Tokens: {config_dict['model']['max_new_tokens']}" + "\n")
    f_log.write(f"Quantization: {config_dict['model']['quantization']}" + "\n")
    f_log.write(f"Batch status: {config_dict['model']['batch_status']}" + "\n")
    f_log.write(f"Batch size: {config_dict['model']['batch_size']}" + "\n")
    f_log.write(f"Dataset Name: {config_dict['dataset']['name']}" + "\n")
    f_log.write(f"Number Examples: {config_dict['dataset']['number_examples']}" + "\n")
    f_log.write(f"Number Subjects: {config_dict['dataset']['number_subjects']}" + "\n")
    f_log.write("--" * 50 + "\n")


def log_subject_initialisation(f_log: TextIO, subject: str) -> time:
    """Log the initialisation message for a specific subject being evaluated.

    Also, initialises the timer.

    Args:
        f_log (TextIO): The log file object to write to.
        subject (str): The subject to be evaluated.

    Returns:
        time: The start time of the timer.

    """
    start_time = time.perf_counter()

    f_log.write(f"Processing subject: {subject}\n")
    f_log.write("--" * 50 + "\n")

    return start_time


def batch_init(
    filtered_dataset: Dataset,
    s_index: int,
    batch_size: int,
) -> tuple[Dataset, list, list, list]:
    """Initilise the batching process.

    Args:
        filtered_dataset (Dataset): The subset of the dataset
                                             containing items relevant to the current subject.
        s_index (int): The starting index for the current batch.
        batch_size (int): The number of items to include in each batch.

    """
    batch_items = filtered_dataset[s_index : s_index + batch_size]

    batch_prompts_for_llm = []
    batch_correct_answers = []
    batch_log_prompts = []

    return batch_items, batch_prompts_for_llm, batch_correct_answers, batch_log_prompts


def log_processing_item(
    f_log: TextIO,
    subject: str,
    subject_dataset: Dataset,
    item: int,
    message: str,
) -> None:
    """Log the progress of processing an individual item within a subject's dataset.

    This function writes a formatted message to the log file indicating the current
    item number being processed out of the total items for the given subject.

    Args:
        f_log (TextIO): The log file object to write to.
        subject (str): The name of the subject currently being evaluated.
        subject_dataset (Dataset): The subset of the dataset containing items relevant
                                   to current subject.
        item (int): The 0-based index of the current item being processed within the
                    `filtered_dataset`.
        message (str): Message sent to the LLM for item number item + 1.

    """
    f_log.write(
        f"Subject {subject} - Processing item number: {item + 1} out of {len(subject_dataset)} \n",
    )
    f_log.write("--" * 50 + "\n")

    f_log.write(
        f"Message sent to the LLM for item {item + 1}: {message}\n",
    )
    f_log.write("--" * 50 + "\n")


def batch_generate_llm_response(
    batch_messages: list[list[dict]],
    model_dict: dict,
    max_new_tokens: int = 38_912,
) -> list[str]:
    """Generate text responses from a Language Model (LLM) for a batch of messages.

    This function prepares the input messages for a batch using the tokenizer's
    chat template, tokenizes them with padding, passes them to the model for
    text generation, and then decodes the generated token IDs back into
    human-readable strings.

    Args:
        batch_messages (List[List[dict]]): A list of message lists, where each inner list
                                            represents the chat history for one item in the batch.
        model_dict (dict): The dictionary containing the model and tokenizer.
        max_new_tokens (int): The maximum number of new tokens to generate for each response.

    Returns:
        List[str]: A list of generated text responses from the LLM, one for each item in the batch.

    """
    # Load the tokenizer and model
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]

    # Apply chat template to each set of messages in the batch
    batch_texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in batch_messages
    ]

    # Tokenize the entire batch with padding
    model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(
        model.device,
    )

    # Generate responses for the batch
    with torch.no_grad():  # Ensure no gradients are calculated during inference
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )

    # Decode the generated IDs. Need to slice correctly for each original input length.
    # Note: When padding_side="left", the input_ids are on the right,
    # so the generated tokens are appended to the *end* of the padded sequence.
    # We still slice based on the original input_ids length.
    decoded_responses = []
    for i in range(len(batch_messages)):
        input_length = model_inputs.input_ids[i].shape[0]
        # In batch decode, generated_ids is (batch_size,
        # sequence_length_including_input_and_generation)
        # We need to slice each sequence from the original
        # input length to get only the generated part.
        output_ids_for_item = generated_ids[i, input_length:]
        decoded_responses.append(tokenizer.decode(output_ids_for_item, skip_special_tokens=True))

    return decoded_responses


def log_response(f_log: TextIO, response: str, correct: str, references: list) -> None:
    """Log the LLM's generated answer and the corresponding correct answer to the log file.

    This function writes both the LLM's output and the ground truth answer
    into the provided log file.

    Args:
        f_log (TextIO): The log file object to write the information to.
        response (str): The LLM's generated response.
        correct (str): The correct answer associated with the question.
        references (list): A list to store the correct answers for reference.

    """
    f_log.write(f"Response made by LLM: {response} \n")
    f_log.write("--" * 50 + "\n")

    f_log.write(f"Correct answer: {correct} \n")
    f_log.write("--" * 50 + "\n")

    # Storing the correct answer in the reference list
    references.append(correct)


def extract_and_log_prediction(
    f_log: TextIO,
    response: str,
    predictions: list,
    references: list,
) -> None:
    """Extract the predicted answer from an LLM's response.

    This is done using regex, then logs the extracted prediction,
    and appends it to a list of predictions.

    It attempts to parse an integer option from the response. If parsing fails
    or no match is found, a default value of 99 is used for the prediction.

    Then, logs the current lists of ground truth references and model predictions.

    Args:
        f_log (TextIO): The log file object to write the extracted prediction to.
        response (str): The raw text response generated by the LLM.
        predictions (list): A list to which the extracted integer prediction (or 99
                            if parsing fails) will be appended.
        references (list): A list containing the correct answers for reference.

    """
    # Extracting the model's answer using Regex
    corrected_match = re.search(
        r"answer is: Option (.*?)\s*\. Done",
        response,
    )  # Added \. to match literal dot
    predicted_answer = (
        corrected_match.group(1).strip() if corrected_match else 99
    )  # Default to 99 if no match found
    f_log.write(f"Predicted answer: {predicted_answer} \n")
    f_log.write("--" * 50 + "\n")

    # Storing the predicted index
    try:
        predictions.append(int(predicted_answer))
    except ValueError:
        predictions.append(99)

    # Printing the references and predictions for debugging
    f_log.write(f"References: {references} \n")
    f_log.write(f"Predictions: {predictions} \n")
    f_log.write("--" * 50 + "\n")


def cal_and_store_subject_acc(
    f_log: TextIO,
    result_path: str,
    start_time: time,
    subject: str,
    pred: list,
    ref: list,
    acc_metric: list,
    results_dict: dict,
    dataset: Dataset,
) -> None:
    """Calculate the accuracy for a given subject.

    Also logs the result, and stores the subject's overall accuracy and number of examples
    in the results dictionary.

    Args:
        f_log (TextIO): The log file object.
        result_path (str): The location of the JSON file containing the results.
        start_time (time): The start time of the timer, used to calculate the execution time.
        subject (str): The name of the subject for which the accuracy is being computed.
        pred (list): A list of the model's predicted answers (typically integer indices).
        ref (list): A list of the ground truth answers (typically integer indices).
        acc_metric (list): A list accumulating accuracy scores for the current subject.
                           The calculated accuracy for the current set of predictions/references
                           will be appended to this list.
        results_dict (dict): The dictionary where the overall evaluation results
                             for each subject are stored.
        dataset (Dataset): The Hugging Face Dataset object containing only the examples for the
                            current subject.

    """
    # Calculates the execution time for that subject
    execution_time = round(time.perf_counter() - start_time, 3)

    # Accuracy metric for evaluation
    accuracy_metric = evaluate.load("accuracy")

    # Calculating accuracy for the subject
    f_log.write(f"Calculating accuracy for the subject: {subject} \n")
    acc_metric.append(accuracy_metric.compute(predictions=pred, references=ref)["accuracy"])
    f_log.write(f"The accuracy for subject {subject} is {acc_metric} \n\n")
    f_log.write("--" * 50 + "\n")

    # Fetching the used and total VRAM
    used_vram, total_vram = get_nvidia_smi_output()

    # Adding the results to the results dictionary
    results_dict[subject] = {
        "accuracy": np.mean(acc_metric),
        "number_examples": len(dataset),
        "execution_time": execution_time,
        "used_VRAM": used_vram,
        "total_VRAM": total_vram,
    }

    # Save the results to the result JSON file
    with Path(result_path).open("w") as f:
        json.dump(results_dict, f, indent=4)


def batch_log_processing_item(
    f_log: TextIO,
    subject: str,
    start_idx: int,
    batch_size: int,
    total_items: int,
) -> None:
    """Log the progress of processing a batch of items.

    Args:
        f_log (TextIO): The log file object to write to.
        subject (str): The name of the subject currently being evaluated.
        start_idx (int): The starting index of the current batch.
        batch_size (int): The size of the current batch.
        total_items (int): The total number of items for the current subject.

    """
    end_idx = min(start_idx + batch_size, total_items)

    f_log.write(
        f"Subject {subject} - Processing items from {start_idx + 1} "
        f"to {end_idx} out of {total_items}\n",
    )
    f_log.write("--" * 50 + "\n")


def batch_generate_llm_prompt(
    batch_items: dict,
    batch_item_idx: int,
    batch_prompts: list,
    batch_log_prompts: list,
    batch_truth: list,
) -> tuple[list[dict], str]:
    """Generate a formatted prompt for the LLM for a single item within a batch.

    Returns the message list and the raw string prompt for logging.

    Args:
        batch_items (Dict): A dictionary representing the current batch of items,
                            e.g., {'question': [q1, q2], 'choices': [[c1,c2..],[c1,c2..]], ...}.
        batch_item_idx (int): The index of the current item within the batch.
        batch_prompts (list): A list to which the prompt sent to the LLM will be appended.
        batch_log_prompts (list): A list to which the raw string prompt for logging
                                  will be appended.
        batch_truth (list): A list to which the ground truth will be appended.

    Returns:
        tuple[list[dict], str]: A tuple containing:
            - list(dict): A list of dictionaries representing the messages to be sent to the LLM.
            - str: The raw string of the prompt generated for logging.

    """
    agg_text = "Question. "
    agg_text += batch_items["question"][batch_item_idx]
    agg_text += ". Choose between the following options. "

    for i, choice in enumerate(batch_items["choices"][batch_item_idx]):
        agg_text += f"Option {i}: {choice}"
        if i < len(batch_items["choices"][batch_item_idx]) - 1:
            agg_text += ", "

    messages = format_message(agg_text)

    batch_prompts.append(messages)
    batch_log_prompts.append(agg_text)
    batch_truth.append(batch_items["answer"][batch_item_idx])

    return batch_prompts, batch_log_prompts, batch_truth


def single_iteration(
    f_log: TextIO,
    model_dict: dict,
    config_dict: dict,
    subject_dataset: Dataset,
    subject: str,
    references: list,
    predictions: list,
) -> None:
    """Iterate through a subject's dataset without batching.

    This function processes each item in a given subject's dataset individually.
    For each item, it constructs a prompt, generates a response from the LLM,
    and then extracts a prediction, compares it against the ground truth,
    and logs the results.

    Args:
        f_log (TextIO): The file object for the evaluation log, where processing
                        details and results will be written.
        model_dict (dict): The dictionary containing the model and tokenizer.
        config_dict (dict): The dict containing the configuration settings.
        subject_dataset (Dataset): The subset of the dataset containing items relevant to
                            the current subject.
        subject (str): The name of the subject currently being evaluated.
        references (list): A list to accumulate all ground truth answers (references)
                            for the current subject's items. This list is updated
                            during the processing of each item.
        predictions (list): A list to accumulate all extracted predictions from the LLM
                            for the current subject's items. This list is updated
                            during the processing of each item.

    """
    max_new_tokens = config_dict["model"]["max_new_tokens"]

    for i, item in tqdm(
        enumerate(subject_dataset),
        total=len(subject_dataset),
        desc=f"Processing {subject}: ",
    ):
        # Initialize processing for a single item
        messages, log_prompt, correct_answer = single_generate_llm_prompt(item)

        log_processing_item(f_log, subject, subject_dataset, i, log_prompt)

        # Generate response for the single item
        response = single_generate_llm_response(messages, model_dict, max_new_tokens)

        # Log response and extract prediction
        log_response(f_log, response, correct_answer, references)
        extract_and_log_prediction(f_log, response, predictions, references)


def batch_iteration(
    f_log: TextIO,
    model_dict: dict,
    config_dict: dict,
    subject_dataset: Dataset,
    subject: str,
    references: list,
    predictions: list,
) -> None:
    """Iterate through a subject's dataset in batches.

    This function orchestrates the batch-wise processing of questions for a given
    subject. For each batch, it constructs prompts, generates responses from the LLM,
    and then processes each individual response to extract predictions, compare
    them against ground truth, and log the results.

    Args:
        f_log (TextIO): The file object for the evaluation log, where processing
                        details and results will be written.
        model_dict (dict): The dictionary containing the model and tokenizer.
        config_dict (dict): The dict containing the configuration settings.
        subject_dataset (Dataset): The subset of the dataset containing items relevant to
                           the current subject.
        subject (str): The name of the subject currently being evaluated.
        references (list): A list to accumulate all ground truth answers (references)
                           for the current subject's items. This list is updated
                           during the processing of each batch.
        predictions (list): A list to accumulate all extracted predictions from the LLM
                            for the current subject's items. This list is updated
                            during the processing of each batch.

    """
    batch_size = config_dict["model"]["batch_size"]
    max_new_tokens = config_dict["model"]["max_new_tokens"]

    # Iterate through the dataset in batches
    for i in tqdm(range(0, len(subject_dataset), batch_size), desc=f"Processing {subject}: "):
        batch_items, batch_prompts, batch_truth, batch_log_prompts = batch_init(
            subject_dataset,
            i,
            batch_size,
        )

        # Iterate over items in the current batch
        for batch_item_idx in range(len(batch_items["question"])):
            # Generate prompts for each item in the batch
            batch_prompts, batch_log_prompts, batch_truth = batch_generate_llm_prompt(
                batch_items,
                batch_item_idx,
                batch_prompts,
                batch_log_prompts,
                batch_truth,
            )

        batch_log_processing_item(f_log, subject, i, batch_size, len(subject_dataset))

        # Generate responses for the entire batch
        batch_responses = batch_generate_llm_response(
            batch_prompts,
            model_dict,
            max_new_tokens,
        )

        # Process each response and correct answer in the batch
        for j, response in enumerate(batch_responses):
            correct = batch_truth[j]

            log_processing_item(f_log, subject, subject_dataset, i + j, batch_log_prompts[j])

            log_response(f_log, response, correct, references)
            extract_and_log_prediction(f_log, response, predictions, references)


def single_generate_llm_prompt(
    item: dict,
) -> tuple[list[dict], str, str]:
    """Initialize the processing for a single item.

    Args:
        item (dict): A dictionary representing the current item from the dataset.

    Returns:
        tuple[list[dict], str, str]: A tuple containing:
            - list[dict]: The list of messages formatted for the LLM.
            - str: The raw string of the prompt generated for logging.
            - str: The correct answer for the current item.

    """
    agg_text = "Question. "
    agg_text += item["question"]
    agg_text += ". Choose between the following options. "

    for idx, choice in enumerate(item["choices"]):
        agg_text += f"Option {idx}: {choice}"
        if idx < len(item["choices"]) - 1:
            agg_text += ", "

    messages = format_message(agg_text)

    correct_answer = item["answer"]
    return messages, agg_text, correct_answer


def single_generate_llm_response(
    messages: list[dict],
    model_dict: dict,
    max_new_tokens: int = 38_912,
) -> str:
    """Generate a text response from a Language Model (LLM) for a single set of messages.

    This function prepares a single input message using the tokenizer's
    chat template, tokenizes it, passes it to the model for text generation,
    and then decodes the generated token IDs back into a human-readable string.

    Args:
        messages (list[dict]): A list of dictionaries representing the chat history
                               for a single conversational turn.
        model_dict (dict): The dictionary containing the model and tokenizer.
        max_new_tokens (int): The maximum number of new tokens to generate for the response.

    Returns:
        str: The generated text response from the LLM.

    """
    # Load the tokenizer and model
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]

    # Apply chat template to the single set of messages
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize the single text. No padding needed for a single input.
    model_inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)

    # Generate response for the single input
    with torch.no_grad():  # Ensure no gradients are calculated during inference
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )

    # Decode the generated IDs.
    # The generated_ids tensor will be of shape (1, sequence_length_including_input_and_generation)
    # We slice from the original input_ids length to get only the generated part.
    input_length = model_inputs.input_ids[0].shape[0]
    output_ids_for_item = generated_ids[0, input_length:]

    return tokenizer.decode(output_ids_for_item, skip_special_tokens=True)


def format_message(agg_text: str) -> list:
    """Format the message list for the LLM.

    Args:
        agg_text (str): The prepared question and answer for the llm.

    Returns:
        list: A list containing the formatted message to be fed to the LLM.

    """
    return [
        {
            "role": "system",
            "content": "You are an expert.\\"
            " Please answer the question to the best of your ability. Be concise and clear.\\"
            " Your answer should be one of the provided options.\\"
            " The final sentence of\\"
            " your response should be: 'The answer is: <option>. Done.' where <option> is the\\"
            " option you choose. Be careful to output the correct option that you identified.\\"
            " Here's an example to illustrate the format://"
            " Question. Find the square root of 16. Choose between the following options.\\"
            " Option 0: 4, Option 1: 5, Option 2: 6, Option 3: 7.\\"
            " The answer is: Option 0. Done.\\"
            "Question. What is the third planet orbiting around the sun.\\"
            "Choose between the following options.\\"
            "Option 0: Mercury, Option 1: The Moon, Option 2: The Earth, Option 3: Saturn.\\"
            " The answer is: Option 2. Done.\\",
        },
        {"role": "user", "content": agg_text},
    ]

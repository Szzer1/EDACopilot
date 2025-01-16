import json
import os
import random
import re

from tqdm import tqdm

import load_data
import openai_api
import logging

# Set up paths and logging configuration

logging.basicConfig(level=logging.INFO)


# Function to extract JSON from a block wrapped in ```json```
def extract_json_content(content):
    json_block_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
    match = json_block_pattern.search(content)
    if match:
        return match.group(1)
    else:
        raise ValueError("No JSON block found in the content")


# Function to save a dictionary to a JSONL file
def save_to_jsonl(data, output_path):
    try:
        with open(f'{output_path}/script_dataset_rf_example.jsonl', 'a', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False)
            file.write("\n")  # Ensure each entry is on a new line
    except Exception as e:
        logging.error(f"Failed to save data to JSONL file: {e}")


# Function to make an API call with retry logic
def get_api_response_with_retry(prompt, chunk, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = openai_api.chat_gpt_api(prompt, chunk)
            json_data = extract_json_content(response)
            return json.loads(json_data)  # Return parsed JSON
        except ValueError as ve:
            logging.warning(f"ValueError encountered: {ve}. Retrying...")
        except json.JSONDecodeError as je:
            logging.warning(f"JSONDecodeError encountered: {je}. Retrying...")
        except Exception as e:
            logging.error(f"Unexpected error: {e}. Retrying...")
        retry_count += 1
    return None  # Return None if max retries are exhausted


if __name__ == '__main__':
    DATA_PATH = '../data'
    OUTPUT_PATH = './script_dataset'
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # Load and split dataset into chunks
    dataset = load_data.load_dataset(DATA_PATH)

    chunk_data = load_data.split_docs(dataset)
    random.shuffle(chunk_data)

    # Main loop to process each chunk of data

    stop_flag = 0

    for chunk in tqdm(chunk_data):
        sys = openai_api.cums_sys_prompt['script_judge'].replace('{}', chunk.metadata["source"])
        json_result = get_api_response_with_retry(sys, chunk.page_content)

        if not json_result["script_found"]:
            continue
        sys_script = openai_api.cums_sys_prompt['script'].replace('{}', chunk.metadata["source"])
        script_result = get_api_response_with_retry(sys_script, chunk.page_content)
        # If we got a valid response, process the result
        if script_result:
            for item in script_result:
                item['reference'] = chunk.page_content # Add the reference from the chunk
                item['source'] = chunk.metadata["source"]
                save_to_jsonl(item, OUTPUT_PATH)
            stop_flag += len(script_result)
        else:
            logging.warning("No valid JSON result returned after retries, skipping this chunk.")


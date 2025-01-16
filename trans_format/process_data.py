import logging
import os
import time
import re
import pandas as pd
import glob
import json
from tqdm import tqdm
from openai import OpenAI

# Configuration
CONFIG = {
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "base_url": os.environ.get("OPENAI_API_URL"),
    "token_limit": 4096,
    "retry_limit": 3
}

client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

PROMPTS = {
    'knowledge_advice_prompt': """
    You are an assistant in the field of Electronic Design Automation (EDA), focusing on helping users convert existing questions into Knowledge Advice-type questions. Knowledge Advice questions require you to provide advanced recommendations on design optimizations and strategies related to EDA tools (e.g., OpenROAD). In the solutions, incorporate detailed insights on design optimization strategies such as power distribution network, area optimization, and timing closure.
    
    Please output the result in the following format:
    
    {
        "knowledge_advice_question": "<Revised Knowledge Advice question>",
        "knowledge_advice_answer": "<Detailed answer offering design recommendations, optimizations, and relevant insights>",
        "topic": "<Relevant topic, such as 'Power Optimization' or 'Timing Closure'>"
    }
    
    **Example Input**:  
    Question: What is PDNGEN?  
    Answer: The PDNGEN module (pdn) in OpenROAD aims to simplify the process of adding a power grid into a floorplan. A user can specify a small set of power grid policies to be applied to the design, such as the layers to use, stripe width, and spacing, and then pdn will generate the actual metal straps. Power grid policies can be defined over both the standard cell area and areas occupied by macros.
    
    **Example Output**:
    
    {
        "knowledge_advice_question": "In a high-performance design using OpenROAD, what are some recommended strategies for configuring PDNGEN to ensure a robust power distribution network (PDN) across both standard cells and macros? Consider factors like layer selection, stripe width, and spacing to optimize power integrity and minimize IR drop in dense designs.",
        "knowledge_advice_answer": "The PDNGEN module (pdn) in OpenROAD is designed to streamline the integration of a power grid into the floorplan. To ensure a reliable power distribution network, it’s essential to carefully configure the PDNGEN settings. First, select layers that offer minimal resistance and ensure ample power delivery to high-demand regions. Wider and closely spaced stripes can improve power integrity but may increase routing congestion, so balancing stripe width and spacing is crucial. Additionally, setting different power grid policies for standard cell regions and macros can help account for varying power demands. For macros, larger metal straps on higher metal layers can prevent IR drop in power-hungry blocks. By customizing these parameters, designers can achieve an optimized power grid that meets performance needs while minimizing potential issues in complex designs.",
        "topic": "Power Distribution Network (PDN) Optimization"
    }
    
    Please use the format above to process new OpenROAD questions and answers.
    """,
    "script_prompt": """You are an assistant familiar with OpenROAD and EDA tools, focusing on helping users write and explain OpenROAD Python API code. Your task is to organize the OpenROAD code snippets and explanations provided by the user into a standardized JSON format. This format helps users quickly understand the purpose, input parameters, output results, and code examples of OpenROAD functions in different tasks.

    Please generate the output according to the following JSON structure and provide detailed descriptions:

    {
        "definition_description": "<A brief description of the code purpose>",
        "functionality_description": "<Detailed explanation of the code's implementation logic and workflow>",
        "inputs": {
            "parameter1": "<Description of the first input parameter and its role>",
            "parameter2": "<Description of the second input parameter and its role>",
            "parameterN": "Add more parameters as needed"
        },
        "outputs": "<Description of what the code produces or outputs>",
        "code_paradigm": "<Brief code of the programming paradigm>"
    }

    Content includes: function purpose, input parameters and their roles, output results, and code style. Provide any example code within the "code_paradigm" field along with an explanation.

    Example One:

    Content: \n 
    query: Template of reading .lib (liberty) files
    code: from openroad import Tech\nfrom pathlib import Path\n\ntech = Tech()\n# Set file path\nlibDir = Path('lib_path')\nlibFiles = libDir.glob('*.lib')\n# Read .lib files\nfor libFile in libFiles:\n  tech.readLiberty(libFile.as_posix())
    
    Response:

    {
        "definition_description": "This code reads all .lib (Liberty) files in a specified directory and loads them into the OpenROAD Tech object.",
        "functionality_description": "1. Initialize a Tech object to handle technology and cell library data.\n2. Define the directory path containing the Liberty files (.lib files) to be loaded.\n3. Use a loop to iterate through each .lib file in the directory, loading each file into the Tech object with the readLiberty method.\n4. Once loaded, these files enable the design to access essential cell library information for subsequent stages of the design flow.",
        "inputs": {
            "libDir": "Path to the directory containing .lib files.",
            "tech": "The OpenROAD Tech object used to load technology and cell library information."
        },
        "outputs": "Loads each .lib file in the specified directory into the Tech object for further use in the design flow.",
        "code_paradigm": "from openroad import Tech\nfrom pathlib import Path\n\ntech = Tech()\n# Set file path\nlibDir = Path('lib_path')\nlibFiles = libDir.glob('*.lib')\n# Read .lib files\nfor libFile in libFiles:\n  tech.readLiberty(libFile.as_posix())"
    }

    Please use the above format to process new OpenROAD code or instructions."""
}


def chat_gpt_api(content, flag):
    if flag == 'qa':
        sys_prompt = PROMPTS["knowledge_advice_prompt"]
    else:
        sys_prompt = PROMPTS["script_prompt"]

    user_prompt = content + "Response: \n ```json\n<your json is here>```"
    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return completion.choices[0].message.content
        except:
            time.sleep(1)
            continue


def process_data(data):
    pattern = r"```json\n(.*?)\n```"
    match = re.search(pattern, data, re.DOTALL)
    if match:
        return match.group(1)
    pattern = r"```\n(.*?)\n```"
    match = re.search(pattern, data, re.DOTALL)
    return match.group(1) if match else data


def process_qa_data(queries, answers):
    kl_query, kl_answer, kl_topic = [], [], []

    for num in tqdm(range(len(queries)), desc="Processing QA data"):
        retry_count = 0
        while retry_count < CONFIG["retry_limit"]:
            content = f"Question: {queries[num]}\n Answer: {answers[num]}"
            response = chat_gpt_api(content, 'qa')
            output = process_data(response)

            try:
                data = json.loads(output)
                kl_query.append(data['knowledge_advice_question'])
                kl_answer.append(data['knowledge_advice_answer'])
                kl_topic.append(data['topic'])
                break
            except json.JSONDecodeError:
                retry_count += 1
                logging.warning(f"Retry {retry_count}/{CONFIG['retry_limit']} for QA query index {num}")
                if retry_count == CONFIG["retry_limit"]:
                    kl_query.append(None)
                    kl_answer.append(None)
                    kl_topic.append(None)
    return kl_query, kl_answer, kl_topic


def process_code_data(queries, answers, jsonl_file):
    for num in tqdm(range(len(queries[:5])), desc="Processing Code data"):
        retry_count = 0
        while retry_count < CONFIG["retry_limit"]:
            content = f"query: {queries[num]}\n code: {answers[num]}"
            response = chat_gpt_api(content, 'code')
            output = process_data(response)

            try:
                data = json.loads(output)
                data['example'] = content
                jsonl_file.write(json.dumps(data) + '\n')
                break
            except json.JSONDecodeError:
                retry_count += 1
                logging.warning(f"Retry {retry_count}/{CONFIG['retry_limit']} for Code query index {num}")


def main(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    kl_result = []
    csv_files = glob.glob(os.path.join(input_path, '**/*.csv'), recursive=True)
    for csv_file in csv_files:
        # Process QA Data
        if "Question-Answer" in csv_file:
            df = pd.read_csv(csv_file).dropna()
            queries, answers = df['Prompts'], df['Answers']
            kl_query, kl_answer, kl_topic = process_qa_data(list(queries), list(answers))

            result_df = pd.DataFrame({
                "type": ["knowledge_advice"]*len(kl_query),
                "topic": kl_topic,
                "query": kl_query,
                "answer": kl_answer,

            })

            kl_result.append(result_df)
        # Process script Data
        if "Prompt-Script" in csv_file:
            df = pd.read_csv(csv_file).dropna()
            queries, answers = df['prompt'], df['code']
            with open(f'{output_path}/script_format.jsonl', 'a') as jsonl_file:
                process_code_data(list(queries), list(answers), jsonl_file)

    final_result = pd.concat(kl_result, ignore_index=True)

    final_result.to_json(f'{output_path}/kl_output.jsonl', orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    input_path = './Non-Augmented_Data'
    output_path = './processed_data'
    main(input_path, output_path)

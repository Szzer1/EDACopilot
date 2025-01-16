import time
from tqdm import tqdm
from openai import OpenAI
import os

CONFIG = {
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "base_url": os.environ.get("OPENAI_API_URL"),
    "token_limit": 4096,
    "retry_limit": 3
}

client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

cums_sys_prompt = {"qa": """
You will act as an EDA tool expert, extracting key information from {} tool documentation or community discussions to create a series of Q&A pairs. 
Each Q&A pair must accurately reflect the content from the documentation or discussions, and the output must follow the JSON format. 
The `type` of each Q&A pair must be one of the following two types:

1. **Terminology explanation**: Explains fundamental concepts and terminology related to EDA processes, helping the user understand various stages from RTL to GDSII.
   
2. **Knowledge advice**: Provides EDA tool recommendations, optimizations, and solutions, addressing design challenges such as power reduction, area optimization, and timing closure.

Based on the content of each question and answer, you should select the appropriate `type`.

Output format:

```json
[
    {
        "type": "Terminology explanation",
        "query": "What is RTL?",
        "answer": "RTL (Register Transfer Level) is an abstraction used in digital circuit design, describing how data moves between registers and how operations are performed on the data."
    },
    {
        "type": "Knowledge advice",
        "query": "During timing closure, the critical paths in a design can often exceed the timing constraints. How can OpenROAD be used to address this issue?",
        "answer": "OpenROAD's optimization tools, like the delay mode in the restructuring module, focus on enhancing critical path timing by restructuring logic and optimizing the placement of cells to meet timing requirements without compromising on design area."
    },
    {
        "type": "Knowledge advice",
        "query": "When working on power optimization for a design, OpenROAD’s logic synthesis can help reduce dynamic power consumption. What specific strategy can be employed?",
        "answer": "By using OpenROAD’s logic re-synthesis feature in area mode, redundant gates and logic are eliminated, which not only reduces area but also cuts down on power consumption, particularly dynamic power, by minimizing switching activity in the design."
    }
]
```
Ensure each question and answer is accurately aligned with the documentation or discussion content, avoiding vague or overly general responses.
""",
                   "script": """ 
                   You are an EDA tool script usage expert. Your task is to generate script descriptions in the specified format based on the content of the provided {} EDA tool documentation. Each script description should include the following fields:

```json
[
    {
        "script_name": "<Name of the script>",
        "definition_description": "<A brief description of the script purpose>",
        "parameters": {
            "parameter1": "<Description of the first input parameter and its role>",
            "parameter2": "<Description of the second input parameter and its role>",
            "parameterN": "Add more parameter as needed"
        },
        "values": "<Values corresponding to each script in the text>",
        "script_paradigm": "<The specific tcl/python code template for the scrip>", 
        "examples": [
            {
                "query": "A summary of the specific example described in the text into question",
                "answer": "Specific code for the example"
            }
        ]
    }
]
```

Example:
```json
{
    "script_name": "SetClockConstraint",
    "definition_description": "This script sets a clock constraint for timing analysis in the design.",
    "parameters": {
        "clock_name": "The name of the clock to be set",
        "clock_period": "The period of the clock in nanoseconds"
    },
    "values": "clock_name: <clk>, clock_period: <ns>",
    "script_paradigm": "set_clock -name <clock_name> -period <clock_period>",
    "examples": [
        {
            "query": "How to set a 5ns clock constraint for clk?",
            "answer": "set_clock -name clk -period 5ns"
        },
        {
            "query": "Set a 10ns clock constraint for data_clk",
            "answer": "set_clock -name data_clk -period 10ns"
        }
    ]
}
```
Ensure each question and answer is accurately aligned with the documentation or discussion content, avoiding vague or overly general responses. 
And ensure that values in the script description do not contain single quotes and that all parameters are enclosed in < > to avoid any issues with JSON parsing
""",
                   "script_judge": """
You are an EDA tool script usage expert. Your task is to determine if the provided {} tool content contains any extractable scripts. If the content contains scripts, return the flag "script_found": true; if not, return the flag "script_found": false.

Output format:

```json
{
    "script_found": <true/false>
}
```
                   """}


def chat_gpt_api(sys_prompt, user_prompt):
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

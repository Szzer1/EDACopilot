# EDACopilot
This repository hosts the corpus generation, dataset examples to use RAG that serve as chatbots to EDA tool.

## Paper
EDA-Copilot: A RAG-Powered Intelligent Assistant for EDA Tools (Accepted by TODAES)

## Content
1) For existing open-source QA datasets, the dummy datasets are uploaded in this repo and can be found [here](https://github.com/Szzer1/EDACopilot/tree/main/trans_format/processed_data)
2) For the code that converts open-source QA datasets to the required format, the code are uploaded in this repo and can be found [here](https://github.com/Szzer1/EDACopilot/tree/main/trans_format)
3) For publicly available documentation on authorized open-source EDA tools, the dummy datasets are uploaded in this repo and can be found [here](https://github.com/Szzer1/EDACopilot/tree/main/generation/dataset)
4) For the code that constructs datasets for open-source EDA tools，the code are uploaded in this repo and can be found [here](https://github.com/Szzer1/EDACopilot/tree/main/generation)


## Usage
### Requirements
- Python3.10

#### 1. Clone this repo
```
git clone "https://github.com/Szzer1/EDACopilot.git"
```


### 2. Set up the environment
```
conda create -n EDACopilot
conda activate EDACopilot
cd EDACopilot
pip install -r requirements.txt
```

### 3. Run QA datasets format conversion
```
cd EDACopilot/trans_format
python process_data.py
```
Note: Update your own ```OPENAI_API_KEY``` and ```OPENAI_API_URL``` in ```.env```. Also, update the input and output paths in ```process_data.py```.

### 4. Run EDA tools dataset construction
```
cd EDACopilot/generation
```
- For QA pairs in text format
```
python qa_generation.py
```
- For script-style format
```
python script_format_generation.py
```
Note: Again, update ```OPENAI_API_KEY``` and ```OPENAI_API_URL``` and modify the input and output paths in ```qa_generation``` and ```script_format_generation```.

## Example
### For QA-pair format
```
{
	"type": "Terminology explanation", 
	"query": "What is the purpose of the less-than operator ( < ) in the context of Shape objects in klayout?", 
	"answer": "The less-than operator ( < ) compares two Shape objects based on pointers, and while it is not guaranteed to be strictly reproducible, it is sufficient to allow Shape objects to function as keys in hash-based data structures. This method was introduced in version 0.29.1.", 
	"reference": "!= Signature : [const] bool != (const Shape other) Description : Inequality operator < Signature : [const] bool < (const Shape other) Description : Less operator The less operator implementation is based on pointers and not strictly reproducible.However, it is good enough so Shape objects can serve as keys in hashes (see also hash ). This method has been introduced in version 0.29.1. == Signature : [const] bool == (const Shape other) Description : Equality operator Equality of shapes is not specified by the identity of the objects but by the", 
	"source": "klayout"
}
```
### For script format

```
{
	"script_name": "gui_zoom_out", 
	"definition_description": "This script zooms out the layout in the graphical user interface (GUI).", 
	"parameters": {
		"x": "The new x-coordinate of the layout center in microns", 
		"y": "The new y-coordinate of the layout center in microns"
	}, 
	"values": "x: <value in microns>, y: <value in microns>", 
	"script_paradigm": "gui::zoom_out <x> <y>", 
	"examples": [
		{"query": "How to zoom out the layout with the center at (100, 200) microns?", "answer": "gui::zoom_out 100 200"},
 		{"query": "Zoom out the layout with the center at (50, 75) microns", "answer": "gui::zoom_out 50 75"}], 
	"reference": "title: gui_zoom_out gui_zoom_out - gui zoom out\nSYNOPSIS\ngui::zoom_out \n    x y\nDESCRIPTION\nTo zoom out the layout:\nOPTIONS\nx, y:    new center of layout in microns\nARGUMENTS\nThis command has no arguments.\nEXAMPLES\nSEE ALSO", 
	"source": "OpenROAD"
}
```

## Citing this work

If this work has been helpful to you, we would greatly appreciate your citation.
```
@article{edacopilot,
  author = {Zhe Xiao, Xu He, HaoYing Wu, Bei Yu and Yang Guo},
  title = {EDA-Copilot: A RAG-Powered Intelligent Assistant for EDA Tools},
  journal={ACM Transactions on Design Automation of Electronic Systems},
  year={2025}
}
```
```
@misc{edacopilot,
  author = {Zhe Xiao, Xu He, HaoYing Wu, Bei Yu and Yang Guo},
  title = {EdaCopilot Repository},
  year = {2025},
  url = {https://github.com/Szzer1/EDACopilot},
  note = {Accessed: 20xx-xx-xx}
}
```

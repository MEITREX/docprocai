import json
import os
import argparse
import typing
import math
import random
import sys

parser = argparse.ArgumentParser(description='Generate training json from json files containing video sections.')

parser.add_argument('--in_dir', type=str, help='Path to a directory containing the json files to be processed.')
parser.add_argument('--out_file', type=str, help='Path to the output json file.')
parser.add_argument('--max_prompt_length', type=int, help='Maximum number of characters in a single training prompt, before it is split into 2.')
parser.add_argument('--eval_split', type=float, help='Fraction of the data to be used for evaluation.')
parser.add_argument("--eval_file", type=str, help="Path to the output json file for evaluation data.")

args = parser.parse_args()

in_dir: str = os.path.realpath(args.in_dir)
out_file: str = os.path.realpath(args.out_file)

max_prompt_length: int = args.max_prompt_length
eval_split: float = args.eval_split

template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

I have an input JSON file I need to process. It contains an array, where each element is a snippet of a lecture video. Each element contains the keys "start_time", which denotes the start time of the snippet in seconds after video start, a "transcript" of the spoken text, and "screen_text", the text on screen as detected by OCR. The transcript and screen_text might contain inaccuracies due to the nature of STT and OCR. The video was split into snippets by detecting when the screen changes by a significant amount. Please create a JSON file containing an array of elements, where each element represents the respective snippet from the input JSON. Each element should contain a title you'd give this snippet. Choose high-quality and concise titles. If you want two back-to-back snippet to be considered as the same chapter, give them the same title in your JSON array. Remember to answer only with a JSON file. This is the input JSON:

```
<<json_input>>
```<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<<json_output>><|eot_id|>
"""


def generate_prompt_from_template(args: dict[str, str]) -> str:
    prompt = template
    for key in args:
        prompt = prompt.replace("<<" + key + ">>", args[key])
    return prompt


def generate_prompts_from_sections(data: list[typing.Any]) -> typing.Generator[str, None, None]:
    data_clone = data.copy()
    # data is a json array of sections. We generate prompts from that. A prompt should contain as many sections as fit into max_prompt_length characters.
    while len(data_clone) > 0:
        prompt_in_json = []
        prompt_out_json = []

        while True:
            if(len(data_clone) == 0):
                break

            section_json = data_clone.pop(0)

            section_input = {
                "start_time": section_json["start_time"],
                "transcript": section_json["transcript"],
                "screen_text": section_json["screen_text"],
            }

            if("title" not in section_json):
                raise ValueError("Title not found for section ", section_json)

            section_output = {
                "title": section_json["title"],
            }

            prompt_in_json.append(section_input)
            prompt_out_json.append(section_output)

            if len(generate_prompt_from_template({
                    "json_input": json.dumps(prompt_in_json, indent=4, ensure_ascii=False), 
                    "json_output": json.dumps(prompt_out_json, indent=4, ensure_ascii=False)
                })) > max_prompt_length and len(prompt_in_json) > 1:
                data_clone.insert(0, section_json)
                prompt_in_json.pop()
                prompt_out_json.pop()
                break

        yield generate_prompt_from_template({
            "json_input": json.dumps(prompt_in_json, indent=4, ensure_ascii=False), 
            "json_output": json.dumps(prompt_out_json, indent=4, ensure_ascii=False)
            })


prompts = []

for subdir, dirs, files in os.walk(in_dir):
    for file in files:
        file_path = os.path.join(subdir, file)

        print("Processing ", file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            in_file_data = json.load(f)
            try:
                prompts.extend([{"text": x} for x in generate_prompts_from_sections(in_file_data)])
            except ValueError as e:
                raise ValueError("Error processing " + file_path)

random.shuffle(prompts)

training_prompts = prompts

if args.eval_split is not None:
    if args.eval_file is None:
        raise ValueError("Validation split specified but no validation file specified.")
    
    eval_file: str = os.path.realpath(args.eval_file)

    eval_count = int(math.ceil(len(prompts) * eval_split))
    eval_prompts = prompts[:eval_count]
    training_prompts = prompts[eval_count:]

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_prompts, f, ensure_ascii=False, indent=4)


with open(out_file, "w", encoding="utf-8") as f:
    json.dump(training_prompts, f, ensure_ascii=False, indent=4)

print("Generated ", len(prompts), " prompts")

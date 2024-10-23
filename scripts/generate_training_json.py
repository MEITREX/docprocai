import json
import os
import typing
import math
import random
import sys
import yaml

config = yaml.load(sys.argv[1])

in_dir: str = os.path.realpath(config["in_dir"])
out_file: str = os.path.realpath(config["out_file"])

max_prompt_length: int = config["max_prompt_length"]
eval_split: float = config["eval_split"]


def generate_prompt_from_template(args: dict[str, str]) -> str:
    prompt = config["prompt"].format(**args)
    return prompt


def generate_prompts_from_sections(data: list[typing.Any]) -> typing.Generator[str, None, None]:
    data_clone = data.copy()
    # data is a json array of sections. We generate prompts from that. A prompt should contain as many sections as fit into max_prompt_length characters.
    while len(data_clone) > 0:
        prompt_in_json = []
        prompt_out_json = {}

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

            prompt_in_json.append(section_input)
            # convert the start time to a string because JSON property keys can only be strings
            prompt_out_json[str(section_json["start_time"])] = section_json["title"]

            if len(generate_prompt_from_template({
                    "json_input": json.dumps(prompt_in_json, indent=4, ensure_ascii=False), 
                    "json_output": json.dumps(prompt_out_json, indent=4, ensure_ascii=False)
                })) > max_prompt_length and len(prompt_in_json) > 1:
                data_clone.insert(0, section_json)
                prompt_in_json.pop()
                prompt_out_json.pop()
                break
            
        prompt = generate_prompt_from_template({
            "json_input": json.dumps(prompt_in_json, indent=4, ensure_ascii=False), 
            "json_output": json.dumps(prompt_out_json, indent=4, ensure_ascii=False)
        })
        yield {
            "text": prompt
        }
            


prompts = []

for subdir, dirs, files in os.walk(in_dir):
    for file in files:
        file_path = os.path.join(subdir, file)

        print("Processing ", file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            in_file_data = json.load(f)
            try:
                prompts.extend([x for x in generate_prompts_from_sections(in_file_data)])
            except ValueError as e:
                raise ValueError("Error processing " + file_path)

random.shuffle(prompts)

training_prompts = prompts

if config["eval_split"] is not None:
    if config["eval_out_file"] is None:
        raise ValueError("Validation split specified but no validation file specified.")
    
    eval_file: str = os.path.realpath(config["eval_out_file"])

    eval_count = int(math.ceil(len(prompts) * eval_split))
    eval_prompts = prompts[:eval_count]
    training_prompts = prompts[eval_count:]

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_prompts, f, ensure_ascii=False, indent=4)


with open(out_file, "w", encoding="utf-8") as f:
    json.dump(training_prompts, f, ensure_ascii=False, indent=4)

print("Generated ", len(prompts), " prompts")

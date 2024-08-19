"""
Script which merges the JSON files outputted by the extract_video_sections.py back into one big json file.
"""

import json
import os
import sys

in_root_dir = sys.argv[1]
out_file = sys.argv[2]

data = []

for root, dirs, files in os.walk(in_root_dir):
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file), encoding="utf-8") as f:
                ele = json.load(f)
                data.append(ele)

data.sort(key=lambda x: x["start_time"])

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
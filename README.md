This repository contains scripts for training data generation for the DocProcAI service's automatic video segment title generation.

The generated training data can be used to train a model for the task of generating video segment titles using the Llama-Factory software.

# extract_video_sections.py
This script uses the video segmentation algorithms of the DocProcAI service (imported from the submodule) to extract video segments from a video file. The script takes a video file as input and outputs multiple JSON files (one file per segment) containing the extracted video segments to the specified directory.

# merge_processed_video_segments.py
This script merges processed video segments which are contained as separate files in a folder, into a single JSON file which then takes the form of a JSON array of the segment objects.

# generate_title_training_json.py
This script generates training data for the task of generating video segment titles. The script reads a JSON file containing video segment and title information and generates a JSON file containing training data for the task of generating video segment titles.
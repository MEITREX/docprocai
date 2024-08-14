import sys
import os
import json

# some PATH trickery to import the docprocai_service git submodule as
# a python package even though it isn't one
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "docprocai_service")))

from fileextractlib.VideoProcessor import VideoProcessor

in_root_path = os.path.realpath(sys.argv[1])
out_root_path = os.path.realpath(sys.argv[2])

# Extract video sections
vp = VideoProcessor()

for subdir, dirs, files in os.walk(in_root_path):
    for file in files:
        file_path = os.path.join(subdir, file)

        print("Generating sections for ", file_path)

        video_data = vp.process(file_path)

        print("Generated ", len(video_data.sections), " sections")

        out_dir = os.path.join(out_root_path, os.path.splitext(os.path.relpath(file_path, in_root_path))[0])
        os.makedirs(out_dir, exist_ok=True)

        for i, section in enumerate(video_data.sections):
            with open(os.path.join(out_dir, str(i) + ".json"), "w", encoding="utf-8") as f:
                json.dump({
                    "start_time": section.start_time,
                    "transcript": section.transcript,
                    "screen_text": section.screen_text
                }, f, ensure_ascii=False, indent=4)

        video_data.vtt.save(os.path.join(out_dir, "captions.vtt"))

        print("Finished generating json dump for ", file_path)

        
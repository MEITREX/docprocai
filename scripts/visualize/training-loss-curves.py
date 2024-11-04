import json
import sys
import matplotlib.pyplot as plt
import math

def smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699

    Source: https://stackoverflow.com/a/75421930
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed

title = sys.argv[1]
trainer_state_files = sys.argv[2::2]
legend_names = sys.argv[3::2]

# Select colors from Okabe & Ito's colorblind-safe palette
colors = [
    (0/255, 158/255, 115/255, 1),
    (0/255, 114/255, 178/255, 1),
    (240/255, 228/255, 66/255, 1),
    (213/255, 94/255, 0/255, 1),
    (204/255, 121/255, 167/255, 1),
]

plt.figure(figsize=(8, 5))

plt.title(title)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
for i, file in enumerate(trainer_state_files):
    with open(file) as f:
        trainer_state = json.load(f)
        
        x = [x["step"] for x in trainer_state['log_history'] if "loss" in x]
        y = [x["loss"] for x in trainer_state['log_history'] if "loss" in x]

        color = colors.pop(0)
        light_color = (color[0], color[1], color[2], color[3] * 0.4)

        plt.plot(x, y, label=legend_names[i], color=light_color, linewidth=1)
        plt.legend(loc="best")
        plt.plot(x, smooth(y, 0.9), label=legend_names[i] + " (smoothed)", color=color, linewidth=1.5)
        plt.legend(loc="best")

plt.show()
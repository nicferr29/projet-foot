import numpy as np
from datasets import load_dataset

ds = load_dataset("SoccerNet/SN-echoes", "whisper_v2")

print(ds)


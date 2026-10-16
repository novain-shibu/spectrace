# Generates run.py (the Kaggle matrix job) + kernel-metadata.json. Prompts are copied from ../vllm1/run.py.
import json

src = open("../vllm1/run.py").read()
P = src[src.index("    P = [") + 4: src.index("    K, NMIN")].replace("\n    ", "\n")

BODY = open("matrix_body.py").read()
open("run.py", "w").write(BODY.replace("#__PROMPTS__", P))
json.dump({"id": "<kaggle-user>/spectrace-matrix-1", "title": "spectrace-matrix-1", "code_file": "run.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": [], "competition_sources": [], "kernel_sources": ["<kaggle-user>/spectrace-wheelhouse"], "model_sources": []},
          open("kernel-metadata.json", "w"))
print("written")

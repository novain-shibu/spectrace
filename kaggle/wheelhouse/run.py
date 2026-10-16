# SpecTrace wheelhouse (CPU job, no GPU slot): download vLLM wheels + models ONCE into /kaggle/working.
# Later GPU jobs attach this kernel's output (kernel_sources) and install offline + load models from disk.
import subprocess, sys, os, time, json
t0 = time.time()
os.makedirs("wheels", exist_ok=True)
r = subprocess.run([sys.executable, "-m", "pip", "download", "-q", "-d", "wheels", "vllm"], capture_output=True, text=True)
print("pip download rc", r.returncode, r.stderr[-800:], "%.0f s" % (time.time() - t0), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]:
    p = snapshot_download(m, local_dir="models/" + m.split("/")[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab.json"])
    print("model", m, p, flush=True)
sz = lambda d: sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(d) for f in fs) / 1e9
print("RESULT", json.dumps({"wheels_GB": sz("wheels"), "models_GB": sz("models"), "n_wheels": len(os.listdir("wheels")), "seconds": time.time() - t0}))

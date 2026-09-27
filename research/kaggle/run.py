import glob
import os
import subprocess
import sys

hits = glob.glob("/kaggle/input/**/gpu_ce.py", recursive=True)
if not hits:
    raise SystemExit("gpu_ce.py not found under /kaggle/input")
data = os.path.dirname(hits[0])
print("data dir", data, os.listdir(data), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "pyarrow"], check=False)
cmd = [sys.executable, os.path.join(data, "gpu_ce.py"), "--data", data, "--out", "/kaggle/working/out",
       "--model", "intfloat/multilingual-e5-large", "--epochs", "1", "--batch", "32", "--lr", "2e-5", "--pairs", "900000", "--freeze-embeddings", "--tag", "klarge"]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
for f in glob.glob("/kaggle/working/out/*.pt"):
    os.remove(f)
print("outputs", os.listdir("/kaggle/working/out"), flush=True)

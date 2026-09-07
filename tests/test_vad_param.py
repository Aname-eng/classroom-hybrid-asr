import os
os.environ['NO_PROXY'] = '*'
import numpy as np
from funasr import AutoModel

vad = AutoModel(model='fsmn-vad', device='cpu', disable_update=True)
cache = {}

# In FunASR, chunk_size for fsmn-vad is chunk duration in ms, e.g. 60 (for 960 samples), or 200 (for 3200 samples), or 480 (for 7680 samples).
# But if chunk is 7680 samples and chunk_size=480:
for i in range(20):
    audio = np.random.randn(7680).astype(np.float32)
    # chunk_size should match ms: 7680 / 16 = 480ms
    res = vad.generate(input=audio, cache=cache, is_final=False, chunk_size=480)
    print(f"Step {i}:", res)

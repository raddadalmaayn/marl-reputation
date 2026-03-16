import os
from pathlib import Path
import time

base = Path('/home/raddad/marl-reputation/results/ablation/checkpoints')
for f in sorted(base.rglob('*.pt')):
    age = time.time() - os.path.getmtime(f)
    print(f"{f.relative_to(base)}: {age:.0f}s ago")

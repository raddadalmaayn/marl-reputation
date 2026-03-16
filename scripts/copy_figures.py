"""Copy figures from results/figures to marl-paper/figures."""
import shutil
from pathlib import Path

src_dir = Path('/home/raddad/marl-reputation/results/figures')
dst_dir = Path('/home/raddad/marl-reputation/marl-paper/figures')
dst_dir.mkdir(exist_ok=True)

for f in src_dir.glob('*.pdf'):
    shutil.copy2(f, dst_dir / f.name)
    print(f'Copied {f.name}')

for f in src_dir.glob('*.png'):
    shutil.copy2(f, dst_dir / f.name)
    print(f'Copied {f.name}')

print('Done')

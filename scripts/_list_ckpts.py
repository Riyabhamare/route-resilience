import os, torch

base_dir = 'models/baseline'
hdd_dir = 'models/hddnet'

print('=== Baseline checkpoints ===')
for f in sorted(os.listdir(base_dir)):
    if f.endswith('.pth'):
        size = os.path.getsize(os.path.join(base_dir, f)) / 1e6
        print(f'  {f} ({size:.1f} MB)')

print()
print('=== HDDNet checkpoints ===')
for f in sorted(os.listdir(hdd_dir)):
    if f.endswith('.pth'):
        size = os.path.getsize(os.path.join(hdd_dir, f)) / 1e6
        print(f'  {f} ({size:.1f} MB)')
        ckpt = torch.load(os.path.join(hdd_dir, f), map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', '?')
        tl = ckpt.get('train_loss', float('nan'))
        vl = ckpt.get('val_loss', float('nan'))
        print(f'    epoch={epoch}, train_loss={tl:.6f}, val_loss={vl:.6f}')

import torch

ckpt = torch.load('models/hddnet_archive_v2/hddnet_best.pth', map_location='cpu', weights_only=False)
print("V2 HDDNet Archive (hddnet_archive_v2/hddnet_best.pth):")
print(f"  Epoch:      {ckpt.get('epoch')}")
print(f"  Val IoU:    {ckpt.get('val_iou')}")
print(f"  Best IoU:   {ckpt.get('best_iou')}")
print(f"  Best Epoch: {ckpt.get('best_epoch')}")
print(f"  Val Loss:   {ckpt.get('val_loss')}")
print(f"  Train Loss: {ckpt.get('train_loss')}")
vs = ckpt.get('val_stats')
if vs:
    print(f"  Val Stats:  {vs}")

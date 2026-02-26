from ultralytics import YOLO

model = YOLO('best.pt')

print("Pokrećem evaluaciju...")
metrics = model.val(data='dataset/data.yaml', split='val')

print("\nREZULTATI:")
print(f"Precision: {metrics.box.map50:.3f}")
print(f"Recall:    {metrics.box.r.mean():.3f}") 
print(f"mAP@50:    {metrics.box.map50:.3f}")
print(f"mAP@50-95: {metrics.box.map:.3f}")
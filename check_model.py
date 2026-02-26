from ultralytics import YOLO

model = YOLO("best.pt")

print("--- ŠTA TVOJ MODEL VIDI ---")
print(model.names)
print("---------------------------")
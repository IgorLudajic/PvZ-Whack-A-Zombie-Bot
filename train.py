from ultralytics import YOLO

def train_model():
    # Učitavamo najmanji YOLO model (najbrži za igrice)
    model = YOLO('yolov8n.pt') 

    # Treniranje
    # data='dataset/data.yaml' -> Putanja do tvojih podataka
    # epochs=50 -> Koliko puta da pređe gradivo (dovoljno za početak)
    # imgsz=640 -> Veličina slike na kojoj uči
    results = model.train(
        data='dataset/data.yaml', 
        epochs=50, 
        imgsz=640,
        device='cpu' # Ako imaš NVIDIA grafičku, stavi 0 ili 'cuda'
    )

if __name__ == '__main__':
    train_model()
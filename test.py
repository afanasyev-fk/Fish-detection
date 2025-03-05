import torch
from ultralytics import YOLO
from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Engine, Events

logging.basicConfig(
    level=logging.INFO,
    filename='logs/test.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

model = YOLO('runs/detect/train/weights/best.pt')

input_dir = Path('dataset/test/images')
labels_dir = Path('dataset/test/labels')

writer = SummaryWriter(log_dir='logs/tensorboard')

def get_gt_count(label_path):
    if label_path.exists():
        with open(label_path, 'r') as f:
            return sum(1 for _ in f)
    return 0

def process_function(engine, batch):
    img_path = batch
    label_path = labels_dir / f"{img_path.stem}.txt"
    
    results = model(img_path, save=True, conf=0.5, iou=0.6)
    num_detections = len(results[0])
    
    num_gt = get_gt_count(label_path)
    
    logging.info(f'Image: {img_path.name} | Detections: {num_detections}/{num_gt}')
    
    writer.add_scalar('Detections per Image', num_detections, engine.state.iteration)
    writer.add_scalar('Ground Truth per Image', num_gt, engine.state.iteration)
    
    return num_detections, num_gt

engine = Engine(process_function)

@engine.on(Events.COMPLETED)
def on_completed(engine):
    logging.info("Процесс обнаружения завершен.")
    writer.close()

data = list(input_dir.iterdir())

engine.run(data)
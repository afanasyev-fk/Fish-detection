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

writer = SummaryWriter(log_dir='logs/tensorboard')

def process_function(engine, batch):
    img_path = batch
    results = model(img_path, save=True)
    num_detections = len(results[0])
    logging.info(f'Image: {img_path.name} | Detections: {num_detections}')
    
    writer.add_scalar('Detections per Image', num_detections, engine.state.iteration)
    return num_detections

engine = Engine(process_function)

@engine.on(Events.COMPLETED)
def on_completed(engine):
    logging.info("Процесс обнаружения завершен.")
    writer.close()

data = list(input_dir.iterdir())

engine.run(data)
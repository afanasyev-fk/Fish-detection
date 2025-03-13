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

total_detections = 0
total_gt = 0

def get_gt_count(label_path):
    """Считает количество объектов в файле аннотаций"""
    if label_path.exists():
        with open(label_path, 'r') as f:
            return sum(1 for _ in f)
    return 0

def process_function(engine, batch):
    global total_detections, total_gt

    img_path = batch
    label_path = labels_dir / f"{img_path.stem}.txt"

    results = model(img_path, save=True)
    num_detections = len(results[0])

    num_gt = get_gt_count(label_path)

    total_detections += num_detections
    total_gt += num_gt
    
    log_message = f'Image: {img_path.name} | Detections: {num_detections}/{num_gt}'
    logging.info(log_message)
    print(log_message)

    writer.add_scalar('Detections per Image', num_detections, engine.state.iteration)
    writer.add_scalar('Ground Truth per Image', num_gt, engine.state.iteration)
    
    return num_detections, num_gt

engine = Engine(process_function)

@engine.on(Events.COMPLETED)
def on_completed(engine):
    global total_detections, total_gt

    detection_percentage = (total_detections / total_gt * 100) if total_gt > 0 else 0

    final_log_message = (
        f"Обнаружено объектов: {total_detections} из {total_gt}\n"
        f"Процент обнаружения: {detection_percentage:.2f}%\n"
    )
    
    logging.info(final_log_message)
    print(final_log_message)

    writer.add_scalar('Total Detections', total_detections, 0)
    writer.add_scalar('Total Ground Truth', total_gt, 0)
    writer.add_scalar('Detection Percentage', detection_percentage, 0)

    logging.info("Процесс обнаружения завершен.")
    writer.close()

data = list(input_dir.iterdir())

engine.run(data)
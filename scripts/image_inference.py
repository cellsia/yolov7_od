import os
os.chdir('/app/yolov7')

import sys
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Añadir el directorio raíz de YOLOv7 al path
yolov7_path = str(Path(__file__).parent.parent.absolute())
if yolov7_path not in sys.path:
    sys.path.append(yolov7_path)

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

def process_image(model, image_path, output_path, device, img_size, conf_thres, iou_thres, half):
    # Cargar y preprocesar imagen
    dataset = LoadImages(image_path, img_size=img_size, stride=int(model.stride.max()))
    path, img, im0s, _ = next(iter(dataset))
    
    # Preparar imagen para inferencia
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inferencia y NMS
    with torch.no_grad():
        pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

    # Procesar detecciones
    det = pred[0]
    im0 = im0s.copy()

    if len(det):
        # Reescalar coordenadas
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        
        # Dibujar detecciones
        for *xyxy, conf, cls in reversed(det):
            label = f"{model.names[int(cls)]} {conf:.2f}"
            plot_one_box(xyxy, im0, label=label, color=(0, 255, 0), line_thickness=2)

    # Guardar imagen
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, im0)
    
    return len(det)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/app/weights/best.pt', help='model weights path')
    parser.add_argument('--input_path', type=str, required=True, help='path to input image or directory')
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    args = parser.parse_args()

    # Configurar dispositivo y modelo
    device = select_device(args.device)
    model = attempt_load(args.weights, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(args.img_size, s=stride)
    half = device.type != 'cpu'
    if half:
        model.half()

    # Procesar entrada
    input_path = Path(args.input_path)
    if input_path.is_file():
        # Crear carpeta predicted junto a la imagen
        # output_dir = input_path.parent / "predicted"
        # output_path = output_dir / input_path.name
        output_path = "/app/yolov7/scripts" + "/predicted_" + input_path.name
        detections = process_image(model, str(input_path), str(output_path), 
                                 device, img_size, args.conf_thres, args.iou_thres, half)
        print(f"Procesada {input_path.name} - {detections} detecciones")
    
    else:
        # Crear carpeta predicted dentro del directorio de entrada
        output_dir = input_path / "predicted"
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(ext)))
            image_files.extend(list(input_path.glob(ext.upper())))
        
        print(f"Buscando imágenes en: {input_path}")
        print(f"Encontradas {len(image_files)} imágenes")
        print(f"Extensiones soportadas: {image_extensions}")
        
        if len(image_files) == 0:
            print("No se encontraron imágenes. Verifica la ruta y los permisos.")
            return
        
        for img_path in tqdm(image_files):
            output_path = output_dir / img_path.name
            detections = process_image(model, str(img_path), str(output_path), 
                                    device, img_size, args.conf_thres, args.iou_thres, half)
            tqdm.write(f"Procesada {img_path.name} - {detections} detecciones")

if __name__ == "__main__":
    main()

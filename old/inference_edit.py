import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import onnxruntime as ort
import yaml 
from numpy import random
from fpdf import FPDF
import os
from pathlib import Path
import numpy as np
import sys
import colorsys  # Add at the top with other imports

from Opt import Opt
sys.path.append('/app/yolov7')
import test_
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from utils.general import increment_path
from utils.general import non_max_suppression

from utils.plots import plot_one_box
from utils.general import scale_coords
from utils.metrics import ap_per_class
from report import generate_pdf_with_front_page
from concurrent.futures import ThreadPoolExecutor

def load_model(weights, device):
    device = select_device(device)

    model = attempt_load(weights, map_location=device)  # Cargar el modelo 
    half = device.type != 'cpu'  # Habilitar precisión FP16 solo en GPU

    # Configurar el modelo para FP16 si está en GPU
    if half:
        model.half()
        
    print("Modelo cargado .pt y dispositivo configurado.")
    return model, half

def get_distinct_colors(n_colors):
    """
    Generate n distinct colors using HSV color space
    """
    colors = []
    hue_partition = 1.0 / n_colors
    
    for i in range(n_colors):
        # Use golden ratio to get well-distributed hues
        hue = (i * hue_partition + 0.618033988749895) % 1
        # Use fixed saturation and value for good visibility
        saturation = 0.8 + random.random() * 0.2  # 0.8-1.0
        value = 0.9 + random.random() * 0.1  # 0.9-1.0
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert to 0-255 range
        color = tuple(int(c * 255) for c in rgb)
        colors.append(color)
    
    return colors

def color_distance(c1, c2):
    """
    Calculate Euclidean distance between two colors
    """
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def get_names_colors(data_config):
    with open(data_config, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', []) 
    
    # Generate initial distinct colors
    base_colors = get_distinct_colors(len(names))
    
    # Ensure minimum distance between colors
    min_distance = 100  # Minimum Euclidean distance between colors
    colors = {}
    
    for i, base_color in enumerate(base_colors):
        current_color = base_color
        attempts = 0
        while attempts < 50:  # Limit attempts to avoid infinite loop
            # Check distance from all previously assigned colors
            too_close = False
            for existing_color in colors.values():
                if color_distance(current_color, existing_color) < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                break
                
            # If too close, generate a new color
            hue = random.random()
            sat = 0.8 + random.random() * 0.2
            val = 0.9 + random.random() * 0.1
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            current_color = tuple(int(c * 255) for c in rgb)
            attempts += 1
        
        colors[i] = current_color
    
    # Create class_colors mapping
    class_colors = {int(name): colors[i] for i, name in enumerate(names)}
    
    print("Nombres de clases y colores configurados:")
    print("Clases:", names)
    print("Colores asignados:", class_colors)
    
    return names, class_colors

def configurar_rutas(input_dir, output_dir, key):

    input_dir = Path(input_dir)
    key_path = Path(key)
    output_dir = Path(output_dir)
    dataset_dir = input_dir / "dataset"
    base_path = Path(dataset_dir)
    save_dir = output_dir / "resultados" / "pt"
    save_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = dataset_dir / "labels"
    print(f"Directorio de salida guardado configurado en: {save_dir}")
    save_dir = Path(save_dir)
    labels_dir2 = save_dir / key_path / "labels"
    labels_dir2.mkdir(parents=True, exist_ok=True)

    processed_images_dir = save_dir / key_path / "processed_images"  # Carpeta para guardar imágenes procesadas
    processed_images_dir.mkdir(parents=True, exist_ok=True)

    return base_path, labels_dir, labels_dir2, processed_images_dir



def obtener_ruta_desde_yaml(yaml_path, key='test'):
    """
    Lee la ruta asociada a una clave específica (por ejemplo, 'test') desde un archivo YAML.
    
    Args:
        yaml_path (str/Path): Ruta al archivo YAML.
        key (str): Clave en el YAML de la que se desea obtener la ruta.
        
    Returns:
        str: Ruta encontrada en el YAML.
    """
    # Cargar el archivo YAML
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    # Obtener la ruta correspondiente a la clave
    ruta = yaml_data.get(key)
    if not ruta:
        raise KeyError(f"No se encontró la clave '{key}' en el archivo YAML.")
    
    print(f"Ruta encontrada en el YAML para '{key}': {ruta}")
    return ruta

def leer_rutas_imagenes(base_path, txt_file_path):
    """
    Lee las rutas de las imágenes desde un archivo .txt.
    """
    with open(txt_file_path, 'r') as file:
        image_paths = [str(base_path / line.strip()) for line in file.readlines()]
    print(f"Se encontraron {len(image_paths)} rutas en el archivo .txt.")
    return image_paths


def preprocesar_imagen(img, device, half):
    """
    Preprocesa una imagen para la inferencia.
    """
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0  # Normalizar
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def realizar_inferencia(model, img, augment, conf_thres, iou_thres, classes, agnostic_nms):
    """
    Realiza la inferencia en el modelo y aplica NMS.
    """
    
    with torch.no_grad():
            preds = model(img, augment=augment)[0]

    pred = non_max_suppression(preds, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    return pred

def procesar_detecciones(pred, img, im0s, names, colors, txt_path, processed_images_dir, path, image_examples, expected_classes):
    """
    Procesa las detecciones, guarda resultados en archivos, dibuja en imágenes y extrae las cajas detectadas.
    """
    detected_classes = {}
    detecciones_procesadas = False
    detected_boxes = [] 

    # Abrir archivo para guardar detecciones
    with open(txt_path, "w") as f:
        for i, det in enumerate(pred):
            if len(det):  # Si hay detecciones
                print(f" - Detecciones encontradas: {len(det)}")
                detecciones_procesadas = True
                # Escalar coordenadas a la imagen original
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    predicted_class = int(cls)  # Clase predicha
                    detected_classes[predicted_class] = detected_classes.get(predicted_class, 0) + 1

                    # Extraer coordenadas de las cajas detectadas
                    x_min, y_min, x_max, y_max = map(int, xyxy)
                    detected_boxes.append((x_min, y_min, x_max, y_max))  # Añadir a detected_boxes

                    # Guardar detecciones en archivo .txt
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    f.write(f"{predicted_class} {x_center / im0s.shape[1]:.6f} {y_center / im0s.shape[0]:.6f} "
                            f"{width / im0s.shape[1]:.6f} {height / im0s.shape[0]:.6f} {conf:.6f}\n")

                    # Dibujar detección predicha en la imagen
                    label = f'{names[predicted_class]} {conf:.2f}'
                    bgr_color = (colors[predicted_class][2], colors[predicted_class][1], colors[predicted_class][0])

                    plot_one_box(xyxy, im0s, color=bgr_color, line_thickness=2)
            '''
            # Añadir etiquetas reales (clases esperadas)
            for expected_class, bboxes in expected_classes_coordinates.items():
                for bbox in bboxes:
                    x_min, y_min, x_max, y_max = bbox
                    xyxy = [x_min, y_min, x_max, y_max]
                    y_true.append(expected_class)  # Añadir clase real
                    label = f'{names[expected_class]}'
                    # Dibujar etiqueta real en la imagen
                    plot_one_box(xyxy, im0s, label=label, color=true_colors[expected_class], line_thickness=2)
            '''
        # Guardar la imagen procesada con detecciones
        processed_img_path = processed_images_dir / f"{Path(path).stem}_processed.jpg"

        cv2.imwrite(str(processed_img_path), im0s)
        print(f"Imagen procesada guardada en: {processed_img_path}")

        image_examples.append((str(processed_img_path), expected_classes, detected_classes))

    if not detecciones_procesadas:
        print(" - No se encontraron detecciones válidas tras aplicar NMS.")
    return image_examples

def box_iou(box1, box2):
    """
    Calculate IoU between two boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    
    # get the coordinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    
    # Intersection area
    inter_area = max(0, inter_rect_x2 - inter_rect_x1) * max(0, inter_rect_y2 - inter_rect_y1)
    
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union = b1_area + b2_area - inter_area
    
    return inter_area / union if union > 0 else 0

def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves
    """
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    # calculate area under PR curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap

def calculate_metrics(predictions, targets, iou_thres=0.5):
    """Calculate precision, recall and mAP for all classes"""
    unique_classes = set(int(t[0]) for t in targets).union(set(int(p[0]) for p in predictions))
    print(f"Classes found in predictions and targets: {sorted(unique_classes)}")
    
    results = {cls: {'precision': 0, 'recall': 0, 'map@0.5': 0, 'map@0.5:0.95': 0} for cls in unique_classes}
    classes = {cls: {'tp': [], 'fp': [], 'n_gt': 0} for cls in unique_classes}
    
    for t in targets:
        classes[int(t[0])]['n_gt'] += 1
    print(f"Ground truth counts per class: {[(cls, classes[cls]['n_gt']) for cls in sorted(classes.keys())]}")
    
    def process_prediction(pred):
        cls, conf, *pred_box = pred
        cls = int(cls)
        matched = False
        max_iou = 0
        
        for t in targets:
            if int(t[0]) == cls:
                iou = box_iou(pred_box, t[1:])
                if iou > max_iou:
                    max_iou = iou
                    matched = iou >= iou_thres
        
        if matched:
            return cls, (conf, 1), (conf, 0)
        else:
            return cls, (conf, 0), (conf, 1)
    
    with ThreadPoolExecutor() as executor:
        results_futures = list(executor.map(process_prediction, predictions))
    
    for cls, tp, fp in results_futures:
        classes[cls]['tp'].append(tp)
        classes[cls]['fp'].append(fp)
    
    mean_metrics = {'precision': 0, 'recall': 0, 'map@0.5': 0, 'map@0.5:0.95': 0}
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    
    for cls in sorted(classes.keys()):
        n_gt = classes[cls]['n_gt']
        if n_gt == 0:
            print(f"Warning: Class {cls} has no ground truth samples")
            continue
        
        tp = np.array(sorted(classes[cls]['tp'], reverse=True))
        fp = np.array(sorted(classes[cls]['fp'], reverse=True))
        if len(tp) == 0:
            print(f"Warning: Class {cls} has no predictions but {n_gt} ground truths")
            continue
        
        ap_values = []
        for iou_t in iou_thresholds:
            tp_cum = np.cumsum(tp[:, 1])
            fp_cum = np.cumsum(fp[:, 1])
            precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
            recall = tp_cum / n_gt
            ap = compute_ap(recall, precision)
            ap_values.append(ap)
            
            if iou_t == 0.5:
                results[cls].update({'precision': precision[-1] if len(precision) > 0 else 0, 'recall': recall[-1] if len(recall) > 0 else 0, 'map@0.5': ap})
                mean_metrics['map@0.5'] += ap
        
        results[cls]['map@0.5:0.95'] = np.mean(ap_values)
        mean_metrics['precision'] += results[cls]['precision']
        mean_metrics['recall'] += results[cls]['recall']
        mean_metrics['map@0.5:0.95'] += results[cls]['map@0.5:0.95']
    
    n_classes = len(classes)
    for k in mean_metrics:
        mean_metrics[k] /= n_classes if n_classes > 0 else 1
    
    print("\nDetailed metrics for each class:")
    for cls in sorted(results.keys()):
        print(f"Class {cls}: {results[cls]}")
    
    return results, mean_metrics

def main(input_dir, img_size, model, device, half, conf_thres, iou_thres, classes, augment, names, data_config, weights, output_dir, key, class_colors, max_examples=20):
    base_path,  labels_dir, labels_dir2, processed_images_dir = configurar_rutas(input_dir, output_dir, key)
    txt_file_path = obtener_ruta_desde_yaml(data_config, key=key)
    image_paths = leer_rutas_imagenes(base_path, txt_file_path)
    image_examples = []

    all_predictions = []
    all_targets = []
    
    for img_path in image_paths:
        # Obtener el nombre base de la imagen y encontrar su archivo de etiquetas
        base_name = Path(img_path).stem  # Nombre sin extensión
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        print(label_path)
        # Leer el archivo de etiquetas
        expected_classes = {}  # Diccionario para clases esperadas y su conteo
        expected_classes_coordinates = {} 
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    # Leer la clase desde la primera columna
                    label = int(line.strip().split()[0]) 
                    expected_classes[label] = expected_classes.get(label, 0) + 1

                                        # Leer clase y coordenadas
                    data = line.strip().split()
                    label = int(data[0])  # Primera columna es la clase
                    x_center, y_center, width, height = map(float, data[1:])

                    # Convertir coordenadas de normalizadas a absolutas
                    x_min = (x_center - width / 2) * img_size
                    y_min = (y_center - height / 2) * img_size
                    x_max = (x_center + width / 2) * img_size
                    y_max = (y_center + height / 2) * img_size

                    # Añadir las coordenadas a `expected_classes`
                    if label not in expected_classes_coordinates:
                        expected_classes_coordinates[label] = []
                    expected_classes_coordinates[label].append((x_min, y_min, x_max, y_max))
                    
                    all_targets.append([label, x_min, y_min, x_max, y_max])
        else:
            print(f"Advertencia: No se encontró el archivo de etiquetas para {img_path}")

        # Cargar imágenes
        dataset = LoadImages(img_path, img_size=img_size, stride=int(model.stride.max()))

        
        #for path, img, im0s, vid_cap in dataset:
        path, img, im0s, vid_cap = next(iter(dataset))

        img = preprocesar_imagen(img, device, half)
        pred = realizar_inferencia(model, img, augment=augment, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes, agnostic_nms=False)
        
        txt_path = labels_dir2 / f"{Path(path).stem}.txt"
        image_examples = procesar_detecciones(pred, img, im0s, names, class_colors, txt_path, processed_images_dir, path, image_examples, expected_classes)

        # Collect predictions
        if len(pred[0]):
            for *xyxy, conf, cls in pred[0]:
                cls_idx = int(cls.item())
                all_predictions.append([cls_idx, conf.item()] + [x.item() for x in xyxy])

           # break

    print(f"\nTotal targets collected: {len(all_targets)}")
    print(f"Total predictions collected: {len(all_predictions)}")
    
    # Calculate metrics
    class_metrics, mean_metrics = calculate_metrics(all_predictions, all_targets, iou_thres=0.5)
    
    print("\nFinal metrics per class:")
    for cls in class_metrics:
        print(f"Class {cls}: {class_metrics[cls]}")

    # Generate report with calculated metrics
    out = output_dir + "/" + key + "_" + "pt" + "_" + "report.pdf"
    generate_pdf_with_front_page(
        pdf_path=out,
        model_name=Path(input_dir).name,
        data_name=key,
        metrics=mean_metrics,
        class_names=names,
        image_examples=image_examples,
        class_colors=class_colors,
        metrics_classes=class_metrics,
        max_examples=max_examples
    )

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running the detection pipeline.")

    parser.add_argument('--input_dir', type=str, default="/app/pfs/eosinofilos",help="Directorio del proyecto")
    parser.add_argument('--output_dir', type=str,help="Directorio de salida")
    parser.add_argument('--img_size', type=int, default=1024,help="Tamaño de las imágenes para el modelo")
    parser.add_argument('--conf_thres', type=float, default=0.25,help="Umbral de confianza")
    parser.add_argument('--iou_thres', type=float, default=0.45,help="Umbral de IoU")
    parser.add_argument('--augment', type=bool, default=False,help="Augmentación durante la inferencia")
    parser.add_argument('--weights', type=str, default="/app/weights/best.pt",help="Ruta al modelo")
    parser.add_argument('--key', type=str, default="test",help="Dataset a testear")
    parser.add_argument('--max_examples', type=int, default=20, help="Numero de imagenes de ejemplo")
    
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_dir_path = Path(args.input_dir)
    if not input_dir_path.is_dir():
        print(f"El directorio de entrada {args.input_dir} no existe.")
        sys.exit(1)

    yaml_files = list(input_dir_path.rglob("*.yaml"))
    if not yaml_files:
        print(f"No se encontró ningún archivo .yaml en el directorio de entrada {args.input_dir}.")
        sys.exit(1)

    data_config = yaml_files[0]
    print(f"Archivo de configuración encontrado: {data_config}")
    
    model,  half = load_model(args.weights, device)
    names, class_colors = get_names_colors(data_config)


    print(f"Device: {device}, Half precision: {half}")

    main(
        input_dir=input_dir_path,
        img_size=args.img_size,
        model=model,
        device=device,
        half=half,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        classes=None,
        augment=args.augment,
        names=names,
        data_config=data_config,
        weights=args.weights,
        output_dir=args.output_dir,
        key = args.key,
        class_colors=class_colors,
        max_examples=args.max_examples
    )

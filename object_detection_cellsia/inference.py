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

def plot_striped_box(img, xyxy, color1, color2, conf=None, line_thickness=2, stripe_length=15):
    """
    Dibuja una caja con el patrón de rayas alternadas en la línea superior.
    color1: color de la clase (BGR)
    color2: color rojo para las rayas (BGR)
    """
    x1, y1, x2, y2 = map(int, xyxy)
    
    # Dibujar los tres lados completos con el color de la clase
    cv2.line(img, (x1, y2), (x2, y2), color1, line_thickness)  # línea inferior
    cv2.line(img, (x1, y1), (x1, y2), color1, line_thickness)  # línea izquierda
    cv2.line(img, (x2, y1), (x2, y2), color1, line_thickness)  # línea derecha
    
    # Dibujar la línea superior con patrón de rayas alternadas
    total_width = x2 - x1
    current_x = x1
    is_red = True  # Empezar con rojo
    
    while current_x < x2:
        end_x = min(current_x + stripe_length, x2)
        color = color2 if is_red else color1  # Alternar entre rojo y color de clase
        cv2.line(img, (current_x, y1), (end_x, y1), color, line_thickness)
        current_x = end_x
        is_red = not is_red  # Cambiar color para el siguiente segmento
    
    # Añadir texto de confianza si se proporciona
    if conf is not None:
        conf_str = f"{conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(conf_str, font, font_scale, font_thickness)
        
        margin = 2
        text_x = x1
        text_y = y1 - margin
        
        # Dibujar fondo del texto
        cv2.rectangle(img, 
                     (text_x, text_y - text_height - margin),
                     (text_x + text_width + margin * 2, text_y + margin),
                     color2, -1)
        
        # Dibujar texto
        cv2.putText(img, conf_str, (text_x + margin, text_y - margin), 
                    font, font_scale, (255, 255, 255), font_thickness)

def load_model(weights, device):
    device = select_device(device)

    model = attempt_load(weights, map_location=device)  # Cargar el modelo 
    half = device.type != 'cpu'  # Habilitar precisión FP16 solo en GPU

    # Configurar el modelo para FP16 si está en GPU
    if half:
        model.half()
        
    print("Modelo cargado .pt y dispositivo configurado.")
    return model, half

def is_forbidden_color(color):
    """
    Comprueba si un color RGB está en el rango de rojos, rosas, morados o grises.
    """
    r, g, b = color
    
    # Detectar rojos (R alto, G y B bajos)
    is_red = r > 150 and g < 100 and b < 100
    
    # Detectar rosas (R y B altos, G bajo)
    is_pink = r > 150 and b > 150 and g < 100
    
    # Detectar morados (R y B altos, pero R menor que en rosa)
    is_purple = r > 100 and b > 150 and g < 100
    
    # Detectar grises (diferencia entre canales menor a 30)
    is_grey = abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30
    
    # Detectar verdes (G alto, R y B bajos)
    is_green = g > 150 and r < 100 and b < 100
    
    return is_red or is_pink or is_purple or is_grey or is_green

def get_names_colors(model):
    names = model.module.names if hasattr(model, 'module') else model.names
    
    colors = {}
    for i in range(len(names)):
        while True:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if not is_forbidden_color(color):
                colors[int(i)] = color
                break
    
    class_colors = {int(name): colors[i] for i, name in enumerate(names)}
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

def calculate_iou(box1, box2):
    """
    Calcula IoU entre dos cajas.
    box format: (x1, y1, x2, y2)
    """
    # Área de intersección
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    
    # Áreas de las cajas
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # IoU
    iou = intersection / float(box1_area + box2_area - intersection)
    return iou

def procesar_detecciones(pred, img, im0s, names, colors, txt_path, processed_images_dir, path, image_examples, expected_classes, expected_classes_coordinates):
    """
    Procesa las detecciones, verificando IoU con las etiquetas reales.
    """
    detected_classes = {}
    detecciones_procesadas = False
    confidence_ranges = {
        '0-10': {'correct': 0, 'incorrect': 0},
        '10-20': {'correct': 0, 'incorrect': 0},
        '20-30': {'correct': 0, 'incorrect': 0},
        '30-40': {'correct': 0, 'incorrect': 0},
        '40-50': {'correct': 0, 'incorrect': 0},
        '50-60': {'correct': 0, 'incorrect': 0},
        '60-70': {'correct': 0, 'incorrect': 0},
        '70-80': {'correct': 0, 'incorrect': 0},
        '80-90': {'correct': 0, 'incorrect': 0},
        '90-100': {'correct': 0, 'incorrect': 0}
    }
    iou_threshold = 0.5

    with open(txt_path, "w") as f:
        # Primero dibujamos las cajas ground truth no detectadas en verde
        matched_boxes = set()
        
        # Procesar detecciones primero para marcar las coincidencias
        if len(pred[0]):
            det = pred[0]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            for *xyxy, conf, cls in reversed(det):
                predicted_class = int(cls)
                detected_box = tuple(map(int, xyxy))
                
                if predicted_class in expected_classes_coordinates:
                    for idx, gt_box in enumerate(expected_classes_coordinates[predicted_class]):
                        if idx not in matched_boxes and calculate_iou(detected_box, gt_box) >= iou_threshold:
                            matched_boxes.add((predicted_class, idx))
                            break
        
        # Dibujar cajas ground truth no detectadas en verde
        for class_id, boxes in expected_classes_coordinates.items():
            for idx, box in enumerate(boxes):
                if (class_id, idx) not in matched_boxes:
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(im0s, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        for i, det in enumerate(pred):
            if len(det):
                print(f" - Detecciones encontradas: {len(det)}")
                detecciones_procesadas = True
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # Marcamos las cajas de ground truth que ya han sido emparejadas
                matched_gt_boxes = set()

                for *xyxy, conf, cls in reversed(det):
                    predicted_class = int(cls)
                    detected_classes[predicted_class] = detected_classes.get(predicted_class, 0) + 1

                    # Coordenadas de la caja detectada
                    x_min, y_min, x_max, y_max = map(int, xyxy)
                    detected_box = (x_min, y_min, x_max, y_max)

                    # Guardar coordenadas y detección
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Guardar en archivo .txt
                    f.write(f"{predicted_class} {x_center / im0s.shape[1]:.6f} {y_center / im0s.shape[0]:.6f} "
                           f"{width / im0s.shape[1]:.6f} {height / im0s.shape[0]:.6f} {conf:.6f}\n")

                    # Verificar si la detección coincide con alguna caja ground truth
                    is_correct = False
                    if predicted_class in expected_classes_coordinates:
                        for idx, gt_box in enumerate(expected_classes_coordinates[predicted_class]):
                            if idx not in matched_gt_boxes:
                                iou = calculate_iou(detected_box, gt_box)
                                if iou >= iou_threshold:
                                    is_correct = True
                                    matched_gt_boxes.add(idx)
                                    break

                    # Debug print como en inference_onnx
                    print(f"Clase {predicted_class}: {'correcta' if is_correct else 'incorrecta'} "
                          f"(IoU máximo: {iou if 'iou' in locals() else 0:.3f})")

                    # Obtener color y convertir a BGR
                    bgr_color = (colors[predicted_class][2], colors[predicted_class][1], colors[predicted_class][0])

                    if not is_correct:
                        # Actualizar rangos de confianza para predicciones incorrectas
                        conf_value = float(conf) * 100
                        for range_key in confidence_ranges.keys():
                            min_val, max_val = map(int, range_key.split('-'))
                            if min_val <= conf_value < max_val:
                                confidence_ranges[range_key]['incorrect'] += 1
                                break
                        # Dibujar caja con línea roja y mostrar confianza
                        plot_striped_box(im0s, xyxy, bgr_color, (0, 0, 255), conf=conf, line_thickness=2)
                    else:
                        # Dibujar caja normal para detecciones correctas
                        plot_one_box(xyxy, im0s, color=bgr_color, line_thickness=2)

                    # Categorizar la confianza en rangos
                    conf_value = float(conf) * 100  # Convertir a porcentaje
                    for range_key in confidence_ranges.keys():
                        min_val, max_val = map(int, range_key.split('-'))
                        if min_val <= conf_value < max_val:
                            if is_correct:
                                confidence_ranges[range_key]['correct'] += 1
                            else:
                                confidence_ranges[range_key]['incorrect'] += 1
                            break

        # Guardar imagen procesada
        processed_img_path = processed_images_dir / f"{Path(path).stem}_processed.jpg"
        cv2.imwrite(str(processed_img_path), im0s)
        print(f"Imagen procesada guardada en: {processed_img_path}")
        
        image_examples.append((str(processed_img_path), expected_classes, detected_classes))

    if not detecciones_procesadas:
        print(" - No se encontraron detecciones válidas tras aplicar NMS.")
    
    return image_examples, confidence_ranges, detected_classes

def main(input_dir, img_size, model, device, half, conf_thres, iou_thres, classes, augment, names, data_config, weights, output_dir, key, class_colors, max_examples=20):
    base_path,  labels_dir, labels_dir2, processed_images_dir = configurar_rutas(input_dir, output_dir, key)
    txt_file_path = obtener_ruta_desde_yaml(data_config, key=key)
    image_paths = leer_rutas_imagenes(base_path, txt_file_path)
    image_examples = []
    
    total_detected_classes = {}  # Add this to track all detections
    total_confidence_ranges = {
        '0-10': {'correct': 0, 'incorrect': 0},
        '10-20': {'correct': 0, 'incorrect': 0},
        '20-30': {'correct': 0, 'incorrect': 0},
        '30-40': {'correct': 0, 'incorrect': 0},
        '40-50': {'correct': 0, 'incorrect': 0},
        '50-60': {'correct': 0, 'incorrect': 0},
        '60-70': {'correct': 0, 'incorrect': 0},
        '70-80': {'correct': 0, 'incorrect': 0},
        '80-90': {'correct': 0, 'incorrect': 0},
        '90-100': {'correct': 0, 'incorrect': 0}
    }

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
        else:
            print(f"Advertencia: No se encontró el archivo de etiquetas para {img_path}")

        # Cargar imágenes
        dataset = LoadImages(img_path, img_size=img_size, stride=int(model.stride.max()))

        
        #for path, img, im0s, vid_cap in dataset:
        path, img, im0s, vid_cap = next(iter(dataset))

        img = preprocesar_imagen(img, device, half)
        pred = realizar_inferencia(model, img, augment=augment, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes, agnostic_nms=False)
        
        txt_path = labels_dir2 / f"{Path(path).stem}.txt"
        image_examples, conf_ranges, current_detected_classes = procesar_detecciones(
            pred, img, im0s, names, class_colors, txt_path, 
            processed_images_dir, path, image_examples, expected_classes,
            expected_classes_coordinates  # Añadir este parámetro
        )
        
        # Update total detections
        for cls, count in current_detected_classes.items():
            total_detected_classes[cls] = total_detected_classes.get(cls, 0) + count
            
        # Acumular rangos de confianza
        for range_key in conf_ranges:
            total_confidence_ranges[range_key]['correct'] += conf_ranges[range_key]['correct']
            total_confidence_ranges[range_key]['incorrect'] += conf_ranges[range_key]['incorrect']

           # break

    test_.opt = Opt(key)

    print(data_config)

    with open(data_config, 'r') as f:
        data = yaml.safe_load(f)  # Cargar el contenido del YAML como diccionario

    if not isinstance(data, dict):
        raise ValueError("Error: El archivo YAML no se cargó correctamente como un diccionario.")

    results, maps, times, metrics_class = test_.test(
        data=data,
        weights=weights,
        batch_size=8,
        imgsz=1024,
        conf_thres=0.001,
        iou_thres=0.65,
        save_json=False,
        save_txt=True,
        save_hybrid=False,
        save_conf=True,
        verbose=True
    )

    # Crear dataset_stats manualmente
    dataset_stats = {
        'total_images': len(image_paths),
        'processed_images': len(image_paths),
        'total_labels': sum(total_detected_classes.values()) if total_detected_classes else 0
    }

    mp, mr, map50, map2, loss = results[:5]
    print(f"Precisión media: {mp:.3f}")
    print(f"Recall medio: {mr:.3f}")
    print(f"mAP@0.5: {map50:.3f}")
    print(f"mAP@0.5:0.95: {map2:.3f}")

    metrics = {
        "precision": mp,
        "recall": mr,
        "map@0.5": map50,
        "map@0.5:0.95": map2,
        "loss": loss,
        "times": times
    }

    out = output_dir +  "/" + key + "_" + "pt" + "_" "report.pdf"  # Concatenación directa

    generate_pdf_with_front_page(
        pdf_path=out,
        model_name=Path(input_dir).name,
        data_name=key,
        metrics = metrics,
        class_names=names,
        image_examples = image_examples, 
        class_colors=class_colors,
        metrics_classes=metrics_class,
        dataset_stats=dataset_stats,
        confidence_ranges=total_confidence_ranges,
        max_examples=max_examples
    )
    
    return None





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running the detection pipeline.")

    parser.add_argument('--input_dir', type=str, default="/app/pfs/eosinofilos",help="Directorio del proyecto")
    parser.add_argument('--output_dir', type=str,help="Directorio de salida")
    parser.add_argument('--img_size', type=int, default=1024,help="Tamaño de las imágenes para el modelo")
    parser.add_argument('--conf_thres', type=float, default=0.25,help="Umbral de confianza")
    parser.add_argument('--iou_thres', type=float, default=0.5,help="Umbral de IoU")
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
    names, class_colors = get_names_colors(model)


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
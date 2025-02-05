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
import test_onnx as test_onnx
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


def load_model(weights, device):
    device = select_device(device)

    session = ort.InferenceSession(weights, 
                                providers=["CUDAExecutionProvider" if device.type != 'cpu' else "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("Modelo cargado .onnx y dispositivo configurado.")
    return session, input_name, output_name

def get_names_colors(data_config):

    with open(data_config, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', []) 
    
    colors = {int(i): (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(names))}
    
    class_colors = {int(name): colors[i] for i, name in enumerate(names)}


    true_colors = {}
    for cls_id in range(len(names)):
        while True:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if color != colors[cls_id]:  # Asegurarse de que no coincidan
                true_colors[cls_id] = color
                break
    
    #class_colors = {name: tuple(color) for name, color in zip(names, colors)}

    print("Nombres de clases y colores configurados:")
    print("Clases:", names)
    print(class_colors)
    #print("Ejemplo de colores:", colors[:3])  # Imprimir algunos colores de ejemplo

    return names, class_colors

def configurar_rutas(input_dir, output_dir, key):

    input_dir = Path(input_dir)
    key_path = Path(key)
    output_dir = Path(output_dir)
    dataset_dir = input_dir / "dataset"
    base_path = Path(dataset_dir)
    save_dir = output_dir / "resultados" / "onnx"
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

def preprocesar_imagen(img, im0s, img_size):
    """
    Preprocesa una imagen para la inferencia en ONNX.
    """
    expected_size = (img_size, img_size)

    # Si la imagen está en formato (C, H, W), convertirla a (H, W, C)
    if img.shape[0] == 3 and len(img.shape) == 3:
        img = img.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
        print(f"Imagen transpuesta a: {img.shape}")

    if img is None:
        raise ValueError("La imagen no se cargó correctamente.")

    img = cv2.resize(img, expected_size, interpolation=cv2.INTER_LINEAR)

    if img.shape[-1] != 3:
        raise ValueError(f"La imagen tiene un número inesperado de canales: {img.shape}")

    img = img.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)

    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    return img


def realizar_inferencia(img, session, input_name, output_name, conf_thres, iou_thres, classes, agnostic_nms):
    """
    Realiza la inferencia en el modelo y aplica NMS.
    """
    preds = session.run([output_name], {input_name: img})[0]
    preds = torch.tensor(preds)  # Convertir a tensor

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
                    predicted_class = int(cls) 
                    print(predicted_class)
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
                    #label = f'{names[predicted_class]} {conf:.2f}'
                    print(f"Clase detectada: {predicted_class}, Color usado para pintar: {colors.get(predicted_class, 'No encontrado')}")
                    # Convierte el color de la caja de RGB a BGR
                    bgr_color = (colors[predicted_class][2], colors[predicted_class][1], colors[predicted_class][0])

                    # Usa el color en formato BGR para que coincida con la imagen
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

def main(input_dir, session, input_name, output_name, img_size,conf_thres, iou_thres, classes, augment, names, data_config, weights, output_dir, key, class_colors, max_examples=20):
    base_path,  labels_dir, labels_dir2, processed_images_dir = configurar_rutas(input_dir, output_dir, key)
    txt_file_path = obtener_ruta_desde_yaml(data_config, key=key)
    image_paths = leer_rutas_imagenes(base_path, txt_file_path)
    image_examples = []

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
                    label = int(line.strip().split()[0]) + 1
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
        dataset = LoadImages(img_path, img_size=img_size)

        
        #for path, img, im0s, vid_cap in dataset:
        path, img, im0s, vid_cap = next(iter(dataset))

        
        img = preprocesar_imagen(img, im0s, img_size)
        pred = realizar_inferencia(img, session, input_name=input_name, output_name=output_name, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes, agnostic_nms=False)
           
        txt_path = labels_dir2 / f"{Path(path).stem}.txt"
        image_examples = procesar_detecciones(pred, img, im0s, names,class_colors, txt_path, processed_images_dir, path, image_examples, expected_classes)

           # break

    test_onnx.opt = Opt(key)

    print(data_config)

    with open(data_config, 'r') as f:
        data = yaml.safe_load(f)  # Cargar el contenido del YAML como diccionario

    if not isinstance(data, dict):
        raise ValueError("Error: El archivo YAML no se cargó correctamente como un diccionario.")

    results, maps, times, metrics_class = test_onnx.test(
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

    out = output_dir +  "/" + key + "_" + "onnx" + "_" "report.pdf"  # Concatenación directa

    generate_pdf_with_front_page(
        pdf_path=out,
        model_name=Path(input_dir).name,
        data_name=key,
        metrics = metrics,
        class_names=names,
        image_examples = image_examples, 
        class_colors=class_colors,
        metrics_classes=metrics_class,
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
    
    session, input_name, output_name = load_model(args.weights, device)
    for input in session.get_inputs():
        print(f"Nombre de entrada: {input.name}, Forma esperada: {input.shape}, Tipo: {input.type}")
    names, class_colors = get_names_colors(data_config)


    #print(f"Device: {device}, Half precision: {half}")

    main(
            input_dir=input_dir_path,
            session = session,
            input_name=input_name,
            output_name=output_name,
            img_size=args.img_size,
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
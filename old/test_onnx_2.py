import argparse
import json
import cv2
import os
from pathlib import Path
from threading import Thread
import random  # Add this import
import logging  # Add this import

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt, plot_one_box  # Add plot_one_box here
from utils.torch_utils import select_device, time_synchronized, TracedModel



def calculate_iou(box1, box2):
    """
    Calcula IoU entre dos cajas con un margen de tolerancia
    """
    # Añadir margen de tolerancia (por ejemplo, 5 píxeles)
    margin = 5
    
    # Expandir las cajas ligeramente
    box1_expanded = [
        box1[0] - margin, box1[1] - margin,
        box1[2] + margin, box1[3] + margin
    ]
    
    # Calcular intersección
    x1 = max(box1_expanded[0], box2[0])
    y1 = max(box1_expanded[1], box2[1])
    x2 = min(box1_expanded[2], box2[2])
    y2 = min(box1_expanded[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    
    # Áreas de las cajas originales
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # IoU
    iou = intersection / float(box1_area + box2_area - intersection)
    return iou


def test(data,
         weights=None,
         batch_size=32,
         imgsz=1024,
         conf_thres=0.25,
         iou_thres=0.5,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         output_dir=None,  # Añadir este parámetro
         key='test'):     # Añadir este parámetro
    # Initialize/load model and set device
    training = model is not None
    model_name = str(data).split('/')[2]
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Modificar cómo se establece el directorio de salida
        if output_dir:
            save_dir = Path(output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

        # Load model
        import onnxruntime

        # Modificar esta parte para manejar weights como lista
        if isinstance(weights, list):
            weight_path = weights[0]  # Tomar el primer elemento si es lista
        else:
            weight_path = weights

        # Ahora usar weight_path para la verificación
        if weight_path.endswith(".onnx"):
            session = onnxruntime.InferenceSession(weight_path, providers=["CUDAExecutionProvider" if device.type != "cpu" else "CPUExecutionProvider"])
            logging.info(f"Input name: {session.get_inputs()[0].name}")
            
            input_name = session.get_inputs()[0].name

            output_name = session.get_outputs()[0].name

            for inp in session.get_inputs():
                expected_shape = inp.shape
            model = session
            is_onnx = True
        else:
            logging.info(f"🔵 Cargando modelo PyTorch: {weight_path}")
            model = attempt_load(weights, map_location=device)  # load FP32 model
            is_onnx = False

        if not isinstance(model, onnxruntime.InferenceSession):  
            gs = max(int(model.stride.max()), 32)  # Valor predeterminado de stride para YOLOv7 en ONNX
        else:
            gs = 32    
        
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        
        if trace and not is_onnx:
            model = TracedModel(model, device, imgsz)
    
    if not isinstance(model, onnxruntime.InferenceSession):
        half = device.type != 'cpu' and half_precision
        if half:
            model.half()
        model.eval()
    else:
        half = False  # ONNX no usa half-precision

    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            logging.info("\n=== Dataset Configuration ===")
            logging.info(f"Loaded data config: {data}")
            logging.info(f"Training path: {data.get('train')}")
            logging.info(f"Validation path: {data.get('val')}")
            logging.info(f"Test path: {data.get('test')}")
            logging.info(f"Number of classes: {data.get('nc')}")
            logging.info(f"Class names: {data.get('names')}")
            logging.info("==========================\n")

    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    logging.info(f"Number of classes: {nc}")  # Debug print

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            if not isinstance(model, onnxruntime.InferenceSession):
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        logging.info(f"\n=== Dataloader Configuration ===")
        logging.info(f"Task: {task}")
        logging.info(f"Dataset path: {data[task]}")
        
        try:
            dataloader = create_dataloader(
                data[task], 
                imgsz, 
                batch_size, 
                gs, 
                opt, 
                pad=0.5, 
                rect=True,
                prefix=colorstr(f'{task}: ')
            )[0]
            
            logging.info(f"Dataset size: {len(dataloader.dataset)} images")
            logging.info(f"Batch size: {batch_size}")
            logging.info(f"Image size: {imgsz}")
            
            # Print information about first batch
            batch = next(iter(dataloader))
            img, targets, paths, shapes = batch
        except Exception as e:
            logging.info(f"Error creating dataloader: {e}")
            raise

    if v5_metric:
        logging.info("Testing with YOLOv5 AP metric...")
    
    seen = 0
    total_predictions = 0  # Nuevo contador para predicciones
    confusion_matrix = ConfusionMatrix(nc=nc)
    # Para PyTorch: usar model.names
    if not isinstance(model, onnxruntime.InferenceSession):
        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # Para ONNX: definir nombres de clases genéricos
    else:
        names = {i: f"class_{i}" for i in range(nc)}  # Generar nombres de clases automáticamente

    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    # Inicializar variables para el seguimiento de las detecciones
    image_examples = []
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

    # Configurar directorios para imágenes procesadas y labels
    if output_dir:
        output_path = Path(output_dir) / "resultados" / "onnx" / key
        processed_images_dir = output_path / "processed_images"
        labels_dir = output_path / "labels"
        processed_images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    # Initialize colors after loading model and classes
    colors = {}
    used_colors = set()
    for i in range(nc):
        color = get_random_color(used_colors)
        colors[i] = color
        used_colors.add(color)

    # After loading model and before processing predictions
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()  # Add this line to define niou

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        if not isinstance(model, onnxruntime.InferenceSession):
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
        else:

            img = img.cpu().numpy()  # Mover a la CPU antes de convertir a NumPy

            # Redimensionar la imagen al tamaño esperado
            expected_size = (imgsz,imgsz) 
            resized_images = []
            # Iterar sobre todas las imágenes en el batch
            for i in range(img.shape[0]):  # Iterar sobre el batch (batch_size)
                single_img = img[i]  # Extraer la imagen i

                # Si la imagen está en formato (C, H, W), transpónlo a (H, W, C)
                if single_img.shape[0] == 3:
                    single_img = single_img.transpose(1, 2, 0)

                # Verificar que la imagen tiene exactamente 3 dimensiones antes de redimensionar
                if len(single_img.shape) != 3:
                    raise ValueError(f"Error: La imagen en el batch tiene una forma inesperada: {single_img.shape}")

                # Redimensionar la imagen
                resized_img = cv2.resize(single_img, expected_size, interpolation=cv2.INTER_LINEAR)
                
                # Agregar la imagen redimensionada a la lista
                resized_images.append(resized_img)

            # Convertir la lista de imágenes redimensionadas a un array NumPy con batch
            img = np.array(img)
            img = img.transpose(0, 3, 1, 2)  # Transponer a (C, H, W) para ONNX
           
            img = img.astype(np.float32) / 255.0
        targets = targets.to(device)

        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            if is_onnx:
                preds = model.run([output_name], {input_name: img})[0]
                out = torch.tensor(preds).to(device)  # Convertir a tensor para usar con el resto del código
                train_out = None  # No se usa en ONNX
            else:
                out, train_out = model(img, augment=augment)  # inference and training outputs
            # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            if is_onnx:
                out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)
            else:
                out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
                
            t1 += time_synchronized() - t

        # Contar predicciones totales
        for det in out:
            total_predictions += len(det)

        # Dentro del bucle principal, después de procesar las predicciones
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            
            if len(pred) > 0:
                # Scale coordinates
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

                # Para cada predicción, verificar IoU con ground truth
                for *xyxy, conf, cls in predn:
                    cls = int(cls)
                    is_correct = False

                    if nl:
                        # Convertir las coordenadas del ground truth
                        tbox = xywh2xyxy(labels[:, 1:5])
                        scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])

                        # Calcular IoU con todas las cajas ground truth de la misma clase
                        target_cls = labels[:, 0].tolist()
                        matching_targets = (cls == labels[:, 0]).nonzero(as_tuple=False).view(-1)
                        
                        if len(matching_targets) > 0:
                            xyxy_tensor = torch.tensor(xyxy, device=device).unsqueeze(0)
                            ious = box_iou(xyxy_tensor, tbox[matching_targets])
                            max_iou = ious.max().item()
                            is_correct = max_iou > iou_thres

                    # Actualizar rangos de confianza
                    conf_value = float(conf) * 100
                    for range_key in total_confidence_ranges:
                        min_val, max_val = [int(x) for x in range_key.split('-')]
                        if min_val <= conf_value < max_val:
                            if is_correct:
                                total_confidence_ranges[range_key]['correct'] += 1
                            else:
                                total_confidence_ranges[range_key]['incorrect'] += 1
                            break

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # Cargar imagen original usando cv2 directamente
            im0s = cv2.imread(str(paths[si]))
            if im0s is None:
                logging.info(f"Error loading image: {paths[si]}")
                continue
            
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])

            # PRIMERO: Procesar todas las predicciones y guardar sus áreas
            prediction_boxes = []
            if len(pred):
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
                for *xyxy, conf, cls in predn:
                    try:
                        box = [int(x.item()) for x in xyxy]
                        prediction_boxes.append({
                            'box': box,
                            'conf': conf,
                            'cls': cls
                        })
                    except Exception as e:
                        logging.info(f"Error al procesar predicción: {e}")
                        continue

            # SEGUNDO: Procesar ground truth y marcar cuáles tienen solapamiento
            matched_gt = set()
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                
                # Para cada ground truth, buscar si tiene solapamiento con alguna predicción
                for i in range(len(tbox)):
                    box = tbox[i]
                    try:
                        x1, y1, x2, y2 = [int(float(coord)) for coord in box]
                        gt_box = [x1, y1, x2, y2]
                        
                        # Verificar solapamiento con cualquier predicción
                        for pred_info in prediction_boxes:
                            if calculate_iou(gt_box, pred_info['box']) > iou_thres:
                                matched_gt.add(i)
                                break
                        
                        # Solo dibujar si NO hay solapamiento (ground truth no detectado)
                        if i not in matched_gt:
                            cv2.rectangle(im0s, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    except Exception as e:
                        logging.info(f"Error al procesar ground truth: {e}")
                        continue

            # TERCERO: Procesar y dibujar predicciones
            if len(pred):
                # Guardar predicciones en archivo .txt
                if output_dir:
                    txt_path = labels_dir / f"{Path(path).stem}.txt"
                    with open(txt_path, "w") as f:
                        for *xyxy, conf, cls in predn:
                            # Convertir bbox a formato normalizado
                            x1, y1, x2, y2 = [float(x) for x in xyxy]
                            w = (x2 - x1) / im0s.shape[1]
                            h = (y2 - y1) / im0s.shape[0]
                            x_center = ((x1 + x2) / 2) / im0s.shape[1]
                            y_center = ((y1 + y2) / 2) / im0s.shape[0]
                            
                            f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

                # Procesar y dibujar cada detección
                for *xyxy, conf, cls in predn:
                    cls = int(cls)
                    is_correct = False

                    # Verificar IoU con ground truth
                    if nl:
                        xyxy_tensor = torch.tensor(xyxy, device=device).unsqueeze(0)
                        tbox = tbox.to(device)
                        box_gt = box_iou(xyxy_tensor, tbox)
                        max_iou, _ = box_gt.max(1)
                        is_correct = max_iou > iou_thres

                    # Dibujar caja según si es correcta o no
                    bgr_color = (colors[cls][2], colors[cls][1], colors[cls][0])
                    if not is_correct:
                        plot_striped_box(im0s, xyxy, bgr_color, (0,0,255), conf=conf)
                    else:
                        plot_one_box(xyxy, im0s, color=bgr_color, line_thickness=2)

                # Guardar imagen procesada
                if output_dir:
                    save_path = processed_images_dir / f"{Path(paths[si]).stem}_processed.jpg"
                    cv2.imwrite(str(save_path), im0s)

                    # Almacenar ejemplo para reporte
                    expected_classes = {int(l): len(labels[labels[:, 0] == l]) for l in labels[:, 0].unique()} if nl else {}
                    detected_classes = {int(cls): len(pred[pred[:, 5] == cls]) for cls in pred[:, 5].unique()}
                    image_examples.append((str(save_path), expected_classes, detected_classes))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    if len(stats) == 0:
        logging.info("\nWARNING: No valid detections found. Check your dataset path and annotations.")
        # Initialize empty metrics for report
        stats = [[np.array([])], [np.array([])], [np.array([])], [np.array([])]]

    stats = [np.concatenate(x, 0) for x in zip(*stats)] if stats else []

    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
        mp, mr, map50, map = 0., 0., 0., 0.
        p = r = ap50 = ap = np.array([0.])  # Inicializar arrays vacíos
        ap_class = []  # Lista vacía para clases

    class_metrics = {}

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    logging.info('\nMétricas Globales:')
    logging.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        logging.info('\nMétricas por Clase:')
        for i, c in enumerate(ap_class):
            logging.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            # Guardar métricas en el diccionario
            class_metrics[names[c]] = {
                "precision": float(p[i]),  # Convertir a float para asegurar serialización
                "recall": float(r[i]),
                "map@0.5": float(ap50[i]),
                "map@0.5:0.95": float(ap[i]),
                "class_name": names[c],
            }

    # Asegurarse de que tenemos al menos las métricas globales
    if not class_metrics:
        class_metrics["all"] = {
            "precision": float(mp),
            "recall": float(mr),
            "map@0.5": float(map50),
            "map@0.5:0.95": float(map),
            "class_name": "all",
        }

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        logging.info('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        logging.info('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            logging.info(f'pycocotools unable to run: {e}')

    # Añadir contadores de imágenes
    total_images = len(dataloader.dataset)  # Total de imágenes en el dataset
    total_processed = seen  # Imágenes procesadas

    # Añadir el conteo a las métricas de retorno
    dataset_stats = {
        'total_images': total_images,
        'processed_images': total_processed,
        'total_labels': int(nt.sum()),
        'total_predictions': total_predictions  # Nueva métrica
    }

    # Generar reporte
    if output_dir:
        from object_detection_cellsia.report import generate_pdf_with_front_page
        out = Path(output_dir) / f"{key}_onnx_report.pdf"
        
        # Convertir metrics a dict        
        metrics_dict = {
            "precision": mp,
            "recall": mr,
            "map@0.5": map50,
            "map@0.5:0.95": map,
            "loss": loss.cpu() / len(dataloader) if len(dataloader) > 0 else 0,
            "times": t
        }
        
        generate_pdf_with_front_page(
            pdf_path=str(out),
            model_name=model_name,
            data_name=key,
            metrics=metrics_dict,
            class_names=names,
            image_examples=image_examples if 'image_examples' in locals() else [],
            class_colors=colors if 'colors' in locals() else {},
            metrics_classes=class_metrics,
            dataset_stats=dataset_stats,
            confidence_ranges=total_confidence_ranges if 'total_confidence_ranges' in locals() else {}
        )

    # Return results
    #model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        logging.info(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t, class_metrics, dataset_stats


def get_random_color(used_colors):
    """
    Genera un color aleatorio que no esté en la lista de usados, evitando colores específicos
    """
    def is_forbidden_color(color):
        r, g, b = color
        
        # Detectar grises (diferencia entre canales menor a 50)
        is_grey = abs(r - g) < 50 and abs(g - b) < 50 and abs(r - b) < 50
        
        # Detectar rosas (R alto, B medio-alto)
        is_pink = r > 180 and b > 100 and g < 180
        
        # Detectar color carne (R alto, G medio-alto, B bajo-medio)
        is_skin = r > 200 and 120 < g < 190 and 80 < b < 150
        
        # Detectar morados (R y B altos, G bajo)
        is_purple = r > 80 and b > 80 and g < min(r, b)
        
        # Detectar verdes (G dominante)
        is_green = g > max(r, b)
        
        # Detectar rojos (R dominante)
        is_red = r > max(g, b) + 30
        
        # Detectar colores muy claros (todos los canales altos)
        is_too_light = min(r, g, b) > 200
        
        # Detectar colores muy oscuros (todos los canales bajos)
        is_too_dark = max(r, g, b) < 50

        return (is_grey or is_pink or is_skin or is_purple or 
                is_green or is_red or is_too_light or is_too_dark)

    while True:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if not is_forbidden_color(color) and color not in used_colors:
            return color


def plot_striped_box(img, xyxy, color1, color2, conf=None, line_thickness=2, stripe_length=15):
    """
    Dibuja una caja con rayas para detecciones incorrectas
    """
    try:
        # Usar list comprehension en lugar de map
        x1, y1, x2, y2 = [int(coord) for coord in xyxy]
        
        # Dibujar tres lados completos
        cv2.line(img, (x1, y2), (x2, y2), color1, line_thickness)  # inferior
        cv2.line(img, (x1, y1), (x1, y2), color1, line_thickness)  # izquierda
        cv2.line(img, (x2, y1), (x2, y2), color1, line_thickness)  # derecha
        
        # Dibujar patrón rayado superior
        total_width = x2 - x1
        current_x = x1
        is_red = True
        
        while current_x < x2:
            end_x = min(current_x + stripe_length, x2)
            color = color2 if is_red else color1
            cv2.line(img, (current_x, y1), (end_x, y1), color, line_thickness)
            current_x = end_x
            is_red = not is_red
        
        if conf is not None:
            conf_str = f"{conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(conf_str, font, font_scale, font_thickness)
            margin = 2
            text_x = x1
            text_y = y1 - margin
            
            cv2.rectangle(img, 
                         (text_x, text_y - text_height - margin),
                         (text_x + text_width + margin * 2, text_y + margin),
                         color2, -1)
            cv2.putText(img, conf_str, (text_x + margin, text_y - margin), 
                        font, font_scale, (255, 255, 255), font_thickness)
    except Exception as e:
        logging.info(f"Error en plot_striped_box: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.10, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single_cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save_conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save_json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no_trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5_metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    
    # Añadir nuevos argumentos para el guardado de imágenes y reporte
    parser.add_argument('--output_dir', type=str, help='directorio para guardar resultados')
    parser.add_argument('--save_images', action='store_true', help='guardar imágenes con detecciones')
    parser.add_argument('--save_report', action='store_true', help='generar reporte PDF')
    parser.add_argument('--max_examples', type=int, default=20, help='número máximo de ejemplos en el reporte')

    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(data=opt.data,
             weights=opt.weights,
             batch_size=opt.batch_size,
             imgsz=opt.img_size,
             conf_thres=opt.conf_thres,
             iou_thres=opt.iou_thres,
             save_json=opt.save_json,
             single_cls=opt.single_cls,
             augment=opt.augment,
             verbose=opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             output_dir=opt.output_dir if (opt.save_images or opt.save_report) else None,
             key=opt.task
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, opt.conf_thres, opt.iou_thres, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                logging.info(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot

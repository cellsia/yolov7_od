import subprocess
import argparse
import sys
from pathlib import Path
import os
import shutil

def find_onnx_file(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".onnx"):
                return os.path.join(root, file)
    return None  # Devuelve None si no encuentra ningún archiv

def clean_old_experiments(output_dir, latest_experiment):
    """
    Elimina todas las carpetas de yolo_experiment excepto la última creada.
    """
    yolo_experiments = sorted(output_dir.glob("yolo_experiment*"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    for experiment in yolo_experiments:
        if experiment != latest_experiment:
            print(f"Eliminando carpeta de experimento anterior: {experiment}")
            shutil.rmtree(experiment)

def main():
    parser = argparse.ArgumentParser(description="Script principal para ejecutar entrenamiento y pruebas de YOLOv7.")

    parser.add_argument('--input_dir', type=str, required=True, help="Directorio del proyecto (datos de entrada).")
    parser.add_argument('--output_dir', type=str, default="/app/output", help="Directorio para guardar resultados del entrenamiento y pruebas.")
    parser.add_argument('--epochs', type=int, default=200, help="Número de épocas para el entrenamiento.")
    parser.add_argument('--img_size', type=int, default=1024, help="Tamaño de las imágenes para el modelo.")
    parser.add_argument('--batch', type=int, default=16, help="Tamaño del batch para el entrenamiento.")
    parser.add_argument('--early_stopping_patience', type=int, default=50, help="Paciencia para early stopping en el entrenamiento.")
    #test
    parser.add_argument('--use_onnx', type=bool, default=False ,help="Cambiar a True si se usa un modelo ONNX")
    parser.add_argument('--conf_thres', type=float, default=0.25,help="Umbral de confianza")
    parser.add_argument('--iou_thres', type=float, default=0.45,help="Umbral de IoU")
    
    args = parser.parse_args()

    # Comprobación de directorios
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        print(f"Error: El directorio de entrada {input_dir} no existe.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Paso 1: Entrenamiento
    print("\n================= INICIO DEL ENTRENAMIENTO =================")
    train_cmd = [
        "python", "/app/yolov7/object_detection_cellsia/run_od_2.py",  # Cambiar al nombre real del script de entrenamiento
        str(input_dir),
        "--output_dir", str(output_dir),
        "--epochs", str(args.epochs),
        "--img_size", str(args.img_size),
        "--batch", str(args.batch),
        "--early_stopping_patience", str(args.early_stopping_patience)
    ]

    try:
        subprocess.run(train_cmd, check=True)
        print("\nEntrenamiento completado con éxito.")
    except subprocess.CalledProcessError as e:
        print(f"Error durante el entrenamiento: {e}")
        sys.exit(1)
    
    # Determinar el path del último archivo best.pt generado
    yolo_experiments = sorted(output_dir.glob("yolo_experiment*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not yolo_experiments:
        print("Error: No se encontraron directorios de experimentos en yolo_experiment.")
        sys.exit(1)

    latest_experiment = yolo_experiments[0]
    best_weights_path = latest_experiment / "weights" / "best.pt"
    if not best_weights_path.is_file():
        print(f"Error: No se encontró el archivo de pesos {best_weights_path}")
        sys.exit(1)
    else:
        print(f"Archivo de pesos best.pt encontrado en: {best_weights_path}")

    clean_old_experiments(output_dir, latest_experiment)
    print(f"Sólo se ha conservado la carpeta: {latest_experiment}")

    # Paso 2: Pruebas
    print("\n================= INICIO DE LAS PRUEBAS =================")
    if args.use_onnx:
        onnx_file = find_onnx_file(output_dir)
        if onnx_file:
            print(f"Ruta del archivo ONNX: {onnx_file}")
            
            train_onnx_cmd = [
                "python", "/app/yolov7/object_detection_cellsia/inference_onnx.py",  # Cambiar al nombre real del script de pruebas
                "--input_dir", str(input_dir),
                "--img_size", str(args.img_size),
                "--conf_thres", str(args.conf_thres),
                "--iou_thres", str(args.iou_thres),
                "--weights", str(onnx_file),
                "--output_dir", str(output_dir),
                "--key", str("train")
            ]
            test_onnx_cmd = [
                "python", "/app/yolov7/object_detection_cellsia/inference_onnx.py",  # Cambiar al nombre real del script de pruebas
                "--input_dir", str(input_dir),
                "--img_size", str(args.img_size),
                "--conf_thres", str(args.conf_thres),
                "--iou_thres", str(args.iou_thres),
                "--weights", str(onnx_file),
                "--output_dir", str(output_dir),
                "--key", str("test")
            ]
            val_onnx_cmd = [
                "python", "/app/yolov7/object_detection_cellsia/inference_onnx.py",  # Cambiar al nombre real del script de pruebas
                "--input_dir", str(input_dir),
                "--img_size", str(args.img_size),
                "--conf_thres", str(args.conf_thres),
                "--iou_thres", str(args.iou_thres),
                "--weights", str(onnx_file),
                "--output_dir", str(output_dir),
                "--key", str("val")
            ]
            try:
                subprocess.run(train_onnx_cmd, check=True)
                print("\nPruebas completadas con éxito.")
                subprocess.run(test_onnx_cmd, check=True)
                print("\nPruebas completadas con éxito.")
                subprocess.run(val_onnx_cmd, check=True)
                print("\nPruebas completadas con éxito.")
            except subprocess.CalledProcessError as e:
                print(f"Error durante las pruebas: {e}")
                sys.exit(1)
        else:
            print("No se encontró ningún archivo ONNX.")

    else:
        train_pt_cmd = [
            "python", "/app/yolov7/object_detection_cellsia/inference.py",  # Cambiar al nombre real del script de pruebas
            "--input_dir", str(input_dir),
            "--img_size", str(args.img_size),
            "--conf_thres", str(args.conf_thres),
            "--iou_thres", str(args.iou_thres),
            "--weights", str(best_weights_path),
            "--output_dir", str(output_dir),
            "--key", str("train")
        ]
        test_pt_cmd = [
            "python", "/app/yolov7/object_detection_cellsia/inference.py",  # Cambiar al nombre real del script de pruebas
            "--input_dir", str(input_dir),
            "--img_size", str(args.img_size),
            "--conf_thres", str(args.conf_thres),
            "--iou_thres", str(args.iou_thres),
            "--weights", str(best_weights_path),
            "--output_dir", str(output_dir),
            "--key", str("test")
        ]
        val_pt_cmd = [
            "python", "/app/yolov7/object_detection_cellsia/inference.py",  # Cambiar al nombre real del script de pruebas
            "--input_dir", str(input_dir),
            "--img_size", str(args.img_size),
            "--conf_thres", str(args.conf_thres),
            "--iou_thres", str(args.iou_thres),
            "--weights", str(best_weights_path),
            "--output_dir", str(output_dir),
            "--key", str("val")
        ]
        try:
            subprocess.run(train_pt_cmd, check=True)
            print("\nPruebas completadas con éxito.")
            subprocess.run(test_pt_cmd, check=True)
            print("\nPruebas completadas con éxito.")
            subprocess.run(val_pt_cmd, check=True)
            print("\nPruebas completadas con éxito.")
        except subprocess.CalledProcessError as e:
            print(f"Error durante las pruebas: {e}")
            sys.exit(1)

    print("\nPipeline completado.")

if __name__ == "__main__":
    main()


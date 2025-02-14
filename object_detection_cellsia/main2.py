
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
    return None  # Devuelve None si no encuentra ningún archivo ONNX

def clean_old_experiments(output_dir, latest_experiment):
    """
    Elimina todas las carpetas de yolo_experiment excepto la última creada.
    """
    yolo_experiments = sorted(output_dir.glob("yolo_experiment*"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    for experiment in yolo_experiments:
        if experiment != latest_experiment:
            print(f"Eliminando carpeta de experimento anterior: {experiment}")
            shutil.rmtree(experiment)

def run_inference(model_type, weights, input_dir, output_dir, img_size, conf_thres, iou_thres, datasets):
    script = "/app/yolov7/object_detection_cellsia/inference_onnx.py" if model_type == "onnx" else "/app/yolov7/object_detection_cellsia/inference.py"
    
    for dataset in datasets:
        cmd = [
            "python", script,
            "--input_dir", str(input_dir),
            "--img_size", str(img_size),
            "--conf_thres", str(conf_thres),
            "--iou_thres", str(iou_thres),
            "--weights", str(weights),
            "--output_dir", str(output_dir),
            "--key", dataset
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"\nInferencia en {dataset} completada con éxito.")
        except subprocess.CalledProcessError as e:
            print(f"Error durante la inferencia en {dataset}: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Script principal para entrenamiento y pruebas de YOLOv7.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directorio del proyecto (datos de entrada).")
    parser.add_argument('--output_dir', type=str, default="/app/output", help="Directorio para resultados.")
    parser.add_argument('--epochs', type=int, default=200, help="Número de épocas para el entrenamiento.")
    parser.add_argument('--img_size', type=int, default=1024, help="Tamaño de imágenes.")
    parser.add_argument('--batch', type=int, default=16, help="Tamaño del batch.")
    parser.add_argument('--early_stopping_patience', type=int, default=50, help="Paciencia para early stopping.")
    parser.add_argument('--conf_thres', type=float, default=0.25, help="Umbral de confianza.")
    parser.add_argument('--iou_thres', type=float, default=0.45, help="Umbral de IoU.")
    parser.add_argument('--run_train', action='store_true', help="Ejecutar inferencia en el conjunto train.")
    parser.add_argument('--run_test', action='store_true', help="Ejecutar inferencia en el conjunto test.")
    parser.add_argument('--run_val', action='store_true', help="Ejecutar inferencia en el conjunto val.")
    parser.add_argument('--run_all', action='store_true', help="Ejecutar inferencia en todos los conjuntos.")
    parser.add_argument('--use_onnx', action='store_true', help="Usar modelo ONNX en lugar de PyTorch.")
    parser.add_argument('--use_both', action='store_true', help="Ejecutar inferencia con ambos modelos: ONNX y PyTorch.")
    
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.is_dir():
        print(f"Error: El directorio de entrada {input_dir} no existe.")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n================= INICIO DEL ENTRENAMIENTO =================")
    train_cmd = [
        "python", "/app/yolov7/object_detection_cellsia/run_od_2.py",
        str(input_dir), "--output_dir", str(output_dir),
        "--epochs", str(args.epochs), "--img_size", str(args.img_size),
        "--batch", str(args.batch), "--early_stopping_patience", str(args.early_stopping_patience)
    ]
    
    try:
        subprocess.run(train_cmd, check=True)
        print("\nEntrenamiento completado con éxito.")
    except subprocess.CalledProcessError as e:
        print(f"Error durante el entrenamiento: {e}")
        sys.exit(1)
    
    yolo_experiments = sorted(output_dir.glob("yolo_experiment*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not yolo_experiments:
        print("Error: No se encontraron directorios de experimentos en yolo_experiment.")
        sys.exit(1)
    latest_experiment = yolo_experiments[0]
    best_weights_path = latest_experiment / "weights" / "best.pt"
    if not best_weights_path.is_file():
        print(f"Error: No se encontró el archivo de pesos {best_weights_path}")
        sys.exit(1)
    
    clean_old_experiments(output_dir, latest_experiment)
    datasets_to_run = ["train", "test", "val"] if args.run_all else []
    if args.run_train:
        datasets_to_run.append("train")
    if args.run_test:
        datasets_to_run.append("test")
    if args.run_val:
        datasets_to_run.append("val")
    
    if not datasets_to_run:
        print("No se seleccionó ningún conjunto para inferencia. Saliendo.")
        sys.exit(0)
    
    if args.use_onnx or args.use_both:
        onnx_file = find_onnx_file(output_dir)
        if onnx_file:
            run_inference("onnx", onnx_file, input_dir, output_dir, args.img_size, args.conf_thres, args.iou_thres, datasets_to_run)
    
    if not args.use_onnx or args.use_both:
        run_inference("pt", best_weights_path, input_dir, output_dir, args.img_size, args.conf_thres, args.iou_thres, datasets_to_run)
    
    print("\nPipeline completado con éxito.")

if __name__ == "__main__":
    main()

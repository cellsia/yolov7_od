import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
os.chdir("/app/yolov7")

def replace_relative_paths(yaml_file, input_dir):
    print(f"Reemplazando '../' en {yaml_file} con {input_dir}...")
    with open(yaml_file, 'r') as file:
        content = file.read()
    content = content.replace('../', f"{input_dir}/")
    with open(yaml_file, 'w') as file:
        file.write(content)
    print(f"Reemplazo completado: '../' cambiado por '{input_dir}' en {yaml_file}.")

def convert_to_trt(onnx_path, output_trt_path, precision="fp16"):
    """
    Convierte un modelo ONNX a TensorRT utilizando el script export.py de tensorrt-python.

    Args:
        onnx_path (str): Ruta al archivo ONNX.
        output_trt_path (str): Ruta donde se guardará el modelo TensorRT.
        precision (str): Precisión a usar (por defecto "fp16", también puede ser "fp32").
    """

    # Verificar si el archivo ONNX existe
    print("hola")
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        print(f"Error: El archivo ONNX '{onnx_path}' no existe.")
        return

    # Asegurar que el directorio de salida exista
    output_dir = Path(output_trt_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clonar tensorrt-python si no está presente
    repo_path = Path("tensorrt-python")
    if not repo_path.exists():
        print("Clonando tensorrt-python...")
        subprocess.run(["git", "clone", "https://github.com/Linaom1214/tensorrt-python.git"], check=True)
    else:
        print("Repositorio tensorrt-python ya clonado, continuando...")

    # Ejecutar la conversión
    print(f"Convirtiendo {onnx_path} a TensorRT ({precision})...")
    subprocess.run([
        "python", str(repo_path / "export.py"),
        "-o", str(onnx_path),
        "-e", str(output_trt_path),
        "-p", "fp16"
    ], check=True)

    print(f"Conversión completada: {output_trt_path}")


def convert_to_absolute(yaml_file, base_dir):
    print(f"Procesando archivo YAML: {yaml_file}")
    temp_file = f"{yaml_file}.tmp"
    with open(yaml_file, 'r') as infile, open(temp_file, 'w') as outfile:
        for line in infile:
            if line.strip().startswith(('train:', 'val:', 'test:')):
                key, relative_path = line.split(':', 1)
                relative_path = relative_path.strip()
                if relative_path.startswith('./') or relative_path.startswith('../'):
                    absolute_path = os.path.normpath(os.path.join(base_dir, relative_path))
                    outfile.write(f"{key}: {absolute_path}\n")
                else:
                    outfile.write(line)
            else:
                outfile.write(line)
    os.replace(temp_file, yaml_file)
    print(f"Rutas convertidas a absolutas en {yaml_file}")


def clean_cache_files(base_dir):
    print(f"Eliminando archivos .cache en {base_dir}...")
    for cache_file in Path(base_dir).rglob("*.cache"):
        cache_file.unlink()
    print("Eliminación de archivos .cache completada.")


def train_yolov7(data_config, output_dir, epochs, img_size, weights, batch, early_stopping_patience):
    print("Iniciando el entrenamiento de YOLOv7...")
    subprocess.run([
        "python", "train_early.py",
        "--img", str(img_size),
        "--batch", str(batch),
        "--epochs", str(epochs),
        "--data", str(data_config),
        "--weights", str(weights),
        "--project", output_dir,
        "--name", "yolo_experiment",
        "--hyp", "/data/hyp.scratch.custom.yaml",
        "--patience", str(early_stopping_patience)
    ])
    print("Entrenamiento completado.")

def save_best_model(output_dir):
    experiments = sorted(Path(output_dir).glob("yolo_experiment*"), key=os.path.getmtime, reverse=True)
    if not experiments:
        print(f"No se encontró ningún experimento en {output_dir}")
        sys.exit(1)

    latest_experiment = experiments[0]
    print(f"Último experimento encontrado: {latest_experiment}")

    best_weights_path = latest_experiment / "weights/best.pt"

    return best_weights_path, latest_experiment

def convert_to_onnx(weights_path):
    print("Convirtiendo a ONNX...")
    subprocess.run([
        "python", "export.py",
        "--weights", str(weights_path),"--img-size", "1024",
        "--grid", "--dynamic", "--simplify"
    ])



def clean_temp_files(output_dir):
    print("Limpiando archivos temporales y caché...")
    for temp_file in Path(output_dir).rglob("*.tmp"):
        temp_file.unlink()
    for log_file in Path(output_dir).rglob("*.log"):
        log_file.unlink()
    print("Limpieza completada.")


def main():
    parser = argparse.ArgumentParser(
        description="Script para entrenar YOLOv7 con argumentos configurables"
    )
    
    parser.add_argument("input_dir", type=str, help="Directorio de entrada donde están las imágenes")
    parser.add_argument("--output_dir", type=str,  help="Directorio de salida para guardar resultados")
    parser.add_argument("--epochs", type=int, default=200, help="Número de épocas para entrenar (default: 50)")
    parser.add_argument("--img_size", type=int, default=1024, help="Tamaño de las imágenes (default: 1024)")
    parser.add_argument("--batch", type=int, default=16, help="Tamaño del batch (default: 16)" )
    parser.add_argument("--early_stopping_patience", type=int, default=50, help="Patience para early stopping (default: 10)")
    parser.add_argument("--weights", type=str, default="/app/yolov7_training.pt", help="Ruta a los pesos preentrenados (default: ../yolov7_training.pt)" )

    args = parser.parse_args()

    print(f"Directorio de entrada: {args.input_dir}")
    print(f"Directorio de salida: {args.output_dir}")
    print(f"Épocas: {args.epochs}")
    print(f"Tamaño de imagen: {args.img_size}")
    print(f"Batch size: {args.batch}")
    print(f"Patience de early stopping: {args.early_stopping_patience}")
    print(f"Pesos: {args.weights}")

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

    # Procesar archivo YAML
    replace_relative_paths(data_config, str(input_dir_path))
    # Pasar a rutas absolutas los directorios que apuntan a las imagenes
    convert_to_absolute(data_config, str(input_dir_path))

    clean_cache_files(args.input_dir)

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        print(f"El archivo de pesos preentrenados {args.weights} no existe.")
        sys.exit(1)
    print(f"Archivo de pesos preentrenados encontrado: {args.weights}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_yolov7(data_config, args.output_dir, args.epochs, args.img_size, weights_path, args.batch, args.early_stopping_patience)

    best_weights_path, latest_exp = save_best_model(args.output_dir)

    convert_to_onnx(best_weights_path)

    onnx_file = latest_exp / "weights/best.onnx"
    
    print(f"Modelo ONNX guardado en {onnx_file}")
    '''
    trt_file = latest_exp / "best_model.trt"
    convert_to_trt(onnx_file, trt_file)
    '''


if __name__ == "__main__":
    main()







import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging
from logging_utils import setup_global_logging
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ignore specific PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message="weights_only")

os.chdir("/app/yolov7")

def replace_relative_paths(yaml_file, input_dir):
    logging.info(f"Replacing '../' in {yaml_file} with {input_dir}...")
    with open(yaml_file, 'r') as file:
        content = file.read()
    content = content.replace('../', f"{input_dir}/")
    with open(yaml_file, 'w') as file:
        file.write(content)
    logging.info(f"Replacement completed: '../' changed to '{input_dir}' in {yaml_file}.")

def convert_to_trt(onnx_path, output_trt_path, precision="fp16"):
    """
    Converts an ONNX model to TensorRT using the tensorrt-python export.py script.

    Args:
        onnx_path (str): Path to the ONNX file.
        output_trt_path (str): Path where the TensorRT model will be saved.
        precision (str): Precision to use (default "fp16", can also be "fp32").
    """

    # Check if the ONNX file exists
    logging.info("hello")
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        logging.error(f"Error: ONNX file '{onnx_path}' does not exist.")
        return

    # Ensure the output directory exists
    output_dir = Path(output_trt_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone tensorrt-python if not present
    repo_path = Path("tensorrt-python")
    if not repo_path.exists():
        logging.info("Cloning tensorrt-python...")
        subprocess.run(["git", "clone", "https://github.com/Linaom1214/tensorrt-python.git"], check=True)
    else:
        logging.info("tensorrt-python repository already cloned, continuing...")

    # Run the conversion
    logging.info(f"Converting {onnx_path} to TensorRT ({precision})...")
    subprocess.run([
        "python", str(repo_path / "export.py"),
        "-o", str(onnx_path),
        "-e", str(output_trt_path),
        "-p", "fp16"
    ], check=True)

    logging.info(f"Conversion completed: {output_trt_path}")


def convert_to_absolute(yaml_file, base_dir):
    logging.info(f"Processing YAML file: {yaml_file}")
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
    logging.info(f"Paths converted to absolute in {yaml_file}")


def clean_cache_files(base_dir):
    logging.info(f"Removing .cache files in {base_dir}...")
    for cache_file in Path(base_dir).rglob("*.cache"):
        cache_file.unlink()
    logging.info("Cache files removal completed.")


def train_yolov7(data_config, output_dir, epochs, img_size, weights, batch, early_stopping_patience, conf_thres, iou_thres):
    logger = logging.getLogger('run_od_2')
    logger.info("\n")
    logger.info("**************Starting YOLOv7 Training***************")
    try:
        process = subprocess.Popen([
            "python", "train_early.py",
            "--img", str(img_size),
            "--batch", str(batch),
            "--epochs", str(epochs),
            "--data", str(data_config),
            "--weights", str(weights),
            "--project", output_dir,
            "--name", "yolo_experiment",
            "--hyp", "/data/hyp.scratch.p6.yaml",
            "--patience", str(early_stopping_patience),
            "--conf-thres", str(conf_thres),
            "--iou-thres", str(iou_thres),
            "--adam"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        for line in process.stdout:
            line = line.strip()
            if line:
                if any(keyword in line for keyword in ['Epoch', 'Loss', 'Metrics', 'Learning Rate', 'epoch', 'stopping', 'best', 'Best', 'early', 'Early']):
                    logger.info(line)
                if 'error' in line.lower():
                    logger.error(line)
                elif 'warning' in line.lower():
                    logger.warning(line)
                else:
                    logger.debug(line)

        process.wait()
        if process.returncode != 0:
            logging.error("Training failed")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        sys.exit(1)

    logger.info("******************End of Training*******************")
    logger.info("\n")

def save_best_model(output_dir):
    experiments = sorted(Path(output_dir).glob("yolo_experiment*"), key=os.path.getmtime, reverse=True)
    if not experiments:
        logging.error(f"No experiments found in {output_dir}")
        sys.exit(1)

    latest_experiment = experiments[0]
    logging.info(f"Latest experiment found: {latest_experiment}")

    best_weights_path = latest_experiment / "weights/best.pt"

    return best_weights_path, latest_experiment

def convert_to_onnx(weights_path):
    logging.info("Converting to ONNX...")
    subprocess.run([
        "python", "export.py",
        "--weights", str(weights_path),"--img-size", "1024",
        "--grid", "--dynamic", "--simplify",  "--batch-size", "1"
    ])



def clean_temp_files(output_dir):
    logger.info("\n")
    logging.info("Cleaning temporary files and cache...")
    for temp_file in Path(output_dir).rglob("*.tmp"):
        temp_file.unlink()
    for log_file in Path(output_dir).rglob("*.log"):
        log_file.unlink()
    logging.info("Cleanup completed.")
    logger.info("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Script to train YOLOv7 with configurable arguments"
    )
    
    parser.add_argument("input_dir", type=str, help="Input directory containing images")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training (default: 50)")
    parser.add_argument("--img_size", type=int, default=1024, help="Image size (default: 1024)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--early_stopping_patience", type=int, default=50, help="Early stopping patience (default: 10)")
    parser.add_argument("--weights", type=str, default="/app/yolov7-w6.pt", help="Path to pretrained weights")
    parser.add_argument("--conf_thres", type=float, default=0.10, help="Confidence threshold for detections")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IoU threshold for NMS")

    args = parser.parse_args()

    # Configure global logging
    log_file = setup_global_logging(args.output_dir)
    logger = logging.getLogger('run_od_2')
    logger.info(f"Logs will be saved to: {log_file}")
    logger.info("\n")

    logger.info("========Training Information========")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Image size: {args.img_size}")
    logger.info(f"Batch size: {args.batch}")
    logger.info(f"Early stopping patience: {args.early_stopping_patience}")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Confidence threshold: {args.conf_thres}")
    logger.info(f"IoU threshold: {args.iou_thres}")
    logger.info("=====================================")
    logger.info("\n")

    logger.info("=========== Init Training ===========")

    input_dir_path = Path(args.input_dir)
    if not input_dir_path.is_dir():
        logger.error(f"Input directory {args.input_dir} does not exist.")
        sys.exit(1)

    yaml_files = list(input_dir_path.rglob("*.yaml"))
    if not yaml_files:
        logger.error(f"No .yaml file found in input directory {args.input_dir}.")
        sys.exit(1)

    data_config = yaml_files[0]
    logger.info(f"Preparing data files...")
    logger.info(f"Configuration file found: {data_config}")

    # Process YAML file
    replace_relative_paths(data_config, str(input_dir_path))
    # Convert to absolute paths the directories that point to the images
    convert_to_absolute(data_config, str(input_dir_path))

    clean_cache_files(args.input_dir)

    logger.info("Searching for weights file...")
    weights_path = Path(args.weights)
    if not weights_path.is_file():
        logger.error(f"Pretrained weights file {args.weights} does not exist.")
        sys.exit(1)
    logger.info(f"Pretrained weights file found: {args.weights}")
    logger.info("=====================================")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_yolov7(data_config, args.output_dir, args.epochs, args.img_size, weights_path, args.batch, args.early_stopping_patience, args.conf_thres, args.iou_thres)

    best_weights_path, latest_exp = save_best_model(args.output_dir)

    convert_to_onnx(best_weights_path)

    onnx_file = latest_exp / "weights/best.onnx"
    
    logger.info(f"ONNX model saved at {onnx_file}")
    
    '''
    trt_file = latest_exp / "best_model.trt"
    convert_to_trt(onnx_file, trt_file)
    '''


if __name__ == "__main__":
    main()







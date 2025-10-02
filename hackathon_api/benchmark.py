# benchmark.py
import time
import argparse
import numpy as np
import cv2
from ultralytics import YOLO

def benchmark(model_path: str, image_path: str, iterations: int = 50, warmup: int = 5):
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("Warming up...")
    for _ in range(warmup):
        _ = model(img)

    print(f"Running {iterations} iterations...")
    latencies = []
    start_total = time.time()
    for i in range(iterations):
        start = time.time()
        _ = model(img)
        latencies.append((time.time() - start) * 1000)
    total_time = time.time() - start_total

    avg_latency = np.mean(latencies)
    fps = iterations / total_time

    try:
        flops = model.info(verbose=False)[-1]
    except Exception:
        flops = "N/A"

    print("\n=== Benchmark Results ===")
    print(f"Device: {model.device}")
    print(f"Iterations: {iterations}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput (FPS): {fps:.2f}")
    print(f"FLOPs: {flops}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="src/ml/weights/best.pt")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    benchmark(args.model, args.image, args.iterations)

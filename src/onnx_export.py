import argparse
import os
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PyTorchToONNX")

def export_pytorch_model(model_path: str, output_path: str, dummy_input_shape=(1, 3, 224, 224)):
    """
    Export a raw PyTorch model to ONNX.
    """
    logger.info(f"Loading PyTorch model from {model_path}")
    model = torch.load(model_path, map_location='cpu')

    if hasattr(model, 'module'):
        model = model.module

    model.eval()

    dummy_input = torch.randn(*dummy_input_shape)
    logger.info(f"Using dummy input shape: {dummy_input_shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    logger.info(f"PyTorch model exported to ONNX: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch → ONNX Exporter")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pt or .pth PyTorch model.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the exported ONNX model.")
    parser.add_argument("--dummy_shape", type=str, default="1,3,224,224",
                        help="Dummy input shape (comma-separated), e.g., 1,3,224,224")

    args = parser.parse_args()
    shape = tuple(map(int, args.dummy_shape.strip().split(",")))

    export_pytorch_model(args.model_path, args.output_path, dummy_input_shape=shape)
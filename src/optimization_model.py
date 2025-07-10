from setuptools import setup, Command
import subprocess

class ExportCommand(Command):
    description = "Export PyTorch model to ONNX using universal_model_exporter.py"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        command = [
            "python", "onnx_export.py",
            "--framework", "pytorch",
            "--model_name", "outputs/checkpoints/model_epoch_500.pt",
            "--output_path", "india_speak_tts_model.onnx",
            "--dummy_shape", "1,3,224,224"
        ]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)

setup(
    name="model_exporter",
    version="0.1",
    description="Universal ONNX model exporter",
    py_modules=["universal_model_exporter"],
    cmdclass={
        "export": ExportCommand,
    },
)
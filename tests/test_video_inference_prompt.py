import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


DEFAULT_PROMPT = (
    "A blue laboratory workbench with a black computer monitor, a black mouse, "
    "and a green digital image processing textbook, with lab stools and benches "
    "in the background."
)


def load_video_inference():
    module_path = Path(__file__).resolve().parents[1] / "video_inference.py"
    spec = importlib.util.spec_from_file_location("video_inference", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VideoInferencePromptTest(unittest.TestCase):
    def setUp(self):
        fake_modules = {
            "safetensors": types.SimpleNamespace(torch=types.SimpleNamespace()),
            "torch": types.SimpleNamespace(bfloat16=object()),
            "diffusers": types.SimpleNamespace(),
            "diffusers.utils": types.SimpleNamespace(
                export_to_video=lambda *args, **kwargs: None,
                load_image=lambda path: path,
            ),
            "cogvideox_interpolation": types.SimpleNamespace(),
            "cogvideox_interpolation.pipeline": types.SimpleNamespace(
                CogVideoXInterpolationPipeline=object
            ),
        }
        self.module_patcher = patch.dict(sys.modules, fake_modules)
        self.module_patcher.start()
        self.addCleanup(self.module_patcher.stop)
        self.module = load_video_inference()

    def test_parse_args_uses_scene_specific_default_prompt(self):
        with patch.object(sys, "argv", ["video_inference.py"]):
            args = self.module.parse_args()

        self.assertEqual(args.prompt, DEFAULT_PROMPT)

    def test_parse_args_allows_prompt_override(self):
        custom_prompt = "custom prompt"
        with patch.object(sys, "argv", ["video_inference.py", "--prompt", custom_prompt]):
            args = self.module.parse_args()

        self.assertEqual(args.prompt, custom_prompt)


if __name__ == "__main__":
    unittest.main()

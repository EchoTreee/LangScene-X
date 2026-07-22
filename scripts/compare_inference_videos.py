from pathlib import Path

import cv2
import numpy as np


def mean_abs_diff(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def main():
    root = Path("/workspace/projects/LangScene-X/runs/inference_results")
    for name in ["第一次inference", "第二次inference", "第三次inference", "第四次inference"]:
        run = root / name
        video = run / "video/rgb/video_ckpt_800.mp4"
        input1 = run / "rgb/0001.png"
        input2 = run / "rgb/0002.png"
        if not (video.exists() and input1.exists() and input2.exists()):
            print(name, "missing required files")
            continue

        cap = cv2.VideoCapture(str(video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        img1 = cv2.resize(cv2.imread(str(input1)), (width, height), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(cv2.imread(str(input2)), (width, height), interpolation=cv2.INTER_AREA)

        sampled = []
        for idx in [0, 12, 24, 36, 48]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                sampled.append((idx, frame))
        cap.release()

        print(name, {"frames": frame_count, "width": width, "height": height})
        print(
            "  mae_to_inputs",
            [
                (idx, round(mean_abs_diff(frame, img1), 2), round(mean_abs_diff(frame, img2), 2))
                for idx, frame in sampled
            ],
        )
        print(
            "  sampled_motion",
            [
                (f"{idx_a}->{idx_b}", round(mean_abs_diff(frame_a, frame_b), 2))
                for (idx_a, frame_a), (idx_b, frame_b) in zip(sampled, sampled[1:])
            ],
        )


if __name__ == "__main__":
    main()

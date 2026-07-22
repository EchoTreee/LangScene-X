from pathlib import Path

import cv2
import numpy as np


def main():
    root = Path("/workspace/projects/LangScene-X")
    run = root / "runs/inference_results/第四次inference"
    out = run / "analysis_frames"
    out.mkdir(parents=True, exist_ok=True)

    video = str(run / "video/rgb/video_ckpt_800.mp4")
    img1 = cv2.imread(str(run / "rgb/0001.png"))
    img2 = cv2.imread(str(run / "rgb/0002.png"))

    cap = cv2.VideoCapture(video)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    indices = [0, 12, 24, 36, 48]

    img1r = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
    img2r = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "input_0001_resized.jpg"), img1r)
    cv2.imwrite(str(out / "input_0002_resized.jpg"), img2r)

    sample_frames = []
    print("analysis_dir", out)
    print("video", {"frames": frames, "width": width, "height": height, "fps": fps})
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print("frame", idx, "read_failed")
            continue
        cv2.imwrite(str(out / f"frame_{idx:02d}.jpg"), frame)
        sample_frames.append((idx, frame))
        mae1 = float(np.mean(np.abs(frame.astype(np.float32) - img1r.astype(np.float32))))
        mae2 = float(np.mean(np.abs(frame.astype(np.float32) - img2r.astype(np.float32))))
        print("frame", idx, {"mae_to_input1": round(mae1, 3), "mae_to_input2": round(mae2, 3)})
    cap.release()

    for (idx_a, frame_a), (idx_b, frame_b) in zip(sample_frames, sample_frames[1:]):
        motion = float(np.mean(np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32))))
        print("sample_motion", f"{idx_a}->{idx_b}", round(motion, 3))

    thumbs = [img1r] + [frame for _, frame in sample_frames] + [img2r]
    labels = ["input1"] + [f"f{idx}" for idx, _ in sample_frames] + ["input2"]
    rendered = []
    for image, label in zip(thumbs, labels):
        thumb = cv2.resize(image, (240, 160), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (239, 24), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            label,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rendered.append(thumb)
    cv2.imwrite(str(out / "contact_sheet.jpg"), np.concatenate(rendered, axis=1))


if __name__ == "__main__":
    main()

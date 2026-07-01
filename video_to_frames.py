"""Sample N frames spread across a generated camera-orbit video, ready to feed
into build_room_multiview.py (VGGT fusion).

Use evenly-spaced frames across the WHOLE clip, not consecutive ones: VGGT needs
baseline between views, and the camera moves most across the full trajectory.

    python video_to_frames.py path/to/clip.mp4 --out out/wanframes --n 8
    python build_room_multiview.py --images out/wanframes --name wan_room
"""
import argparse
import os


def extract(video, out_dir, n):
    os.makedirs(out_dir, exist_ok=True)
    try:
        import cv2
        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            raise RuntimeError("cv2 reported 0 frames")
        idxs = [round(i * (total - 1) / max(1, n - 1)) for i in range(n)]
        saved = 0
        for j, fi in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imwrite(os.path.join(out_dir, f"frame_{j:02d}.png"), frame)
            saved += 1
        cap.release()
        return saved, total
    except Exception as e:
        print(f"  cv2 failed ({e}); trying imageio...")
        import imageio.v3 as iio
        frames = iio.imread(video, plugin="pyav")  # (T,H,W,3)
        total = len(frames)
        idxs = [round(i * (total - 1) / max(1, n - 1)) for i in range(n)]
        from PIL import Image
        for j, fi in enumerate(idxs):
            Image.fromarray(frames[fi]).save(os.path.join(out_dir, f"frame_{j:02d}.png"))
        return len(idxs), total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="generated .mp4 (ComfyUI/output)")
    ap.add_argument("--out", default="out/wanframes")
    ap.add_argument("--n", type=int, default=8, help="frames to sample across clip")
    args = ap.parse_args()
    saved, total = extract(args.video, args.out, args.n)
    print(f"sampled {saved}/{args.n} frames (of {total}) -> {args.out}/")
    print(f"Next: python build_room_multiview.py --images {args.out} --name wan_room")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.gt_loader import GTLoader
from evaluation.mot_eval import FrameEvaluator
from visualization.vis_tracks import TrackVisualizer
from tracker.tracker import Tracker
from tracker.track import TrackState


def parse_args():
    parser = argparse.ArgumentParser(description="MOT evaluation pipeline")
    parser.add_argument("--data_dir", required=True, help="Data folder containing images and filtered/")
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--out_dir", default="eval_output/")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None,
               help="게이팅 네트워크 체크포인트 경로 (없으면 휴리스틱 초기값 사용)")
    return parser.parse_args()


def tracks_to_boxes(tracks):
    if not tracks:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.stack([t.to_tlbr() for t in tracks]).cpu()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Override device in cfg
    from CFG.cfg import cfg
    cfg.device = args.device

    loader = GTLoader(args.data_dir)
    tracker = Tracker()
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device,
                        weights_only=True)
        tracker.gating.load_state_dict(ckpt["model"])
        tracker.gating.eval()
        print(f"[Info] 게이팅 체크포인트 로드: {args.checkpoint}")
    else:
        print("[Info] 체크포인트 없음 — 휴리스틱 초기값 사용")
    evaluator = FrameEvaluator(iou_thresh=args.iou_thresh)

    visualizer = None
    if args.save_video:
        video_path = os.path.join(args.out_dir, "tracks.mp4")
        visualizer = TrackVisualizer(fps=30, out_path=video_path)

    total_frames = len(loader)
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)

    results = []

    for idx in range(total_frames):
        img_path, gt_boxes = loader[idx]

        det_input = gt_boxes.to(args.device) if gt_boxes.shape[0] > 0 else gt_boxes

        tracked = tracker.update(det_input)
        lost = tracker.lost_tracks

        pred_boxes = tracks_to_boxes(tracked)
        gt_coords = gt_boxes[:, :4] if gt_boxes.shape[0] > 0 else torch.zeros((0, 4))

        frame_result = evaluator.match_frame(gt_coords, pred_boxes)
        results.append(frame_result)

        if args.save_video and visualizer is not None:
            visualizer.add_frame(img_path, gt_boxes, tracked, lost)

        if (idx + 1) % 100 == 0:
            partial = evaluator.compute_metrics(results)
            print(f"[Frame {idx + 1}/{total_frames}] "
                  f"P={partial['precision']:.3f} R={partial['recall']:.3f} F1={partial['f1']:.3f}")

    metrics = evaluator.compute_metrics(results)
    evaluator.print_report(metrics)

    metrics_path = os.path.join(args.out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Metrics saved to {metrics_path}")

    if args.save_video and visualizer is not None:
        visualizer.release()
        print(f"Video saved to {os.path.join(args.out_dir, 'tracks.mp4')}")


if __name__ == "__main__":
    main()

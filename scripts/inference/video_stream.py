import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from saving_utils import save_video
from stream_handler import FrameStatus, InputStreamHandler



from sam3.model_builder import build_sam3_stream_predictor
from sam3.visualization_utils import render_masklet_frame

sys.path.insert(0, "/home/opervu/ws/make_hdr")
from hdr import merge_hdr  # noqa: E402
from frame_capture import capture_hdr_sets
import time
from arena_api.system import system # noqa: E402

YARP_IMAGE_PORT = "/sam3/rgbImage:i"
DEFAULT_TEXT_PROMPT = ["sponge", "forceps"]


def is_inside_region(cx, cy, region):
    """Check if normalized point (cx, cy) is inside region (x1, y1, x2, y2) all normalized."""
    x1, y1, x2, y2 = region
    return x1 <= cx <= x2 and y1 <= cy <= y2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model with a specified checkpoint directory on a specified video."
    )

    parser.add_argument(
        "--run_output_name",
        type=str,
        default=None,
        help="Name of the run's output directory and video file (without extension). "
        "If not specified, uses datetime.",
    )
    parser.add_argument(
        "--custom_det_chkpt",
        type= str,
        default="chkpt/checkpoint.pt",
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default="outputs",
        help="Root output directory",
    )

    # === Input source options ===
    parser.add_argument(
    "--text_prompt",
    type=str,
    nargs="+",
    default=DEFAULT_TEXT_PROMPT,
    help=f"Text prompts for segmentation (default: {DEFAULT_TEXT_PROMPT})",
)
    parser.add_argument(
        "--stream_type",
        type=str,
        choices=["yarp", "video", "webcam", "hdr"],
        default="yarp",
        help="Input source kind: yarp | video | webcam | hdr (Arena HDR camera)",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to input video when --source=video",
    )
    parser.add_argument(
        "--webcam_index",
        type=int,
        default=0,
        help="Webcam index for --source=webcam",
    )
    parser.add_argument(
        "--yarp_port",
        type=str,
        default=YARP_IMAGE_PORT,
        help="YARP port name for --source=yarp",
    )

    # === Counting options ===
    parser.add_argument(
        "--counting_region",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Normalized rectangle region (x1 y1 x2 y2) for in/out counting, e.g. 0.1 0.1 0.9 0.9. "
             "Objects crossing into this region count as 'in', leaving count as 'out'.",
    )

    # === Visualization options ===
    viz_results_group = parser.add_mutually_exclusive_group()
    viz_results_group.add_argument(
        "--viz_results",
        dest="viz_results",
        action="store_true",
        help="Visualize results on the fly.",
    )
    viz_results_group.add_argument(
        "--no_viz_results",
        dest="viz_results",
        action="store_false",
        help="Do not visualize results on the fly.",
    )
    parser.set_defaults(viz_results=False)

    # === Saving options ===
    save_images_group = parser.add_mutually_exclusive_group()
    save_images_group.add_argument(
        "--save_images",
        dest="save_images",
        action="store_true",
        help="Save generated images.",
    )
    save_images_group.add_argument(
        "--no_save_images",
        dest="save_images",
        action="store_false",
        help="Do not save generated images.",
    )
    parser.set_defaults(save_images=True)

    save_video_group = parser.add_mutually_exclusive_group()
    save_video_group.add_argument(
        "--save_video",
        dest="save_video",
        action="store_true",
        help="Save output video.",
    )
    save_video_group.add_argument(
        "--no_save_video",
        dest="save_video",
        action="store_false",
        help="Do not save output video.",
    )
    parser.set_defaults(save_video=True)

    # ===============================

    args = parser.parse_args()

    # Log args
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(f"Starting realtime inference using text prompt: '{args.text_prompt}'")

    # Counting state
    counting_region = tuple(args.counting_region) if args.counting_region else None
    prev_obj_inside = {}   # obj_id -> bool (inside region or not)
    counts = {}            # category -> current live count inside region
    if counting_region:
        print(f"Counting region (normalized): {counting_region}")

    # Use datetime as video ID if run_output_name not specified
    if args.run_output_name is not None:
        video_id = args.run_output_name
    else:
        video_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If we need to save results, create output dir
    output_dir = os.path.join(
        args.output_root_dir,
        video_id,
    )
    if args.save_images or args.save_video:
        os.makedirs(output_dir, exist_ok=True)

    images_dir = None
    if args.save_images:
        images_dir = os.path.join(output_dir, "frames")
        os.makedirs(images_dir, exist_ok=True)

    # Initialize predictor (single-GPU streaming)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam3_stream_predictor(device=device)
    # predictor.model.recondition_every_nth_frame= 5
    print("  Official model loaded.\n")
    print("=" * 60)
    print("Loading custom checkpoint ...")
    custom_ckpt = torch.load(args.custom_det_chkpt, map_location="cpu", weights_only=True)

    # The checkpoint may wrap the state_dict under a "model" key
    if "model" in custom_ckpt and isinstance(custom_ckpt["model"], dict):
        custom_ckpt = custom_ckpt["model"]

    print(f"  Custom checkpoint has {len(custom_ckpt)} keys")
    demo_model = predictor.model

    detector_state = demo_model.detector.state_dict()
    detector_keys = set(detector_state.keys())

    remapped = {}
    for k, v in custom_ckpt.items():
        if k in detector_keys:
            remapped[k] = v

    print(f"  Matched {len(remapped)} / {len(detector_keys)} detector keys")

    # Warn about keys that did NOT match
    unmatched_custom = len(custom_ckpt) - len(remapped)
    if unmatched_custom:
        print(f"  ⚠ {unmatched_custom} custom checkpoint keys could not be matched "
            f"(likely tracker-only or naming mismatch)")

    # Load the custom detector weights
    if remapped:
        missing, unexpected = demo_model.detector.load_state_dict(
            remapped, strict=False
        )
        if missing:
            print(f"  ⚠ Detector missing keys: {len(missing)} "
                f"(will keep official weights for these)")
        if unexpected:
            print(f"  ⚠ Detector unexpected keys: {len(unexpected)}")
        print("  Custom detector weights applied.\n")
    else:
        print("  ❌ No detector keys matched! Using official detector weights.\n")

    demo_model.cuda().eval()
    print("Model ready (detector=custom, tracker=official).\n")

    resp = predictor.handle_request({"type": "start_session"})
    session_id = resp["session_id"]

    # Initialize input source
    if args.stream_type == "hdr":
        # Arena HDR: connect to the LUCID camera and build an endless
        # generator of exposure-bracketed cycles. Each cycle holds
        # `num_exposures` frames, longest exposure first.
        tries = 0
        tries_max = 6
        sleep_time_secs = 10
        devices = None
        while tries < tries_max:
            devices = system.create_device()
            if not devices:
                print(f'Try {tries + 1} of {tries_max}: waiting '
                      f'{sleep_time_secs} secs for a device to be connected!')
                time.sleep(sleep_time_secs)
                tries += 1
            else:
                break
        else:
            raise Exception('No device found! Please connect a device and run '
                            'the example again.')

        device = system.select_device(devices)
        print(f'Device used: {device}')

        # 4 exposures halving down from 100 ms: 100000, 50000, 25000, 12500 us
        num_exposures = 4
        exposure_time_max = 100000.0
        hdr_sets = capture_hdr_sets(device, num_exposures=num_exposures,
                                    exposure_time_max=exposure_time_max,
                                    num_cycles=-1, frame_rate=1.0)
        # Longest first: [100000, 50000, 25000, 12500]
        hdr_exposure_times = np.array(
            [exposure_time_max / (2 ** i) for i in range(num_exposures)],
            dtype=np.float32,
        )
        src = None
    else:
        src = InputStreamHandler(
            kind=args.stream_type,
            video_path=args.video_path,
            webcam_index=args.webcam_index,
            yarp_port_name=args.yarp_port,
        )
        print(f"Opening source: {args.stream_type}")
        src.open()

    peak_memory = 0
    frame_idx = 0

    frame_timestamps = []  # To compute output fps
    video_frames = []  # Buffer of frames for final video save

    stop_processing = False
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            while stop_processing is not True:
                if args.stream_type == "hdr":
                    # Next exposure cycle: merge the shortest (dark) and
                    # longest (bright) exposure into one HDR frame (RGB).
                    try:
                        cycle = next(hdr_sets)
                    except StopIteration:
                        print("\nHDR capture ended.")
                        break
                    sorted_frames = sorted(cycle, key=lambda img: img.mean())
                    dark_frame = sorted_frames[0]     # shortest exposure
                    bright_frame = sorted_frames[-1]  # longest exposure
                    # Exposure times in the same order as [dark, bright].
                    exposure_times = hdr_exposure_times[[-1, 0]]
                    frame_rgb = merge_hdr([dark_frame, bright_frame],
                                          exposure_times)
                else:
                    # Read frame (RGB)
                    stream_buffer = src.read()
                    if stream_buffer.status == FrameStatus.NO_FRAME:
                        # YARP: no new frame yet; try again.
                        continue
                    if stream_buffer.status == FrameStatus.EOS:
                        # End of stream for video/webcam or closed YARP port.
                        break
                    frame_rgb = stream_buffer.frame

                # Push frame
                predictor.handle_request(
                    {"type": "add_frame", "session_id": session_id, "frame": frame_rgb}
                )

                # Add text prompt only on first frame
                if frame_idx == 0:
                    predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "text": args.text_prompt,
                        }
                    )

                # # You can potentially add more prompts on later frames too
                # if frame_idx == 30:
                #     predictor.handle_request(
                #         {
                #             "type": "add_prompt",
                #             "session_id": session_id,
                #             "frame_index": 30,
                #             "text": "bottle",
                #         }
                #     )

                # Run per-frame inference
                resp = predictor.handle_request(
                    {
                        "type": "run_inference",
                        "session_id": session_id,
                        "frame_index": frame_idx,
                    }
                )
                outputs = resp.get("outputs")
                if outputs is not None:
                    overlay_rgb = render_masklet_frame(
                        frame_rgb, outputs, frame_idx=frame_idx, alpha=0.5
                    )

                    # --- In/Out counting ---
                    if counting_region and len(outputs.get("out_obj_ids", [])) > 0:
                        boxes = outputs["out_boxes_xywh"]   # (cx, cy, w, h) normalized
                        categories = outputs.get("out_obj_categories", ["unknown"] * len(boxes))
                        for i, obj_id in enumerate(outputs["out_obj_ids"]):
                            cx, cy = float(boxes[i][0]), float(boxes[i][1])
                            inside = is_inside_region(cx, cy, counting_region)
                            cat = categories[i] if i < len(categories) else "unknown"
                            prev = prev_obj_inside.get(int(obj_id))
                            if prev is False and inside:
                                counts[cat] = counts.get(cat, 0) + 1
                            elif prev is True and not inside:
                                counts[cat] = counts.get(cat, 0) - 1
                            prev_obj_inside[int(obj_id)] = inside

                        # Print live counts
                        count_str = " | ".join(
                            f"{c}: {counts.get(c, 0)}" for c in sorted(counts.keys())
                        )
                        print(f"  [Inside] {count_str}", end="")
                else:
                    overlay_rgb = frame_rgb

                # Draw counting region
                if counting_region:
                    h, w = frame_rgb.shape[:2]
                    x1, y1, x2, y2 = counting_region
                    px1, py1 = int(x1 * w), int(y1 * h)
                    px2, py2 = int(x2 * w), int(y2 * h)
                    # cv2.rectangle expects BGR
                    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(overlay_bgr, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

                # Save image per frame with 4-digit index
                if args.save_images:
                    img_path = os.path.join(images_dir, f"{frame_idx:04d}.png")
                    Image.fromarray(overlay_rgb).save(img_path)

                # Collect video frames for final save
                if args.save_video:
                    # Keep numpy RGB arrays to pass directly to save_video
                    video_frames.append(overlay_rgb)

                if args.viz_results:
                    cv2.imshow(
                        "SAM3 Livestream", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_processing = True

                # Update running statistics
                frame_idx += 1
                frame_timestamps.append(time.time())
                current_peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                peak_memory = max(peak_memory, current_peak_memory)
                print(
                    f"Processed frame {frame_idx}. "
                    f"Current peak memory: {current_peak_memory:.2f} GB, "
                    f"Overall peak memory: {peak_memory:.2f} GB.",
                    end="\r",
                )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping processing gracefully...")

    finally:
        # Source cleanup
        if args.stream_type == "hdr":
            # Closing the generator restores the camera's sequencer/exposure
            # settings; the device must still be destroyed to avoid
            # corrupting the camera's saved preset.
            try:
                hdr_sets.close()
            except Exception:
                pass
            system.destroy_device()
        else:
            src.close()

        if args.save_video:
            if len(frame_timestamps) >= 2:
                elapsed = frame_timestamps[-1] - frame_timestamps[0]
                # Use average FPS over the whole run
                effective_fps = (len(frame_timestamps) - 1) / elapsed
            else:
                effective_fps = 30.0

            save_video(
                frames=video_frames,
                output_name=video_id,
                output_dir=output_dir,
                fps=effective_fps,
                overlay_text=f"Text prompt: {args.text_prompt}",
            )
            print(
                f"\nSaved video to {os.path.join(output_dir, video_id + '.mp4')} at {effective_fps:.2f} FPS."
            )

        # Close any OpenCV windows
        if args.viz_results:
            cv2.destroyAllWindows()

        # Print final counting summary
        if counting_region:
            print(f"\n{'='*45}")
            print(f"Live Count Summary (region: {counting_region}):")
            for cat in sorted(counts.keys()):
                print(f"  {cat}: inside={counts.get(cat, 0)}")
            print(f"{'='*45}")

        print(f"Processed {frame_idx} frames.")
        print(f"Peak GPU memory usage: {peak_memory:.2f} GB.")

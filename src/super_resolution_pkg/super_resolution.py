import os
import subprocess
import tempfile
import argparse
import platform
import queue
import threading
import cv2
import time
import torch
import numpy as np

# Allow MPS to fall back to CPU for any unsupported ops, rather than crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Compatibility shim: basicsr imports torchvision.transforms.functional_tensor,
# which was removed in torchvision >= 0.16. Patch it before importing basicsr.
import importlib
try:
    importlib.import_module('torchvision.transforms.functional_tensor')
except ModuleNotFoundError:
    import types
    import torchvision.transforms.functional as F
    fake_module = types.ModuleType('torchvision.transforms.functional_tensor')
    fake_module.rgb_to_grayscale = F.rgb_to_grayscale
    import sys
    sys.modules['torchvision.transforms.functional_tensor'] = fake_module

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

RESOLUTION_MAP = {
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
    "5k": (5120, 2880),
}

def get_input_resolution(input_path):
    """Gets the resolution of an image or video file."""
    input_ext = os.path.splitext(input_path)[1].lower()
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']

    if input_ext in image_extensions:
        img = cv2.imread(input_path)
        if img is not None:
            return img.shape[1], img.shape[0]
    elif input_ext in video_extensions:
        try:
            command = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                input_path,
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            width, height = map(int, result.stdout.strip().split('x'))
            return width, height
        except Exception as e:
            print(f"Error getting video resolution: {e}")
    return None

def get_video_framerate(video_path):
    """Gets the frame rate of a video using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        num, den = map(int, result.stdout.split('/'))
        return num / den
    except Exception as e:
        print(f"Error getting frame rate: {e}")
        raise


def has_audio_stream(video_path):
    """Checks if a video file has an audio stream."""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return result.stdout.strip() != ""
    except subprocess.CalledProcessError as e:
        print(f"Error checking for audio stream: {e.stderr}")
        return False


def get_video_frame_count(video_path):
    """Gets the total number of frames in a video using container metadata."""
    command = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        count = result.stdout.strip()
        if count and count != "N/A":
            return int(count)
    except Exception:
        pass
    return None


def get_system_memory_gb():
    """Returns total physical memory in GB, or None if detection fails."""
    try:
        # macOS: sysctl hw.memsize returns total bytes
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    try:
        # Linux fallback
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024 ** 3)
    except Exception:
        return None


def get_device_config():
    """Detects the best available hardware and returns (gpu_id, use_half, device) configuration.

    Returns:
        tuple: (gpu_id, use_half, device) where gpu_id is 0 for CUDA, None for MPS/CPU,
               use_half indicates whether FP16 precision can be used, and device is a
               torch.device for the selected backend.
    """
    if torch.cuda.is_available():
        print("Detected NVIDIA GPU with CUDA support.")
        return 0, True, torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        print("Detected Apple Silicon GPU (MPS).")
        return None, False, torch.device('mps')
    else:
        print("No GPU detected. Using CPU (this will be slow).")
        return None, False, torch.device('cpu')


def run_mps_diagnostics(upsampler):
    """Run thorough checks to verify MPS is actually being used for inference.

    Prints a diagnostic report covering:
      1. System / PyTorch environment
      2. MPS backend availability
      3. Model parameter placement (every parameter on the expected device)
      4. Basic tensor matmul benchmark (MPS vs CPU)
      5. Real model inference benchmark (MPS vs CPU) on a 256x256 test image
    Returns True if all checks pass, False otherwise.
    """
    SEPARATOR = "-" * 60
    all_passed = True

    print("\n" + "=" * 60)
    print("  MPS DEVICE DIAGNOSTICS")
    print("=" * 60)

    # ── 1. System info ───────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("1. System Information")
    print(SEPARATOR)
    mem_gb = get_system_memory_gb()
    print(f"  Platform     : {platform.platform()}")
    print(f"  Processor    : {platform.processor() or 'N/A'}")
    print(f"  Memory       : {f'{mem_gb:.0f} GB' if mem_gb else 'unknown'}")
    print(f"  Python       : {platform.python_version()}")
    print(f"  PyTorch      : {torch.__version__}")

    # ── 2. MPS backend availability ──────────────────────────────
    print(f"\n{SEPARATOR}")
    print("2. MPS Backend Availability")
    print(SEPARATOR)
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    print(f"  MPS built into PyTorch : {mps_built}")
    print(f"  MPS device available   : {mps_available}")
    if mps_available:
        # Basic allocation test
        try:
            t = torch.zeros(1, device="mps")
            del t
            print(f"  MPS tensor allocation  : OK")
        except Exception as e:
            print(f"  MPS tensor allocation  : FAILED ({e})")
            all_passed = False
    else:
        print("  ** MPS is NOT available. Model will run on CPU. **")
        all_passed = False

    result_tag = lambda ok: "PASS" if ok else "FAIL"

    # ── 3. Model parameter device check ──────────────────────────
    print(f"\n{SEPARATOR}")
    print("3. Model Parameter Device Check")
    print(SEPARATOR)
    expected_device_type = "mps" if mps_available else "cpu"
    total_params = 0
    misplaced_params = 0
    for name, param in upsampler.model.named_parameters():
        total_params += 1
        if param.device.type != expected_device_type:
            misplaced_params += 1
            if misplaced_params <= 5:  # only print first few
                print(f"  MISPLACED: {name} -> {param.device}")
    if misplaced_params > 5:
        print(f"  ... and {misplaced_params - 5} more misplaced parameters")
    params_ok = misplaced_params == 0
    print(f"  Total parameters : {total_params}")
    print(f"  On '{expected_device_type}'      : {total_params - misplaced_params}")
    print(f"  Misplaced        : {misplaced_params}")
    print(f"  Result           : [{result_tag(params_ok)}]")
    if not params_ok:
        all_passed = False

    # ── 4. Tensor matmul benchmark (MPS vs CPU) ─────────────────
    # This is informational only. Apple's AMX coprocessor makes CPU matmul
    # extremely fast for small matrices, so MPS dispatch overhead can dominate
    # at small sizes. The real inference benchmark (check 5) is the definitive
    # pass/fail test.
    print(f"\n{SEPARATOR}")
    print("4. Tensor Matmul Benchmark (4096x4096, 10 iterations) [informational]")
    print(SEPARATOR)
    size = 4096
    iters = 10

    # CPU timing
    a_cpu = torch.randn(size, size, device="cpu")
    b_cpu = torch.randn(size, size, device="cpu")
    torch.matmul(a_cpu, b_cpu)  # warmup
    cpu_start = time.time()
    for _ in range(iters):
        torch.matmul(a_cpu, b_cpu)
    cpu_elapsed = time.time() - cpu_start
    print(f"  CPU total     : {cpu_elapsed:.3f}s ({cpu_elapsed/iters*1000:.1f}ms/iter)")

    if mps_available:
        a_mps = torch.randn(size, size, device="mps")
        b_mps = torch.randn(size, size, device="mps")
        torch.matmul(a_mps, b_mps)
        torch.mps.synchronize()  # warmup
        mps_start = time.time()
        for _ in range(iters):
            torch.matmul(a_mps, b_mps)
        torch.mps.synchronize()
        mps_elapsed = time.time() - mps_start
        speedup = cpu_elapsed / mps_elapsed if mps_elapsed > 0 else 0
        print(f"  MPS total     : {mps_elapsed:.3f}s ({mps_elapsed/iters*1000:.1f}ms/iter)")
        print(f"  Speedup       : {speedup:.1f}x")
        if speedup <= 1.0:
            print(f"  Note          : MPS slower at this size; Apple AMX is very fast for BLAS ops.")
        del a_mps, b_mps
    else:
        print("  MPS           : skipped (not available)")
    del a_cpu, b_cpu

    # ── 5. Real model inference benchmark ────────────────────────
    print(f"\n{SEPARATOR}")
    print("5. Real-ESRGAN Inference Benchmark (256x256 test image)")
    print(SEPARATOR)
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    if mps_available:
        # Time on MPS (model is already there)
        upsampler.enhance(test_img)  # warmup
        torch.mps.synchronize()
        mps_start = time.time()
        upsampler.enhance(test_img)
        torch.mps.synchronize()
        mps_infer = time.time() - mps_start
        print(f"  MPS inference : {mps_infer:.3f}s")

        # Move model to CPU, time, then move back
        upsampler.model.cpu()
        upsampler.device = torch.device("cpu")
        upsampler.enhance(test_img)  # warmup
        cpu_start = time.time()
        upsampler.enhance(test_img)
        cpu_infer = time.time() - cpu_start
        print(f"  CPU inference : {cpu_infer:.3f}s")

        speedup = cpu_infer / mps_infer if mps_infer > 0 else 0
        infer_ok = speedup > 1.0
        print(f"  Speedup       : {speedup:.1f}x")
        print(f"  Result        : [{result_tag(infer_ok)}]"
              + ("" if infer_ok else " (MPS not faster — ops may be falling back to CPU)"))
        if not infer_ok:
            all_passed = False

        # Restore model to MPS
        upsampler.model.to(torch.device("mps"))
        upsampler.device = torch.device("mps")
        torch.mps.synchronize()
    else:
        cpu_start = time.time()
        upsampler.enhance(test_img)
        cpu_infer = time.time() - cpu_start
        print(f"  CPU inference : {cpu_infer:.3f}s")
        print(f"  MPS           : skipped (not available)")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if all_passed:
        print("  DIAGNOSTICS PASSED: MPS is active and accelerating inference.")
    else:
        print("  DIAGNOSTICS WARNING: One or more checks failed.")
        print("  The model may be running on CPU or MPS is underperforming.")
    print("=" * 60 + "\n")

    return all_passed


def initialize_upsampler(scale):
    """Initializes the Real-ESRGAN upsampler."""
    base_model_dir = os.path.join(os.path.dirname(__file__), "models")
    if scale == 4:
        model_path = os.path.join(base_model_dir, "RealESRGAN_x4plus.pth")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    elif scale == 2:
        model_path = os.path.join(base_model_dir, "RealESRGAN_x2plus.pth")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    else:
        print("Error: Invalid scale factor. Please choose 2 or 4.")
        return None

    gpu_id, use_half, device = get_device_config()

    # Pick tile size based on device and available memory.
    # tile=0 means no tiling (full-frame inference, fastest but most memory).
    if gpu_id is not None:
        # CUDA: dedicated VRAM, no tiling needed
        tile = 0
    elif device.type == "mps":
        # Apple Silicon: GPU shares unified memory with CPU.
        # Larger tiles = fewer GPU dispatch round-trips = much faster.
        # A 720p frame with tile=512 produces 6 tiles; tile=0 does it in one pass.
        mem_gb = get_system_memory_gb()
        if mem_gb is not None:
            print(f"System memory: {mem_gb:.0f} GB")
            if mem_gb >= 16:
                tile = 0     # no tiling — 720p/1080p fits comfortably
            elif mem_gb >= 8:
                tile = 768   # 2 tiles for 720p instead of 6
            else:
                tile = 512
        else:
            tile = 512  # safe fallback if detection fails
    else:
        # CPU: conservative tiling
        tile = 512

    print(f"Tile size: {tile if tile > 0 else 'disabled (full-frame)'}")
    print(f"Initializing Real-ESRGAN model for {scale}x upscaling...")
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        gpu_id=gpu_id,
        device=device
    )

    # Verify the model actually landed on the expected device.
    # Compare by device type (e.g. "mps") rather than string representation
    # because PyTorch may report "mps:0" even when we requested "mps".
    actual_device = next(upsampler.model.parameters()).device
    print(f"Model loaded on device: {actual_device}")
    if actual_device.type != device.type:
        print(f"WARNING: Expected device type '{device.type}', but model is on '{actual_device}'.")

    # Warm up the MPS pipeline. The first inference triggers Metal shader
    # compilation which is very slow; doing it on a tiny image avoids counting
    # that cost against real frames.  We also use the warmup result as an FP32
    # reference to test whether FP16 produces acceptable quality.
    if device.type == "mps":
        print("Warming up MPS pipeline (compiling Metal shaders)...")
        warmup_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        fp32_out, _ = upsampler.enhance(warmup_img)
        torch.mps.synchronize()
        print("MPS pipeline ready.")

        # FP16 gives ~2x speedup on Apple Silicon (double FP16 throughput +
        # halved memory bandwidth).  Test quality against FP32 reference.
        print("Testing FP16 precision for faster inference...")
        try:
            upsampler.model.half()
            upsampler.half = True
            fp16_out, _ = upsampler.enhance(warmup_img)
            torch.mps.synchronize()
            diff = np.abs(fp32_out.astype(np.float32) - fp16_out.astype(np.float32))
            if diff.max() <= 30:
                print(f"  FP16 enabled (max pixel diff: {diff.max():.0f}/255, "
                      f"mean: {diff.mean():.1f})")
            else:
                print(f"  FP16 quality insufficient (max diff: {diff.max():.0f}), "
                      f"staying on FP32")
                upsampler.model.float()
                upsampler.half = False
        except Exception as e:
            print(f"  FP16 not available ({e}), staying on FP32")
            upsampler.model.float()
            upsampler.half = False

    return upsampler


def upscale_image(input_path, output_path, upsampler, target_resolution):
    """Upscales a single image and resizes to target resolution."""
    start_time = time.time()
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Cannot read image at {input_path}")
        return

    try:
        upscaled_img, _ = upsampler.enhance(img)
        final_img = cv2.resize(upscaled_img, target_resolution, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(output_path, final_img)
        end_time = time.time()
        print(f"Successfully upscaled image to {output_path}")
        print(f"Image processing took {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error upscaling image {input_path}: {e}")


def extract_first_frame(input_path, width, height):
    """Extracts the first frame from a video as a BGR numpy array."""
    cmd = [
        "ffmpeg", "-i", input_path,
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-v", "error", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    expected = width * height * 3
    if len(result.stdout) < expected:
        return None
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3).copy()


def _format_eta(secs_per_frame, total_frames):
    """Format an ETA string from per-frame time and total frame count."""
    if total_frames is None:
        return "unknown"
    total_s = int(secs_per_frame * total_frames)
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def upscale_video(input_path, output_path, upsampler, target_resolution):
    """Upscales a video using Real-ESRGAN with pipelined I/O.

    Flow:
      1. Process a test frame so the user can inspect quality and timing.
      2. Offer a "fast" mode that pre-resizes input when the model overshoots
         the target (e.g. 4x on 720p → 2880p, but target is only 2160p).
      3. User confirms before the full pipeline runs.
      4. Three-stage threaded pipeline with ffmpeg pipes (zero intermediate files):
           reader thread → read_q → main thread (GPU) → write_q → writer thread
    """
    start_time = time.time()
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    input_res = get_input_resolution(input_path)
    if input_res is None:
        print("Error: Could not determine input video resolution.")
        return
    in_w, in_h = input_res
    target_w, target_h = target_resolution
    framerate = get_video_framerate(input_path)
    total_frames = get_video_frame_count(input_path)
    has_audio = has_audio_stream(input_path)

    print(f"Video: {in_w}x{in_h} -> {target_w}x{target_h}, "
          f"{total_frames or '?'} frames at {framerate:.2f} fps")

    # ── Determine available modes ────────────────────────────────────
    scale = upsampler.scale
    optimal_in_w = target_w // scale
    optimal_in_h = target_h // scale
    fast_available = (optimal_in_w < in_w) and (optimal_in_h < in_h)

    # ── Test frame ───────────────────────────────────────────────────
    print("\nProcessing test frame...")
    test_frame = extract_first_frame(input_path, in_w, in_h)
    use_fast = False  # default; may be overridden by user choice below
    if test_frame is None:
        print("Warning: Could not extract test frame, skipping preview.")
    else:
        preview_dir = os.path.dirname(output_path) or "."

        # Quality mode
        t0 = time.time()
        quality_out, _ = upsampler.enhance(test_frame)
        quality_out = cv2.resize(quality_out, target_resolution,
                                 interpolation=cv2.INTER_LANCZOS4)
        quality_time = time.time() - t0
        quality_preview = os.path.join(preview_dir, "test_frame_quality.png")
        cv2.imwrite(quality_preview, quality_out)
        del quality_out

        # Fast mode (if applicable)
        fast_time = None
        fast_preview = None
        if fast_available:
            pre_resized = cv2.resize(test_frame, (optimal_in_w, optimal_in_h),
                                     interpolation=cv2.INTER_LANCZOS4)
            t0 = time.time()
            fast_out, _ = upsampler.enhance(pre_resized)
            fast_out = cv2.resize(fast_out, target_resolution,
                                  interpolation=cv2.INTER_LANCZOS4)
            fast_time = time.time() - t0
            fast_preview = os.path.join(preview_dir, "test_frame_fast.png")
            cv2.imwrite(fast_preview, fast_out)
            del fast_out, pre_resized

        del test_frame

        # ── Show results ─────────────────────────────────────────────
        print(f"\n{'=' * 55}")
        print("  TEST FRAME RESULTS")
        print(f"{'=' * 55}")
        print(f"  [1] Quality : {quality_time:.1f}s/frame "
              f"(ETA {_format_eta(quality_time, total_frames)})")
        print(f"      Preview : {quality_preview}")
        if fast_available and fast_time is not None:
            speedup = quality_time / fast_time if fast_time > 0 else 0
            print(f"  [2] Fast    : {fast_time:.1f}s/frame "
                  f"(ETA {_format_eta(fast_time, total_frames)}, "
                  f"{speedup:.1f}x faster)")
            print(f"      Preview : {fast_preview}")
            print(f"      (input pre-resized {in_w}x{in_h} -> "
                  f"{optimal_in_w}x{optimal_in_h})")
        print(f"{'=' * 55}")

        # ── User choice ──────────────────────────────────────────────
        try:
            if fast_available:
                choice = input("\nChoose mode [1/2] or 'q' to quit: ").strip().lower()
                if choice == 'q':
                    print("Cancelled.")
                    return
                use_fast = (choice == '2')
            else:
                choice = input("\nProceed? [y/N]: ").strip().lower()
                if choice not in ('y', 'yes'):
                    print("Cancelled.")
                    return
                use_fast = False
        except EOFError:
            pass  # non-interactive: keep default (quality)

    # ── Full pipeline ────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.aac")
        if has_audio:
            print("Extracting audio...")
            try:
                subprocess.run(
                    ["ffmpeg", "-i", input_path, "-vn", "-acodec", "copy",
                     audio_path],
                    check=True, capture_output=True, text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error extracting audio: {e.stderr}")
                return

        # Decode pipe: ffmpeg -> raw BGR24 frames -> stdout
        decode_cmd = [
            "ffmpeg", "-i", input_path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-v", "error", "pipe:1",
        ]
        decode_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)

        # Encode pipe: stdin raw BGR24 frames -> ffmpeg -> output file
        encode_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{target_w}x{target_h}",
            "-r", str(framerate),
            "-i", "pipe:0",
        ]
        if has_audio:
            encode_cmd.extend(["-i", audio_path, "-c:a", "copy"])
        encode_cmd.extend([
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-v", "error",
            output_path,
        ])
        encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

        # Threaded pipeline
        read_q = queue.Queue(maxsize=4)
        write_q = queue.Queue(maxsize=4)
        error_event = threading.Event()
        frame_byte_size = in_w * in_h * 3

        def reader():
            """Reads decoded frames; optionally pre-resizes for fast mode."""
            try:
                while not error_event.is_set():
                    raw = decode_proc.stdout.read(frame_byte_size)
                    if len(raw) < frame_byte_size:
                        read_q.put(None)  # EOF
                        break
                    frame = np.frombuffer(
                        raw, dtype=np.uint8
                    ).reshape(in_h, in_w, 3).copy()
                    if use_fast:
                        frame = cv2.resize(
                            frame, (optimal_in_w, optimal_in_h),
                            interpolation=cv2.INTER_LANCZOS4)
                    read_q.put(frame)
            except Exception:
                error_event.set()
                read_q.put(None)

        def writer():
            """Resizes if needed and writes frames to the encode pipe."""
            try:
                while not error_event.is_set():
                    item = write_q.get()
                    if item is None:
                        break
                    h_out, w_out = item.shape[:2]
                    if w_out != target_w or h_out != target_h:
                        item = cv2.resize(
                            item, target_resolution,
                            interpolation=cv2.INTER_LANCZOS4)
                    encode_proc.stdin.write(item.tobytes())
            except Exception:
                error_event.set()

        reader_thread = threading.Thread(target=reader, daemon=True)
        writer_thread = threading.Thread(target=writer, daemon=True)
        reader_thread.start()
        writer_thread.start()

        mode_label = "fast" if use_fast else "quality"
        print(f"\nStarting pipelined upscaling ({mode_label} mode)...")
        frame_times = []
        i = 0
        try:
            while True:
                img = read_q.get()
                if img is None or error_event.is_set():
                    write_q.put(None)
                    break

                frame_start = time.time()
                try:
                    upscaled, _ = upsampler.enhance(img)
                except Exception as e:
                    print(f"\nError upscaling frame {i + 1}: {e}")
                    error_event.set()
                    write_q.put(None)
                    break
                write_q.put(upscaled)
                frame_elapsed = time.time() - frame_start
                frame_times.append(frame_elapsed)

                if i == 0 and total_frames:
                    print(f"\nFirst frame: {frame_elapsed:.2f}s. "
                          f"Est. total: "
                          f"{_format_eta(frame_elapsed, total_frames)} "
                          f"for {total_frames} frames. Ctrl+C to cancel.\n")

                frames_done = i + 1
                if total_frames:
                    avg_time = sum(frame_times) / len(frame_times)
                    remaining = avg_time * (total_frames - frames_done)
                    eta_min, eta_sec = divmod(int(remaining), 60)
                    eta_hr, eta_min = divmod(eta_min, 60)
                    pct = 100.0 * frames_done / total_frames
                    print(
                        f"Frame {frames_done}/{total_frames} [{pct:5.1f}%] "
                        f"({frame_elapsed:.2f}s/frame, "
                        f"ETA: {eta_hr}h{eta_min:02d}m{eta_sec:02d}s)  ",
                        end="\r")
                else:
                    print(f"Frame {frames_done} "
                          f"({frame_elapsed:.2f}s/frame)  ", end="\r")
                i += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            error_event.set()
            write_q.put(None)

        # Cleanup
        writer_thread.join(timeout=10)
        try:
            encode_proc.stdin.close()
        except Exception:
            pass
        encode_proc.wait()
        decode_proc.terminate()
        decode_proc.wait()

        if error_event.is_set():
            print("Processing stopped due to errors.")
            return

    end_time = time.time()
    total_processed = len(frame_times)
    avg = (end_time - start_time) / total_processed if total_processed else 0
    print(f"\nSuccessfully created upscaled video at {output_path}")
    print(f"Processed {total_processed} frames in "
          f"{end_time - start_time:.2f}s ({avg:.2f}s/frame avg).")

def process_file(input_file, output_file, args, upsampler):
    input_resolution = get_input_resolution(input_file)
    if input_resolution is None:
        print(f"Skipping {input_file}: Could not determine input resolution.")
        return

    if args.target_resolution:
        if args.target_resolution in RESOLUTION_MAP:
            target_resolution = RESOLUTION_MAP[args.target_resolution]
        else:
            print("Error: Invalid target resolution.")
            return
    else:
        scale_factor = args.scale
        target_resolution = (int(input_resolution[0] * scale_factor), int(input_resolution[1] * scale_factor))

    video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    
    input_ext = os.path.splitext(input_file)[1].lower()

    if input_ext in video_extensions:
        upscale_video(input_file, output_file, upsampler, target_resolution)
    elif input_ext in image_extensions:
        upscale_image(input_file, output_file, upsampler, target_resolution)
    else:
        print(f"Skipping {input_file}: Unsupported file format.")

def main():
    parser = argparse.ArgumentParser(description="Upscale a video or image using Real-ESRGAN.")
    parser.add_argument("input_path", help="Path to the input video or image file or directory.")
    parser.add_argument("output_path", help="Path to save the output file or directory.")
    parser.add_argument("--scale", type=float, help="Upscaling factor (e.g., 1.5, 2.0). Mutually exclusive with --target-resolution.")
    parser.add_argument("--target-resolution", type=str, help="Target resolution (e.g., 1080p, 4k). Mutually exclusive with --scale.")
    
    args = parser.parse_args()

    if args.scale and args.target_resolution:
        print("Error: Please provide either --scale or --target-resolution, not both.")
        exit()

    if not args.scale and not args.target_resolution:
        print("Error: Please provide either --scale or --target-resolution.")
        exit()

    if args.scale:
        if args.scale <= 2:
            model_scale = 2
        elif args.scale <= 4:
            model_scale = 4
        else:
            print("Error: Scaling factors greater than 4 are not yet supported.")
            exit()
    else: # args.target_resolution
        if os.path.isdir(args.input_path):
            first_file = next((f for f in os.listdir(args.input_path) if os.path.isfile(os.path.join(args.input_path, f))), None)
            if not first_file:
                print("Error: Input directory is empty.")
                exit()
            input_resolution = get_input_resolution(os.path.join(args.input_path, first_file))
        else:
            input_resolution = get_input_resolution(args.input_path)
        
        if input_resolution is None:
            print("Error: Could not determine input resolution for the first file.")
            exit()
            
        if args.target_resolution in RESOLUTION_MAP:
            target_resolution = RESOLUTION_MAP[args.target_resolution]
            scale_factor = target_resolution[0] / input_resolution[0]
            if scale_factor <= 2:
                model_scale = 2
            elif scale_factor <= 4:
                model_scale = 4
            else:
                print("Error: Scaling factors greater than 4 are not yet supported.")
                exit()
        else:
            print("Error: Invalid target resolution.")
            exit()


    upsampler = initialize_upsampler(model_scale)
    if upsampler is None:
        exit()

    run_mps_diagnostics(upsampler)

    if os.path.isdir(args.input_path):
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        
        for filename in os.listdir(args.input_path):
            input_file = os.path.join(args.input_path, filename)
            output_file = os.path.join(args.output_path, filename)

            if os.path.isfile(input_file):
                process_file(input_file, output_file, args, upsampler)
    else:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        process_file(args.input_path, args.output_path, args, upsampler)


if __name__ == "__main__":
    main()
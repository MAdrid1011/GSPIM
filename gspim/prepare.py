"""Download and prepare registered inputs without starting training."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

from integrations.manifest import CACHE, DATA_SHA256, DATA_URL, FRAMES, ROOT, UPSTREAMS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, digest: str) -> None:
    if path.is_symlink():
        raise ValueError(f"input must be a regular file, not a symbolic link: {path}")
    if sha256(path) != digest:
        raise ValueError(f"SHA256 mismatch: {path}; remove the file and prepare it again")


def download(url: str, destination: Path, digest: str | None = None) -> None:
    """Publish a complete download atomically, retaining no partial final file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "GSPIM"})
    try:
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if digest is not None:
            verify_file(temporary, digest)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_upstreams(root: Path = ROOT) -> None:
    """Create pinned source checkouts used by model construction."""
    for upstream in UPSTREAMS:
        destination = root / "third_party" / upstream.name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", upstream.url, str(destination)], check=True)
            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", upstream.commit], check=True)
        actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        if actual != upstream.commit:
            raise ValueError(f"{destination}: expected revision {upstream.commit}, found {actual}")



def prepare_scene(scene: str, root: Path = ROOT, cache: Path = CACHE) -> None:
    """Verify original videos/poses, then preserve both image conventions."""
    directory = root / "datasets/n3v" / scene
    directory.mkdir(parents=True, exist_ok=True)
    expected = DATA_SHA256[scene]
    missing = [name for name in expected if not (directory / name).is_file()]
    if missing:
        archive = cache / "downloads" / f"n3v-{scene}.zip"
        if not archive.is_file():
            download(DATA_URL.format(scene=scene), archive)
        with zipfile.ZipFile(archive) as source:
            bad = source.testzip()
            if bad is not None:
                raise ValueError(f"N3V archive CRC mismatch: {bad}")
            for name in missing:
                matches = [entry for entry in source.infolist() if Path(entry.filename).name == name]
                if len(matches) != 1:
                    raise ValueError(f"N3V archive needs exactly one {name}")
                destination = directory / name
                temporary = destination.with_suffix(destination.suffix + ".part")
                try:
                    with source.open(matches[0]) as stream, temporary.open("wb") as target:
                        shutil.copyfileobj(stream, target)
                    verify_file(temporary, expected[name])
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
        archive.unlink()
    for name, digest in expected.items():
        verify_file(directory / name, digest)
    prepare_images(directory)


def prepare_images(directory: Path) -> None:
    """Decode actual camera IDs; use bilinear Ex4DGS and native bicubic pixels."""
    import numpy as np
    from PIL import Image

    poses = np.load(directory / "poses_bounds.npy")
    videos = sorted(directory.glob("cam[0-9][0-9].mp4"))
    if len(videos) != len(poses) or not videos or videos[0].stem != "cam00":
        raise ValueError(f"{directory}: sorted video IDs must match pose rows and begin with cam00")
    original = poses[0, :15].reshape(3, 5)
    height, width = (int(value) for value in original[:2, -1])
    image_root = directory / "images"
    ex_root = directory / "cam00"
    image_root.mkdir(exist_ok=True)
    ex_root.mkdir(exist_ok=True)
    native_byte = (np.arange(256) / 255.0 * 255.0).astype(np.byte).view(np.uint8)
    def prepare_video(video):
        outputs = [image_root / f"{video.stem}_{frame:04d}.png" for frame in range(FRAMES)]
        ex_outputs = [ex_root / f"{frame:04d}.png" for frame in range(FRAMES)] if video.stem == "cam00" else []
        if all(path.is_file() for path in outputs + ex_outputs):
            return
        process = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(video),
            "-frames:v", str(FRAMES), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], stdout=subprocess.PIPE)
        try:
            for frame, output in enumerate(outputs):
                size = width * height * 3
                raw = bytearray()
                while len(raw) < size:
                    part = process.stdout.read(size - len(raw))
                    if not part:
                        raise ValueError(f"short decoded video: {video}, frame {frame}")
                    raw.extend(part)
                pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                pending = []
                if not output.is_file():
                    pending.append((output, Image.fromarray(native_byte[pixels]).resize((width // 2, height // 2))))
                if ex_outputs and not ex_outputs[frame].is_file():
                    pending.append((ex_outputs[frame], Image.fromarray(pixels).resize(
                        (width // 2, height // 2), Image.Resampling.BILINEAR)))
                for path, image in pending:
                    temporary = path.with_suffix(".png.part")
                    try:
                        image.save(temporary, format="PNG", compress_level=3)
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
            if process.wait() != 0:
                raise RuntimeError(f"FFmpeg failed: {video}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
        print(f"Prepared {directory.name}/{video.stem}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as workers:
        tuple(workers.map(prepare_video, videos))


def input_problems(pairs, root: Path = ROOT, frames: int = FRAMES,
                   checkpoint: Path | None = None, images: Path | None = None) -> list[str]:
    """List every missing selected input before CUDA is initialized."""
    problems = []
    for workload, scene in pairs:
        model = checkpoint or workload.checkpoint(scene, root)
        image_root = images or workload.image_directory(scene, root)
        if not model.is_file():
            method = ("run python -m gspim.demo --prepare" if workload.model_url else
                      f"place the supplied {workload.directory} model at this path; see docs/workloads.md")
            problems.append(f"{model}: {method}")
        if workload.name != "ex4dgs" and not (image_root.parent / "poses_bounds.npy").is_file():
            problems.append(f"{image_root.parent / 'poses_bounds.npy'}: run python -m gspim.demo --prepare")
        missing = [image_root / f"{workload.image_prefix}{frame:04d}.png" for frame in range(frames)
                   if not (image_root / f"{workload.image_prefix}{frame:04d}.png").is_file()]
        if missing:
            problems.append(f"{image_root}: {len(missing)} missing images, starting with {missing[0].name}; "
                            "run python -m gspim.demo --prepare")
    return list(dict.fromkeys(problems))


def prepare(pairs, root: Path = ROOT, cache: Path = CACHE) -> list[str]:
    """Prepare public inputs and verify supplied models; return missing items."""
    pairs = tuple(pairs)
    for directory in ("datasets/n3v", "checkpoints", "third_party", "outputs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for scene in dict.fromkeys(scene for _, scene in pairs):
        prepare_scene(scene, root, cache)
    for workload, scene in pairs:
        checkpoint = workload.checkpoint(scene, root)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.is_file() and workload.model_url is not None:
            download(workload.model_url.format(scene=scene), checkpoint, workload.sha256(scene))
        if checkpoint.is_file():
            verify_file(checkpoint, workload.sha256(scene))
    prepare_upstreams(root)
    return input_problems(pairs, root)

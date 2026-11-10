"""Prepare registered datasets and verify the supplied checkpoints."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

from integrations.manifest import (
    CACHE, DATA_SHA256, DATA_URL, FRAMES, GOLDEN_RATIO, OFFICIAL_SCENE_SOURCES, ROOT, SCENE_FRAMES, UPSTREAMS,
)


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
    """Initialize pinned simulation and plotting sources."""
    for upstream in UPSTREAMS:
        destination = root / "third_party" / upstream.name
        if upstream == GOLDEN_RATIO and (root / ".gitmodules").is_file():
            subprocess.run(["git", "-C", str(root), "submodule", "update", "--init", "--",
                            str(destination.relative_to(root))], check=True)
        elif not (destination / ".git").exists():
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
    if scene in OFFICIAL_SCENE_SOURCES:
        prepare_official_scene(scene, directory, cache)
        return
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


def _calibration(path: Path):
    """Read the named OpenCV matrices in the official camera package."""
    import numpy as np

    text = path.read_text(encoding="utf-8")
    names = re.findall(r'^\s*-\s*"([^"]+)"', text, re.MULTILINE)
    matrices = {}
    for name, rows, cols, body in re.findall(
            r"(?m)^([A-Za-z]+_[0-9]+):\s*!!opencv-matrix\s*\n\s*rows:\s*(\d+)\s*\n"
            r"\s*cols:\s*(\d+)\s*\n\s*dt:\s*[a-z]+\s*\n\s*data:\s*\[([^]]+)\]", text):
        matrices[name] = np.asarray([float(item) for item in body.split(",")],
                                   dtype=np.float64).reshape(int(rows), int(cols))
    if not names or len(set(names)) != len(names):
        raise ValueError(f"invalid camera names: {path}")
    return names, matrices


def _read_frame(stream, size: int, video: Path, frame: int) -> bytes:
    pixels = bytearray()
    while len(pixels) < size:
        part = stream.read(size - len(pixels))
        if not part:
            raise ValueError(f"short decoded video: {video}, frame {frame}")
        pixels.extend(part)
    return bytes(pixels)


def prepare_official_scene(scene: str, directory: Path, cache: Path) -> None:
    """Undistort official pinhole cameras and write both input layouts."""
    import numpy as np

    calibration = directory / "optimized"
    video_root = directory / "videos_libx265"
    names = []
    if all((calibration / name).is_file() for name in ("intri.yml", "extri.yml")):
        names, _ = _calibration(calibration / "intri.yml")
    videos = {path.stem.split("_", 1)[0]: path for path in video_root.glob("*.mp4")}
    if not names or set(videos) != set(names):
        archive = cache / "downloads" / f"{scene}_libx265.tar.gz"
        if not archive.is_file():
            raise FileNotFoundError(
                f"{scene}: place the official scene archive at {archive}; see docs/data.md")
        verify_file(archive, OFFICIAL_SCENE_SOURCES[scene]["scene_archive_sha256"])
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                path = Path(member.name)
                if not member.isfile() or not (
                        member.name in ("optimized/intri.yml", "optimized/extri.yml") or
                        (len(path.parts) == 2 and path.parts[0] == "videos_libx265" and
                         path.suffix == ".mp4")):
                    continue
                target = directory / path
                if target.is_file():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                try:
                    with source.extractfile(member) as raw, temporary.open("wb") as output:
                        shutil.copyfileobj(raw, output)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        names, _ = _calibration(calibration / "intri.yml")
        videos = {path.stem.split("_", 1)[0]: path for path in video_root.glob("*.mp4")}
    names, intrinsic = _calibration(calibration / "intri.yml")
    extrinsic_names, extrinsic = _calibration(calibration / "extri.yml")
    if names != extrinsic_names or set(videos) != set(names):
        raise ValueError(f"{scene}: video IDs and calibration cameras differ")
    names = sorted(names)
    frames = SCENE_FRAMES[scene]
    image_root, ex_root = directory / "images", directory / "cam00"
    pose_file = directory / "poses_bounds.npy"
    images_complete = all(
        (image_root / f"cam{name}_{frame:04d}.png").is_file()
        for name in names for frame in range(frames)
    )
    aliases_complete = all(
        (image_root / f"cam00_{frame:04d}.png").is_file() and
        (ex_root / f"{frame:04d}.png").is_file()
        for frame in range(frames)
    )
    if pose_file.is_file() and images_complete and aliases_complete:
        return

    import cv2

    cv2.setNumThreads(1)
    corrected = {
        name: cv2.getOptimalNewCameraMatrix(
            intrinsic[f"K_{name}"], intrinsic[f"D_{name}"],
            (1920, 1080), 0, (1920, 1080),
        )[0]
        for name in names
    }
    rows = []
    for name in names:
        rotation = extrinsic[f"Rot_{name}"].T
        center = -rotation @ extrinsic[f"T_{name}"].reshape(3)
        # LLFF stores down, right, back, camera center, then image metadata.
        pose = np.column_stack((rotation[:, 1], rotation[:, 0], -rotation[:, 2], center,
                                [1080., 1920., corrected[name][0, 0]]))
        bound = extrinsic.get(f"bounds_{name}")
        far = max(20., float(np.max(np.abs(bound))) * 2) if bound is not None and bound.size == 6 else 20.
        rows.append(np.concatenate((pose.reshape(-1), [.1, far])))
    poses = np.asarray(rows, dtype=np.float64)
    if pose_file.is_file():
        if not np.array_equal(np.load(pose_file), poses):
            raise ValueError(f"camera poses differ from official calibration: {pose_file}")
    else:
        temporary = pose_file.with_suffix(".npy.part")
        try:
            with temporary.open("wb") as output:
                np.save(output, poses)
            temporary.replace(pose_file)
        finally:
            temporary.unlink(missing_ok=True)
    image_root.mkdir(exist_ok=True)
    ex_root.mkdir(exist_ok=True)

    def prepare_camera(name):
        outputs = [image_root / f"cam{name}_{frame:04d}.png" for frame in range(frames)]
        if not all(path.is_file() for path in outputs):
            video = videos[name]
            process = subprocess.Popen([
                "ffmpeg", "-v", "error", "-threads", "2", "-i", str(video),
                "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "bgr24",
                "pipe:1"], stdout=subprocess.PIPE)
            maps = cv2.initUndistortRectifyMap(
                intrinsic[f"K_{name}"], intrinsic[f"D_{name}"], None,
                corrected[name], (1920, 1080), cv2.CV_16SC2,
            )
            try:
                for frame, output in enumerate(outputs):
                    pixels = _read_frame(process.stdout, 1920 * 1080 * 3, video, frame)
                    if not output.is_file():
                        image = np.frombuffer(pixels, np.uint8).reshape(1080, 1920, 3)
                        image = cv2.resize(
                            cv2.remap(image, *maps, cv2.INTER_LINEAR),
                            (960, 540), interpolation=cv2.INTER_AREA,
                        )
                        success, encoded = cv2.imencode(
                            ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1],
                        )
                        if not success:
                            raise ValueError(f"PNG encoding failed: {output}")
                        temporary = output.with_suffix(".png.part")
                        try:
                            temporary.write_bytes(encoded.tobytes())
                            temporary.replace(output)
                        finally:
                            temporary.unlink(missing_ok=True)
                if process.wait() != 0:
                    raise RuntimeError(f"FFmpeg failed: {video}")
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        if name == names[0]:
            for frame, output in enumerate(outputs):
                for alias in (image_root / f"cam00_{frame:04d}.png", ex_root / f"{frame:04d}.png"):
                    if not alias.is_file():
                        temporary = alias.with_suffix(".png.part")
                        try:
                            shutil.copy2(output, temporary)
                            temporary.replace(alias)
                        finally:
                            temporary.unlink(missing_ok=True)
        print(f"Prepared {scene}/cam{name}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as workers:
        tuple(workers.map(prepare_camera, names))


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
                raw = _read_frame(process.stdout, size, video, frame)
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
            problems.append(
                f"{model}: extract the GSPIM checkpoint bundle at checkpoints/; "
                "see docs/workloads.md")
        if workload.name != "ex4dgs" and not (image_root.parent / "poses_bounds.npy").is_file():
            problems.append(f"{image_root.parent / 'poses_bounds.npy'}: run python -m gspim.demo --prepare")
        required_frames = min(frames, SCENE_FRAMES.get(scene, frames))
        missing = [image_root / f"{workload.image_prefix}{frame:04d}.png" for frame in range(required_frames)
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
        if checkpoint.is_file():
            verify_file(checkpoint, workload.sha256(scene))
    prepare_upstreams(root)
    return input_problems(pairs, root)

from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_EXPECTED_RELATIVE_PATH = Path("calendars/day.txt")

Downloader = Callable[[Path], None]


@dataclass(frozen=True)
class CachePreparationResult:
    source: str
    local_dir: Path
    drive_archive_path: Path


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def copy_tree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    reset_dir(dst)
    shutil.copytree(src, dst)


def validate_qlib_dir(
    data_dir: Path,
    expected_relative_path: Path = DEFAULT_EXPECTED_RELATIVE_PATH,
) -> Path:
    expected_path = data_dir / expected_relative_path
    if not expected_path.exists():
        raise FileNotFoundError(
            f"Qlib data directory is missing {expected_relative_path}: {data_dir}"
        )
    return expected_path


def pack_directory_to_archive(
    source_dir: Path,
    archive_path: Path,
    expected_relative_path: Path = DEFAULT_EXPECTED_RELATIVE_PATH,
) -> Path:
    validate_qlib_dir(source_dir, expected_relative_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive_path = archive_path.parent / f".{archive_path.name}.tmp"
    if tmp_archive_path.exists():
        tmp_archive_path.unlink()

    try:
        with tarfile.open(tmp_archive_path, mode="w:gz") as archive:
            archive.add(source_dir, arcname=source_dir.name)
        tmp_archive_path.replace(archive_path)
    finally:
        if tmp_archive_path.exists():
            tmp_archive_path.unlink()

    return archive_path


def extract_archive_to_dir(
    archive_path: Path,
    target_dir: Path,
    expected_relative_path: Path = DEFAULT_EXPECTED_RELATIVE_PATH,
) -> Path:
    if not archive_path.exists():
        raise FileNotFoundError(f"Qlib archive not found: {archive_path}")

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    reset_dir(target_dir)
    temp_root = target_dir.parent / f".{target_dir.name}.extracting"
    reset_dir(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            _safe_extract(archive, temp_root)
        extracted_dir = _resolve_extracted_data_dir(temp_root, target_dir.name, expected_relative_path)
        extracted_dir.replace(target_dir)
        validate_qlib_dir(target_dir, expected_relative_path)
        return target_dir
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


def ensure_drive_qlib_data(
    local_dir: Path,
    drive_archive_path: Path,
    downloader: Downloader,
    legacy_drive_dir: Path | None = None,
    expected_relative_path: Path = DEFAULT_EXPECTED_RELATIVE_PATH,
) -> CachePreparationResult:
    if drive_archive_path.exists():
        extract_archive_to_dir(drive_archive_path, local_dir, expected_relative_path)
        return CachePreparationResult(
            source="archive",
            local_dir=local_dir,
            drive_archive_path=drive_archive_path,
        )

    if legacy_drive_dir is not None:
        legacy_expected_path = legacy_drive_dir / expected_relative_path
        if legacy_expected_path.exists():
            copy_tree(legacy_drive_dir, local_dir)
            validate_qlib_dir(local_dir, expected_relative_path)
            pack_directory_to_archive(local_dir, drive_archive_path, expected_relative_path)
            return CachePreparationResult(
                source="legacy-directory",
                local_dir=local_dir,
                drive_archive_path=drive_archive_path,
            )

    reset_dir(local_dir)
    downloader(local_dir)
    validate_qlib_dir(local_dir, expected_relative_path)
    pack_directory_to_archive(local_dir, drive_archive_path, expected_relative_path)
    return CachePreparationResult(
        source="download",
        local_dir=local_dir,
        drive_archive_path=drive_archive_path,
    )


def _safe_extract(archive: tarfile.TarFile, target_dir: Path) -> None:
    target_dir_resolved = target_dir.resolve()
    for member in archive.getmembers():
        member_path = (target_dir / member.name).resolve()
        if not str(member_path).startswith(str(target_dir_resolved)):
            raise ValueError(f"Archive contains an unsafe path: {member.name}")
    archive.extractall(target_dir)


def _resolve_extracted_data_dir(
    extract_root: Path,
    target_dir_name: str,
    expected_relative_path: Path,
) -> Path:
    candidates: list[Path] = [extract_root / target_dir_name]
    candidates.extend(path for path in extract_root.iterdir() if path.is_dir())
    candidates.append(extract_root)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        if (candidate / expected_relative_path).exists():
            return candidate

    raise FileNotFoundError(
        f"Extracted archive does not contain {expected_relative_path} under {extract_root}"
    )

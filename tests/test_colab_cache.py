from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from alphagen_qlib.cache import ensure_drive_qlib_data


def _write_sample_qlib_dir(root: Path, marker: str) -> None:
    (root / "calendars").mkdir(parents=True, exist_ok=True)
    (root / "features").mkdir(parents=True, exist_ok=True)
    (root / "calendars" / "day.txt").write_text(
        "2024-01-02\n2024-01-03\n",
        encoding="utf-8",
    )
    (root / "features" / "marker.txt").write_text(marker, encoding="utf-8")


def _pack_sample_archive(source_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w:gz") as archive:
        archive.add(source_dir, arcname=source_dir.name)


def test_ensure_drive_qlib_data_prefers_archive_cache(tmp_path: Path) -> None:
    archive_source = tmp_path / "archive-source"
    _write_sample_qlib_dir(archive_source, marker="from-archive")

    local_dir = tmp_path / "local" / "us_data"
    drive_archive = tmp_path / "drive" / "archives" / "us_data.tar.gz"
    legacy_drive_dir = tmp_path / "drive" / "qlib_data" / "us_data"
    _pack_sample_archive(archive_source, drive_archive)
    _write_sample_qlib_dir(legacy_drive_dir, marker="from-legacy")

    result = ensure_drive_qlib_data(
        local_dir=local_dir,
        drive_archive_path=drive_archive,
        legacy_drive_dir=legacy_drive_dir,
        downloader=lambda _: pytest.fail("downloader should not be called when archive exists"),
    )

    assert result.source == "archive"
    assert (local_dir / "features" / "marker.txt").read_text(encoding="utf-8") == "from-archive"


def test_ensure_drive_qlib_data_falls_back_to_legacy_dir_and_writes_archive(tmp_path: Path) -> None:
    local_dir = tmp_path / "local" / "us_data"
    drive_archive = tmp_path / "drive" / "archives" / "us_data.tar.gz"
    legacy_drive_dir = tmp_path / "drive" / "qlib_data" / "us_data"
    _write_sample_qlib_dir(legacy_drive_dir, marker="from-legacy")

    result = ensure_drive_qlib_data(
        local_dir=local_dir,
        drive_archive_path=drive_archive,
        legacy_drive_dir=legacy_drive_dir,
        downloader=lambda _: pytest.fail("downloader should not be called when legacy cache exists"),
    )

    assert result.source == "legacy-directory"
    assert (local_dir / "features" / "marker.txt").read_text(encoding="utf-8") == "from-legacy"
    assert drive_archive.exists()

    extracted_dir = tmp_path / "verify"
    with tarfile.open(drive_archive, mode="r:gz") as archive:
        archive.extractall(extracted_dir)
    assert (
        extracted_dir / local_dir.name / "features" / "marker.txt"
    ).read_text(encoding="utf-8") == "from-legacy"


def test_ensure_drive_qlib_data_downloads_and_packs_archive_when_cache_missing(tmp_path: Path) -> None:
    local_dir = tmp_path / "local" / "us_data"
    drive_archive = tmp_path / "drive" / "archives" / "us_data.tar.gz"
    legacy_drive_dir = tmp_path / "drive" / "qlib_data" / "us_data"
    downloader_calls: list[Path] = []

    def downloader(target_dir: Path) -> None:
        downloader_calls.append(target_dir)
        _write_sample_qlib_dir(target_dir, marker="from-download")

    result = ensure_drive_qlib_data(
        local_dir=local_dir,
        drive_archive_path=drive_archive,
        legacy_drive_dir=legacy_drive_dir,
        downloader=downloader,
    )

    assert result.source == "download"
    assert downloader_calls == [local_dir]
    assert (local_dir / "features" / "marker.txt").read_text(encoding="utf-8") == "from-download"
    assert drive_archive.exists()
    assert not legacy_drive_dir.exists()

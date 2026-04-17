from __future__ import annotations

import sys
from types import ModuleType

import pytest

import alphagen_qlib.stock_data as stock_data


def test_normalize_qlib_region_supports_common_aliases() -> None:
    assert stock_data.normalize_qlib_region("cn") == "cn"
    assert stock_data.normalize_qlib_region("China") == "cn"
    assert stock_data.normalize_qlib_region("REG_US") == "us"
    assert stock_data.normalize_qlib_region("usa") == "us"


def test_normalize_qlib_region_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported Qlib region"):
        stock_data.normalize_qlib_region("eu")


def test_initialize_qlib_is_idempotent_for_same_provider_and_region(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    fake_qlib = ModuleType("qlib")
    fake_config = ModuleType("qlib.config")
    fake_config.REG_CN = "REG_CN"
    fake_config.REG_US = "REG_US"

    def fake_init(*, provider_uri: str, region: str) -> None:
        calls.append((provider_uri, region))

    fake_qlib.init = fake_init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qlib", fake_qlib)
    monkeypatch.setitem(sys.modules, "qlib.config", fake_config)
    monkeypatch.setattr(stock_data, "_QLIB_INITIALIZED", False)
    monkeypatch.setattr(stock_data, "_QLIB_PROVIDER_URI", None)
    monkeypatch.setattr(stock_data, "_QLIB_REGION", None)

    stock_data.initialize_qlib("/tmp/us_data", region="us")
    stock_data.initialize_qlib("/tmp/us_data", region="us")

    assert calls == [("/tmp/us_data", "REG_US")]


def test_initialize_qlib_rejects_switching_region_without_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_qlib = ModuleType("qlib")
    fake_config = ModuleType("qlib.config")
    fake_config.REG_CN = "REG_CN"
    fake_config.REG_US = "REG_US"
    fake_qlib.init = lambda **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qlib", fake_qlib)
    monkeypatch.setitem(sys.modules, "qlib.config", fake_config)
    monkeypatch.setattr(stock_data, "_QLIB_INITIALIZED", False)
    monkeypatch.setattr(stock_data, "_QLIB_PROVIDER_URI", None)
    monkeypatch.setattr(stock_data, "_QLIB_REGION", None)

    stock_data.initialize_qlib("/tmp/cn_data", region="cn")

    with pytest.raises(RuntimeError, match="already initialized"):
        stock_data.initialize_qlib("/tmp/us_data", region="us")

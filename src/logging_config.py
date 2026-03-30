"""リポジトリ共通のログ設定。"""

from __future__ import annotations

import logging
import os

DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(log_level: str | None = None) -> int:
    """環境変数または引数に基づいて root logger のレベルを設定する。"""

    resolved_level = (log_level or os.getenv("LOG_LEVEL") or DEFAULT_LOG_LEVEL).upper()
    logging.basicConfig(level=_resolve_log_level(resolved_level), format=LOG_FORMAT)
    logging.getLogger().setLevel(_resolve_log_level(resolved_level))
    return logging.getLogger().level


def _resolve_log_level(log_level: str) -> int:
    """文字列のログレベルを logging モジュールの整数値へ変換する。"""

    return getattr(logging, log_level, logging.INFO)

"""R6-T: local-only bridge between a normal publisher mock and R6-S SQLite.

No Telegram/provider code. No configuration/env activation path. The real publisher
contains a hard-OFF constant and its ON branch unconditionally stops before send.
Fixture methods are usable ONLY on disposable SQLite test history, not on prod.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.services import safe03_r6s_joint_reservation_offline as s

PRODUCTION_ENABLED = False


class R6TProductionBlock(RuntimeError):
    pass


def stop_live_normal_route() -> None:
    """Cannot return a send-capable token; never try delivery in R6-T."""
    raise R6TProductionBlock('r6t_unwired_delivery_hook_must_stop_before_send')


@dataclass(frozen=True)
class LocalResult:
    status: str
    reason: str
    reservation_key: str = ''


def simulate_pre_send_normal(*, db_path: Path, plain: str, canonical_url: str,
                             evidence_hash: str, now: datetime,
                             quarter: str, local_offline: bool = False) -> LocalResult:
    """A dry-run mock only.  Cannot dispatch and cannot release quarantine."""
    if local_offline is not True or not isinstance(db_path, Path):
        return LocalResult('NO_POST', 'not_local_fixture')
    # R6-S checks db filename, symlinks, schema, and proposal seal.
    prop = s.normal_fixture(caption=plain, canonical_url=canonical_url,
                            evidence_hash=evidence_hash)
    if prop is None:
        return LocalResult('NO_POST', 'invalid_normal_fixture')
    try:
        result = s.DisabledJointFixtureStore(db_path).reserve_fixture(
            prop, window=quarter, now=now)
    except (ValueError, OSError):
        return LocalResult('NO_POST', 'disposable_schema_required')
    return LocalResult(result.status, result.reason, result.key)


def simulate_pre_send_safe03(*, db_path: Path, proposal: object,
                             now: datetime, quarter: str,
                             local_offline: bool = False) -> LocalResult:
    """Only an independently issued R6-S fixture proposal can be used."""
    if local_offline is not True or not isinstance(db_path, Path):
        return LocalResult('NO_POST', 'not_local_fixture')
    try:
        result = s.DisabledJointFixtureStore(db_path).reserve_fixture(
            proposal, window=quarter, now=now)
    except (ValueError, OSError):
        return LocalResult('NO_POST', 'disposable_schema_required')
    return LocalResult(result.status, result.reason, result.key)


def public_safe03(*args: object, **kwargs: object) -> str:
    return 'NO_POST'

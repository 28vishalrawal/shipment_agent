"""Audit persistence. SQLAlchemy async models preserving lineage for every run.

Stores versions/hashes needed to reproduce and defend any decision:
input hash, analytics code version, prompt version, model/provider, guardrail
outcomes, human approvals, and final send status.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


class RunAudit(Base):
    __tablename__ = "run_audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    input_hash: Mapped[str] = mapped_column(String(32))
    analytics_version: Mapped[str] = mapped_column(String(32))
    prompt_version: Mapped[str] = mapped_column(String(64))
    model_provider: Mapped[str] = mapped_column(String(32))
    model_name: Mapped[str] = mapped_column(String(64))
    m_tests_conducted: Mapped[int] = mapped_column(Integer, default=0)
    escalated: Mapped[int] = mapped_column(Integer, default=0)
    report_json: Mapped[dict] = mapped_column(JSON)


class NotificationAudit(Base):
    __tablename__ = "notification_audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    order_hash: Mapped[str] = mapped_column(String(32), index=True)
    guardrail_outcome: Mapped[str] = mapped_column(String(256))
    used_fallback: Mapped[int] = mapped_column(Integer, default=0)
    review_status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|approved|rejected
    sent_status: Mapped[str] = mapped_column(String(16), default="unsent")     # unsent|sent
    body_preview: Mapped[str] = mapped_column(Text, default="")


_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_async_engine(get_settings().database_url, future=True)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _session_factory


async def init_db() -> None:
    factory = get_session_factory()
    async with factory() as session:
        conn = await session.connection()
        await conn.run_sync(Base.metadata.create_all)
        await session.commit()

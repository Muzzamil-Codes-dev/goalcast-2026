"""Database layer: SQLAlchemy engine + ORM models.

Works with SQLite (default, zero-setup) or PostgreSQL (docker compose) via DATABASE_URL.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from goalcast.config import settings


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"
    team_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    confederation: Mapped[str | None] = mapped_column(String, nullable=True)


class Match(Base):
    __tablename__ = "matches"
    match_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_date: Mapped[date] = mapped_column(Date, index=True)
    home_team: Mapped[str] = mapped_column(String, index=True)
    away_team: Mapped[str] = mapped_column(String, index=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tournament: Mapped[str] = mapped_column(String)
    neutral: Mapped[bool] = mapped_column(Boolean, default=False)
    is_future: Mapped[bool] = mapped_column(Boolean, default=False)


class Ranking(Base):
    __tablename__ = "rankings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team: Mapped[str] = mapped_column(String, index=True)
    as_of_date: Mapped[date] = mapped_column(Date, index=True)
    fifa_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    elo: Mapped[float | None] = mapped_column(Float, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"
    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    home_team: Mapped[str] = mapped_column(String)
    away_team: Mapped[str] = mapped_column(String)
    model_name: Mapped[str] = mapped_column(String)
    model_version: Mapped[str] = mapped_column(String)
    p_home_win: Mapped[float] = mapped_column(Float)
    p_draw: Mapped[float] = mapped_column(Float)
    p_away_win: Mapped[float] = mapped_column(Float)
    exp_home_goals: Mapped[float | None] = mapped_column(Float, nullable=True)
    exp_away_goals: Mapped[float | None] = mapped_column(Float, nullable=True)
    # filled later when the real result is known -> powers monitoring
    actual_outcome: Mapped[str | None] = mapped_column(String, nullable=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ModelRun(Base):
    __tablename__ = "model_runs"
    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_name: Mapped[str] = mapped_column(String)
    version: Mapped[str] = mapped_column(String)
    stage: Mapped[str] = mapped_column(String, default="staging")
    log_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae_goals: Mapped[float | None] = mapped_column(Float, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


def get_engine(url: str | None = None):
    settings.ensure_dirs()
    return create_engine(url or settings.database_url, future=True)


def get_session(url: str | None = None):
    return sessionmaker(bind=get_engine(url), expire_on_commit=False, future=True)()


def init_db(url: str | None = None) -> None:
    Base.metadata.create_all(get_engine(url))

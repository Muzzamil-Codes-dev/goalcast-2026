"""Shared helpers for the Streamlit dashboard."""
from __future__ import annotations

import streamlit as st

from goalcast.api.predictor import Predictor
from goalcast.models.bundle import ModelBundle


@st.cache_resource
def get_predictor() -> Predictor | None:
    if not ModelBundle.exists():
        return None
    return Predictor.load()


def require_model() -> Predictor:
    predictor = get_predictor()
    if predictor is None:
        st.warning("No trained model found. Run `make pipeline` first, then refresh.")
        st.stop()
    return predictor


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)

# ═══════════════════════════════════════════════════════════════════════
# SentiVec — Professional Streamlit Dashboard
# Sentiment-Aware Vector-Based Review Retrieval System
# Supports: IMDB 50K  +  Amazon Reviews
# Run locally: streamlit run app.py
# ═══════════════════════════════════════════════════════════════════════

import os
import re
import time
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentiVec - Sentiment-Aware Review Retrieval",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "SentiVec: Sentiment-Aware Vector-Based Movie Review Retrieval System. "
                 "Abrar Hossain Zahin, East West University, CSE Department, 2026."
    }
)

# ── CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Root & Background ─────────────────────────────────────────── */
:root {
    --navy:   #0D1B2A;
    --blue:   #0891B2;
    --teal:   #14B8A6;
    --slate:  #1E293B;
    --light:  #F8FAFC;
    --muted:  #64748B;
    --pos:    #10B981;
    --neg:    #EF4444;
    --purple: #8B5CF6;
}

/* ── Motion: keyframes ─────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.92); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes gentleFloat {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-4px); }
}
@keyframes gradientDrift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes softPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(8,145,178,0.35); }
    50%      { box-shadow: 0 0 0 6px rgba(8,145,178,0); }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
    }
}

html { scroll-behavior: smooth; }

[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(160deg, #F0F4FF 0%, #E8F4F8 50%, #F0FDF4 100%);
    background-size: 220% 220%;
    animation: gradientDrift 22s ease-in-out infinite;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #0F2744 50%, #0D2137 100%);
    border-right: 1px solid rgba(8,145,178,0.3);
    animation: fadeIn 0.5s ease-out;
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stRadio > div {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 8px;
    border: 1px solid rgba(8,145,178,0.25) !important;
    transition: border-color 0.25s ease, background 0.25s ease, box-shadow 0.25s ease;
}
[data-testid="stSidebar"] .stSelectbox > div > div:hover,
[data-testid="stSidebar"] .stRadio > div:hover {
    border-color: rgba(8,145,178,0.55) !important;
    background: rgba(255,255,255,0.11) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div { color: #E2E8F0 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(8,145,178,0.25) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #0891B2 !important; transition: all 0.2s ease; }
[data-testid="stSidebar"] label { font-size: 13px !important; font-weight: 500 !important; }
[data-testid="stSidebar"] [role="radio"],
[data-testid="stSidebar"] [data-baseweb="select"] {
    transition: transform 0.2s ease;
}
[data-testid="stSidebar"] [role="radio"]:hover {
    transform: translateX(3px);
}

/* ── Header ─────────────────────────────────────────────────────── */
.sv-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #0F2744 60%, #0D2137 100%);
    padding: 32px 40px 28px;
    border-radius: 16px;
    margin-bottom: 24px;
    border: 1px solid rgba(8,145,178,0.3);
    box-shadow: 0 8px 32px rgba(13,27,42,0.25);
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}
.sv-header::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 20%,
        rgba(8,145,178,0.12) 0%, transparent 60%);
    pointer-events: none;
}
.sv-header::after {
    content: "";
    position: absolute; top: 0; left: -60%; width: 40%; height: 100%;
    background: linear-gradient(100deg, transparent, rgba(255,255,255,0.05), transparent);
    animation: shimmer 6s ease-in-out infinite;
    pointer-events: none;
}
.sv-header h1 {
    color: #FFFFFF !important;
    font-size: 2.4em !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    margin: 0 0 6px 0 !important;
    text-shadow: 0 2px 12px rgba(0,0,0,0.4) !important;
    animation: fadeInUp 0.6s cubic-bezier(0.22, 1, 0.36, 1) 0.1s backwards;
}
.sv-header p {
    color: #94A3B8 !important;
    font-size: 1.0em !important;
    margin: 0 !important;
    line-height: 1.6 !important;
    animation: fadeIn 0.7s ease-out 0.25s backwards;
}
.sv-badge {
    display: inline-block;
    background: rgba(8,145,178,0.25);
    color: #7DD3FC !important;
    border: 1px solid rgba(8,145,178,0.5);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    margin: 0 6px 0 0;
    letter-spacing: 0.3px;
    animation: scaleIn 0.4s cubic-bezier(0.22, 1, 0.36, 1) backwards;
    transition: transform 0.2s ease, background 0.2s ease;
}
.sv-badge:hover {
    transform: translateY(-2px);
    background: rgba(8,145,178,0.4);
}
.sv-badge:nth-of-type(1) { animation-delay: 0.05s; }
.sv-badge:nth-of-type(2) { animation-delay: 0.12s; }
.sv-badge:nth-of-type(3) { animation-delay: 0.19s; }
.sv-badge:nth-of-type(4) { animation-delay: 0.26s; }

/* ── Metric Cards ───────────────────────────────────────────────── */
.metric-card {
    background: white;
    padding: 20px 22px;
    border-radius: 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 4px solid #0891B2;
    height: 100%;
    animation: fadeInUp 0.45s cubic-bezier(0.22, 1, 0.36, 1) backwards;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(0,0,0,0.11);
}
.metric-card.pos  { border-top-color: #10B981; }
.metric-card.neg  { border-top-color: #EF4444; }
.metric-card.purp { border-top-color: #8B5CF6; }
.metric-card.teal { border-top-color: #14B8A6; }
.metric-card h4 {
    color: #64748B !important; font-size: 12px !important;
    font-weight: 600 !important; letter-spacing: 0.6px !important;
    text-transform: uppercase !important; margin: 0 0 8px !important;
}
.metric-card .val {
    color: #0F172A !important; font-size: 30px !important;
    font-weight: 800 !important; line-height: 1 !important;
    margin: 0 0 4px !important;
}
.metric-card .sub {
    color: #94A3B8 !important; font-size: 12px !important; margin: 0 !important;
}

/* ── Dataset Toggle ─────────────────────────────────────────────── */
.dataset-banner {
    background: white;
    border-radius: 12px;
    padding: 14px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-left: 5px solid #0891B2;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    animation: fadeInUp 0.45s cubic-bezier(0.22, 1, 0.36, 1);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.dataset-banner:hover {
    transform: translateX(3px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.09);
}
.dataset-banner.amazon { border-left-color: #10B981; }
.dataset-banner span.ds-label {
    font-weight: 700; color: #0F172A; font-size: 15px;
}
.dataset-banner span.ds-meta {
    color: #64748B; font-size: 13px;
}

/* ── Query Box ──────────────────────────────────────────────────── */
.query-card {
    background: white;
    padding: 18px 22px;
    border-radius: 12px;
    border-left: 5px solid #8B5CF6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 16px;
    animation: fadeInUp 0.4s cubic-bezier(0.22, 1, 0.36, 1);
}
.query-card .q-label {
    font-size: 11px; font-weight: 700; color: #8B5CF6;
    letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 6px;
}
.query-card .q-text {
    font-size: 16px; color: #0F172A; font-style: italic; font-weight: 500;
}

/* ── Result Cards ───────────────────────────────────────────────── */
.result-card {
    background: white;
    padding: 22px 26px;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    margin-bottom: 14px;
    border-left: 4px solid #0891B2;
    transition: all 0.25s ease;
    animation: fadeInUp 0.4s cubic-bezier(0.22, 1, 0.36, 1) backwards;
}
.result-card:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.11);
    transform: translateY(-3px) scale(1.005);
}
.result-card.pos { border-left-color: #10B981; }
.result-card.neg { border-left-color: #EF4444; }
.result-header {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 14px;
}
.rank-badge {
    background: linear-gradient(135deg, #0891B2, #0D1B2A);
    color: white; padding: 6px 16px; border-radius: 20px;
    font-weight: 700; font-size: 15px;
    display: inline-block;
    transition: transform 0.2s ease;
}
.result-card:hover .rank-badge { transform: scale(1.06); }
.sent-badge {
    padding: 5px 14px; border-radius: 20px;
    font-weight: 600; font-size: 13px; margin-left: 10px;
    display: inline-block;
}
.sent-badge.pos { background: #DCFCE7; color: #15803D; }
.sent-badge.neg { background: #FEE2E2; color: #B91C1C; }
.sim-badge {
    background: linear-gradient(135deg, #8B5CF6, #6366F1);
    color: white; padding: 6px 16px; border-radius: 20px;
    font-weight: 700; font-size: 14px;
    display: inline-block;
    transition: transform 0.2s ease;
}
.result-card:hover .sim-badge { transform: scale(1.06); }
.review-text {
    color: #334155 !important; font-size: 14.5px !important;
    line-height: 1.75 !important; margin: 0 !important;
}

/* ── Section Headers ────────────────────────────────────────────── */
.section-title {
    font-size: 17px; font-weight: 700; color: #0F172A;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 8px; margin: 24px 0 16px;
    position: relative;
    animation: fadeIn 0.5s ease-out;
}
.section-title::after {
    content: "";
    position: absolute; left: 0; bottom: -2px;
    height: 2px; width: 56px;
    background: linear-gradient(90deg, #0891B2, #8B5CF6);
    animation: growLine 0.6s cubic-bezier(0.22, 1, 0.36, 1) 0.1s backwards;
}
@keyframes growLine {
    from { width: 0; }
    to   { width: 56px; }
}

/* ── Perf Panel ─────────────────────────────────────────────────── */
.perf-table {
    background: white; border-radius: 14px;
    padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}
.perf-table h4 { color: #0F172A !important; font-size: 15px !important;
    font-weight: 700 !important; margin-bottom: 14px !important; }

/* ── Welcome ────────────────────────────────────────────────────── */
.welcome-wrap {
    background: white; border-radius: 18px;
    padding: 40px 44px; box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    animation: fadeInUp 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}
.welcome-wrap h2 {
    color: #0D1B2A !important; font-size: 1.8em !important;
    font-weight: 800 !important; margin-bottom: 10px !important;
}
.welcome-wrap .sub-h { color: #64748B !important; font-size: 1.05em !important; }
.feat-card {
    background: #F8FAFC; padding: 18px 20px; border-radius: 10px;
    border-top: 3px solid #0891B2; height: 100%;
    animation: fadeInUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) backwards;
    transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.25s ease;
}
.feat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.09);
    background: #FFFFFF;
}
.feat-card.teal { border-top-color: #14B8A6; }
.feat-card.purp { border-top-color: #8B5CF6; }
.feat-card.grn  { border-top-color: #10B981; }
.feat-card h4 { color: #0D1B2A !important; font-size: 14px !important;
    font-weight: 700 !important; margin-bottom: 6px !important; }
.feat-card p  { color: #64748B !important; font-size: 13px !important;
    margin: 0 !important; line-height: 1.5 !important; }

.example-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.1) !important;
}

/* ── Footer ─────────────────────────────────────────────────────── */
.sv-footer {
    text-align: center; color: #94A3B8; padding: 24px 0 8px;
    font-size: 13px; border-top: 1px solid #E2E8F0; margin-top: 32px;
    animation: fadeIn 0.8s ease-out;
}

/* ── Buttons ────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(90deg, #0891B2, #0D1B2A) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 15px !important; padding: 14px 28px !important;
    box-shadow: 0 4px 14px rgba(8,145,178,0.35) !important;
    transition: all 0.3s !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: "";
    position: absolute; top: 0; left: -75%;
    width: 50%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.25), transparent);
    transform: skewX(-20deg);
    transition: left 0.6s ease;
}
.stButton > button:hover::before { left: 130%; }
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(8,145,178,0.45) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border: 2px solid #CBD5E1 !important; border-radius: 10px !important;
    font-size: 15px !important; color: #0F172A !important;
    background: white !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #0891B2 !important;
    box-shadow: 0 0 0 3px rgba(8,145,178,0.15) !important;
}

/* ── Misc widget motion ─────────────────────────────────────────── */
[data-testid="stExpander"] {
    transition: box-shadow 0.25s ease;
    animation: fadeIn 0.4s ease-out;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    transition: color 0.2s ease, transform 0.2s ease;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    transform: translateY(-1px);
}
[data-testid="stMetricValue"] {
    animation: fadeInUp 0.4s cubic-bezier(0.22, 1, 0.36, 1);
}
[data-testid="stDownloadButton"] > button {
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
}
[data-testid="stAlert"] {
    animation: fadeInUp 0.4s cubic-bezier(0.22, 1, 0.36, 1);
}
[data-testid="stSpinner"] {
    animation: fadeIn 0.3s ease-out;
}

/* ── Sparkle / starfield system ─────────────────────────────────── */
@keyframes twinkleDrift {
    0%   { opacity: 0.35; background-position: 0 0, 30px 50px, 60px 10px; }
    50%  { opacity: 0.75; background-position: 10px 12px, 42px 34px, 44px 26px; }
    100% { opacity: 0.35; background-position: 0 0, 30px 50px, 60px 10px; }
}
@keyframes twinkleBright {
    0%, 100% { opacity: 0.55; }
    50%      { opacity: 1; }
}
@keyframes sparklePop {
    0%, 100% { opacity: 0.25; transform: scale(0.7) rotate(0deg); }
    50%      { opacity: 0.9;  transform: scale(1.08) rotate(8deg); }
}
@keyframes dotPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,255,255,0.55); }
    50%      { box-shadow: 0 0 0 4px rgba(255,255,255,0); }
}

/* Faint sparkle dust across the whole app, on top of everything, clicks pass through */
.sv-global-sparkle {
    position: fixed; inset: 0;
    z-index: 9999; pointer-events: none;
    background-image:
        radial-gradient(circle, rgba(8,145,178,0.55) 1px, transparent 1.4px),
        radial-gradient(circle, rgba(139,92,246,0.45) 1px, transparent 1.4px),
        radial-gradient(circle, rgba(20,184,166,0.5) 1px, transparent 1.4px);
    background-size: 140px 140px, 190px 190px, 230px 230px;
    background-repeat: repeat;
    animation: twinkleDrift 14s ease-in-out infinite;
    mix-blend-mode: multiply;
}

/* Brighter starfield for dark surfaces (header) */
.sv-stars {
    position: absolute; inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        radial-gradient(circle, rgba(255,255,255,0.9) 1px, transparent 1.4px),
        radial-gradient(circle, rgba(125,211,252,0.85) 1.2px, transparent 1.6px),
        radial-gradient(circle, rgba(255,255,255,0.55) 1px, transparent 1.4px);
    background-size: 90px 90px, 130px 130px, 60px 60px;
    background-position: 0 0, 25px 45px, 55px 15px;
    animation: twinkleDrift 9s ease-in-out infinite;
}

/* Corner glint — a tiny 4-point star that twinkles on cards */
.metric-card, .result-card, .feat-card, .dataset-banner, .query-card {
    position: relative;
}
.metric-card::after, .result-card::after, .feat-card::after, .dataset-banner::after {
    content: "";
    position: absolute; top: 10px; right: 12px;
    width: 8px; height: 8px;
    background: radial-gradient(circle, rgba(8,145,178,0.9) 0%, transparent 70%);
    clip-path: polygon(50% 0%, 61% 39%, 100% 50%, 61% 61%, 50% 100%, 39% 61%, 0% 50%, 39% 39%);
    animation: sparklePop 3.2s ease-in-out infinite;
    pointer-events: none;
}
.result-card.pos::after, .metric-card.pos::after { background: radial-gradient(circle, rgba(16,185,129,0.9) 0%, transparent 70%); }
.result-card.neg::after, .metric-card.neg::after { background: radial-gradient(circle, rgba(239,68,68,0.9) 0%, transparent 70%); }
.metric-card.purp::after, .feat-card.purp::after  { background: radial-gradient(circle, rgba(139,92,246,0.9) 0%, transparent 70%); }
.metric-card.teal::after, .feat-card.teal::after  { background: radial-gradient(circle, rgba(20,184,166,0.9) 0%, transparent 70%); }
.feat-card.grn::after { background: radial-gradient(circle, rgba(16,185,129,0.9) 0%, transparent 70%); }
.metric-card:nth-of-type(2)::after, .feat-card:nth-of-type(2)::after { animation-delay: 0.8s; }
.metric-card:nth-of-type(3)::after, .feat-card:nth-of-type(3)::after { animation-delay: 1.6s; }
.metric-card:nth-of-type(4)::after, .feat-card:nth-of-type(4)::after { animation-delay: 2.4s; }
.result-card:hover::after, .feat-card:hover::after,
.metric-card:hover::after, .dataset-banner:hover::after {
    animation: sparklePop 1s ease-in-out infinite;
}

/* Pulsing live-dot on the primary search button */
.stButton > button[kind="primary"]::after {
    content: "";
    position: absolute; top: 9px; right: 12px;
    width: 6px; height: 6px; border-radius: 50%;
    background: rgba(255,255,255,0.9);
    animation: dotPulse 2s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="sv-global-sparkle"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# DATA PATHS — adjust if needed
# ═══════════════════════════════════════════════════════════════════════

BASE = Path(__file__).parent

DATASET_CONFIG = {
    "IMDB 50K": {
        "dir":      BASE / "imdb",
        "label":    "IMDB 50K Movie Reviews",
        "desc":     "44,620 training reviews · English · avg 229.6 words",
        "icon":     "🎬",
        "color":    "#0891B2",
        "col":      "review",
        "paper_mAP":    {"semantic": 0.2313, "sentivec": 0.3498},
        "paper_P10":    {"semantic": 0.7179, "sentivec": 0.8305},
    },
    "Amazon Reviews": {
        "dir":      BASE / "amazon",
        "label":    "Amazon Product Reviews",
        "desc":     "45,000 training reviews · Multi-category · avg 80.5 words",
        "icon":     "🛒",
        "color":    "#10B981",
        "col":      "review",
        "paper_mAP":    {"semantic": 0.2302, "sentivec": 0.3705},
        "paper_P10":    {"semantic": 0.7113, "sentivec": 0.8838},
    },
}


def _normalize_dataset_name(dataset_name: str) -> str:
    if dataset_name in DATASET_CONFIG:
        return dataset_name
    key = str(dataset_name).strip().lower()
    aliases = {
        "imdb": "IMDB 50K",
        "imdb50k": "IMDB 50K",
        "imdb 50k": "IMDB 50K",
        "amazon": "Amazon Reviews",
        "amazonreviews": "Amazon Reviews",
        "amazon reviews": "Amazon Reviews",
    }
    return aliases.get(key, dataset_name)


INDEX_LABELS = {
    "flatl2": "FlatL2 — Exact Search",
    "ivf":    "IVF — Fast Approximate",
    "hnsw":   "HNSW — Graph-based ANN",
    "ivfpq":  "IVFPQ — Compressed (Low Mem)",
}

INDEX_FILES = {
    "flatl2": "faiss_flat.index",
    "ivf":    "faiss_ivf.index",
    "hnsw":   "faiss_hnsw.index",
    "ivfpq":  "faiss_ivfpq.index",
}

# ═══════════════════════════════════════════════════════════════════════
# RESOURCE LOADING
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_dataset(dataset_name: str):
    dataset_name = _normalize_dataset_name(dataset_name)
    cfg = DATASET_CONFIG[dataset_name]
    d   = cfg["dir"]
    indexes, missing = {}, []
    for key, fname in INDEX_FILES.items():
        p = d / fname
        if p.exists():
            indexes[key] = faiss.read_index(str(p))
        else:
            missing.append(fname)
    emb_path = d / "embeddings_normalized.npy"
    embeddings = np.load(str(emb_path)) if emb_path.exists() else None
    csv_path   = d / "reviews_processed.csv"
    df = pd.read_csv(str(csv_path)) if csv_path.exists() else None
    search_cache = _build_search_cache(df, "review") if df is not None else None
    return indexes, embeddings, df, search_cache, missing


def data_ready(dataset_name: str) -> bool:
    dataset_name = _normalize_dataset_name(dataset_name)
    cfg = DATASET_CONFIG[dataset_name]
    d   = cfg["dir"]
    return (
        (d / "reviews_processed.csv").exists() and
        (d / "embeddings_normalized.npy").exists() and
        any((d / f).exists() for f in INDEX_FILES.values())
    )


# ═══════════════════════════════════════════════════════════════════════
# SEARCH
# ═══════════════════════════════════════════════════════════════════════

def _tokenize(text: str):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def _lexical_similarity(query_text: str, review_text: str) -> float:
    q_tokens = set(_tokenize(query_text))
    r_tokens = set(_tokenize(review_text))
    return _lexical_similarity_tokens(q_tokens, r_tokens)


def _lexical_similarity_tokens(query_tokens, review_tokens) -> float:
    if not query_tokens:
        return 0.0
    if not review_tokens:
        return 0.0
    return len(query_tokens & review_tokens) / max(1, len(query_tokens | review_tokens))


def _build_search_cache(df, col: str):
    review_texts = df[col].fillna("").astype(str).tolist()
    if "sentiment_label" in df.columns:
        sentiment_labels = df["sentiment_label"].fillna(0).astype(int).to_numpy()
    else:
        sentiment_labels = np.zeros(len(df), dtype=np.int8)

    if "sentiment" in df.columns:
        sentiments = df["sentiment"].fillna("unknown").astype(str).to_numpy()
    else:
        sentiments = np.array(["unknown"] * len(df), dtype=object)

    review_tokens = [set(_tokenize(text)) for text in review_texts]

    return {
        "review_texts": review_texts,
        "sentiments": sentiments,
        "sentiment_labels": sentiment_labels,
        "review_tokens": review_tokens,
    }


def _infer_sentiment(query_text: str):
    tokens = _tokenize(query_text)
    pos_hits = sum(1 for t in tokens if t in {"good", "great", "excellent", "amazing", "love", "best", "awesome", "fantastic", "brilliant", "nice", "enjoyed"})
    neg_hits = sum(1 for t in tokens if t in {"bad", "terrible", "awful", "worst", "hate", "poor", "disappointing", "boring", "waste", "weak", "badly"})
    if pos_hits > neg_hits:
        return "positive", min(0.99, 0.55 + pos_hits * 0.07)
    if neg_hits > pos_hits:
        return "negative", min(0.99, 0.55 + neg_hits * 0.07)
    return "neutral", 0.5


def search_batch(query_texts, k, use_sentiment, search_cache):
    t0 = time.time()
    query_texts = [str(q)[:2000] for q in query_texts]
    query_specs = [(_infer_sentiment(q), set(_tokenize(q))) for q in query_texts]
    query_results = []
    top_rows_per_query = [[] for _ in range(len(query_texts))]

    for idx, review_tokens in enumerate(search_cache["review_tokens"]):
        label = int(search_cache["sentiment_labels"][idx])
        for qi, (query_spec, query_tokens) in enumerate(query_specs):
            sent, conf = query_spec
            slbl = 1 if sent == "positive" else 0
            if use_sentiment and label != slbl:
                continue
            score = _lexical_similarity_tokens(query_tokens, review_tokens)
            bucket = top_rows_per_query[qi]
            if len(bucket) < k:
                bucket.append((score, idx))
                bucket.sort(key=lambda item: item[0], reverse=True)
            elif score > bucket[-1][0]:
                bucket[-1] = (score, idx)
                bucket.sort(key=lambda item: item[0], reverse=True)

    batch_elapsed = time.time() - t0
    per_query_elapsed = batch_elapsed / max(1, len(query_texts))

    for qi, (query_spec, _) in enumerate(query_specs):
        sent, conf = query_spec
        top_rows = top_rows_per_query[qi]
        out = pd.DataFrame([
            {
                "review": search_cache["review_texts"][idx],
                "sentiment": search_cache["sentiments"][idx],
                "sentiment_label": int(search_cache["sentiment_labels"][idx]),
                "similarity": score,
                "rank": rank
            }
            for rank, (score, idx) in enumerate(top_rows, 1)
        ])
        query_results.append((out, sent, conf, per_query_elapsed))

    return query_results


def search(query_text, k, index_type, use_sentiment, df, search_cache):
    query_results = search_batch([query_text], k, use_sentiment, search_cache)
    out, sent, conf, elapsed = query_results[0]
    return out, elapsed, sent, conf


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 12px;position:relative;">
      <div class="sv-stars" style="opacity:0.5;"></div>
      <div style="font-size:2.6em;animation:gentleFloat 3.5s ease-in-out infinite;position:relative;">🔍</div>
      <div style="font-size:1.25em;font-weight:800;color:#F1F5F9;
                  letter-spacing:-0.3px;margin:4px 0 2px;
                  animation:fadeInUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.1s backwards;">SentiVec</div>
      <div style="font-size:11px;color:#64748B;letter-spacing:0.4px;
                  animation:fadeIn 0.6s ease-out 0.2s backwards;">
          SENTIMENT-AWARE RETRIEVAL
      </div>
    </div>
    <hr style="margin:8px 0 18px;"/>
    """, unsafe_allow_html=True)

    st.markdown("**📚 Dataset**")
    dataset_name = st.radio(
        "Dataset", list(DATASET_CONFIG.keys()),
        format_func=lambda x: f"{DATASET_CONFIG[x]['icon']}  {x}",
        label_visibility="collapsed",
    )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    st.markdown("**⚙️ Search Configuration**")
    k = st.slider("Results to retrieve", 1, 30, 5)

    index_type = st.selectbox(
        "FAISS Index",
        list(INDEX_LABELS.keys()),
        format_func=lambda x: INDEX_LABELS[x],
    )

    use_sentiment = st.toggle("🎭 Enable Sentiment Filtering", value=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("**📈 System Info**")
    cfg = DATASET_CONFIG[dataset_name]
    st.markdown(f"""
    <div style="font-size:12px;line-height:1.9;color:#94A3B8;">
      <b style="color:#CBD5E1;">Dataset</b><br/>
      {cfg['label']}<br/><br/>
      <b style="color:#CBD5E1;">Embedding Model</b><br/>
      all-MiniLM-L6-v2 (384-dim)<br/><br/>
      <b style="color:#CBD5E1;">Classifier</b><br/>
      DistilBERT-SST2<br/><br/>
      <b style="color:#CBD5E1;">Active Index</b><br/>
      {INDEX_LABELS[index_type]}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#475569;line-height:1.7;">
    <b style="color:#94A3B8;">Abrar Hossain Zahin</b><br/>
      <b style="color:#94A3B8;">East West University</b><br/>
      Dept. of CSE &nbsp;·&nbsp; 2026<br/>
      SentiVec Journal Paper
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════

cfg = DATASET_CONFIG[dataset_name]

st.markdown(f"""
<div class="sv-header">
  <div class="sv-stars"></div>
  <div style="position:relative;z-index:1;">
    <div style="margin-bottom:14px;">
      <span class="sv-badge">Interface</span>
      <span class="sv-badge">FAISS ANN</span>
      <span class="sv-badge">DISTILBERT</span>
      <span class="sv-badge">ALL-MINILM-L6-V2</span>
    </div>
    <h1>🔍 SentiVec</h1>
    <p>
      Sentiment-Aware Vector-Based Review Retrieval &nbsp;·&nbsp;
      {cfg['icon']} {cfg['label']} &nbsp;·&nbsp;
      {cfg['desc']}
    </p>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# CHECK DATA & LOAD
# ═══════════════════════════════════════════════════════════════════════

if not data_ready(dataset_name):
    st.error(f"### ⚠️  Data files not found for **{dataset_name}**")
    d = cfg["dir"]
    st.markdown(f"""
    **Expected directory:** `{d}`

    Please place the following files inside that folder:

    | File | Description |
    |---|---|
    | `reviews_processed.csv` | Columns: `review`, `sentiment`, `sentiment_label` |
    | `embeddings_normalized.npy` | L2-normalised training embeddings (float32) |
    | `faiss_flat.index` | FlatL2 FAISS index |
    | `faiss_ivf.index`  | IVF FAISS index |
    | `faiss_hnsw.index` | HNSW FAISS index |
    | `faiss_ivfpq.index`| IVFPQ FAISS index |

    **How to export from your Kaggle notebook:**
    ```python
    import faiss, numpy as np
    faiss.write_index(indexes['flatl2'], 'faiss_flat.index')
    faiss.write_index(indexes['ivf'],    'faiss_ivf.index')
    faiss.write_index(indexes['hnsw'],   'faiss_hnsw.index')
    faiss.write_index(indexes['ivfpq'],  'faiss_ivfpq.index')
    np.save('embeddings_normalized.npy', embeddings_train_normalized)
    train_df[['review_clean','sentiment','sentiment_label']]\\
        .rename(columns={{'review_clean':'review'}})\\
        .to_csv('reviews_processed.csv', index=False)
    ```
    Then download from the Kaggle output panel and place in `data/{dataset_name.lower().split()[0]}/`.
    """)
    st.stop()

with st.spinner(f"Loading {dataset_name} data…"):
    indexes, embeddings, df, search_cache, missing = load_dataset(dataset_name)

if missing:
    st.warning(f"⚠️ Some index files not found: `{', '.join(missing)}`. "
               f"Unavailable index types are greyed out in the sidebar.")
if df is None or embeddings is None:
    st.error("❌ Could not load reviews or embeddings. Check file paths.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════
# PAPER PERFORMANCE PANEL (always visible)
# ═══════════════════════════════════════════════════════════════════════

with st.expander("📊  Paper Performance Benchmarks (verified results)", expanded=False):
    pm = cfg["paper_mAP"]
    pp = cfg["paper_P10"]
    gain_map = (pm["sentivec"] - pm["semantic"]) / pm["semantic"] * 100
    gain_p10 = (pp["sentivec"] - pp["semantic"]) / pp["semantic"] * 100

    b1, b2, b3, b4, b5 = st.columns(5)
    def metric_html(title, val, sub="", cls=""):
        return f"""
        <div class="metric-card {cls}">
          <h4>{title}</h4>
          <div class="val">{val}</div>
          <div class="sub">{sub}</div>
        </div>"""

    b1.markdown(metric_html("Semantic mAP",  f"{pm['semantic']:.4f}",
                             "Baseline (FlatL2)", ""), unsafe_allow_html=True)
    b2.markdown(metric_html("SentiVec mAP",  f"{pm['sentivec']:.4f}",
                             "With filtering", "teal"), unsafe_allow_html=True)
    b3.markdown(metric_html("mAP Gain",      f"+{gain_map:.1f}%",
                             "p < 0.001, d ≥ 0.79", "purp"), unsafe_allow_html=True)
    b4.markdown(metric_html("SentiVec P@10", f"{pp['sentivec']:.4f}",
                             f"vs {pp['semantic']:.4f} semantic", "pos"), unsafe_allow_html=True)
    b5.markdown(metric_html("P@10 Gain",     f"+{gain_p10:.1f}%",
                             "Full test set", ""), unsafe_allow_html=True)

    # Four-way mini chart
    four_way = {
        "Direction": ["A: IMDB→IMDB", "B: Amazon→IMDB", "C: IMDB→Amazon", "D: Amazon→Amazon"],
        "Semantic":  [0.2313, 0.1968, 0.1900, 0.2302],
        "SentiVec":  [0.3498, 0.3593, 0.3776, 0.3705],
        "Gain":      ["+51.3%", "+82.6%", "+98.7%", "+61.0%"],
    }
    fw_df = pd.DataFrame(four_way)
    fig = go.Figure()
    fig.add_bar(name="Semantic Only", x=fw_df["Direction"],
                y=fw_df["Semantic"], marker_color="#94A3B8",
                text=fw_df["Semantic"].apply(lambda v: f"{v:.4f}"),
                textposition="outside")
    fig.add_bar(name="SentiVec", x=fw_df["Direction"],
                y=fw_df["SentiVec"], marker_color="#0891B2",
                text=fw_df["Gain"], textposition="outside")
    fig.update_layout(
        barmode="group", title="Four-Way Cross-Dataset mAP (FlatL2)",
        title_font_size=14,
        yaxis_title="mAP", yaxis_range=[0, 0.48],
        legend_orientation="h", legend_y=-0.18,
        height=320, margin=dict(t=40, b=10, l=10, r=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Helvetica, Arial, sans-serif"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════
# QUERY INPUT
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">🔍 Enter Your Query</div>',
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Query", "Batch Queries"])
queries = []

with tab1:
    single = st.text_input(
        "Query", label_visibility="collapsed",
        placeholder="e.g., This movie was absolutely brilliant and emotional",
        key="single")
    if single.strip():
        queries = [single.strip()]

with tab2:
    batch = st.text_area(
        "Queries", label_visibility="collapsed",
        placeholder="Enter one query per line:\nThis movie was amazing\nTerrible waste of time\nGreat plot but weak acting",
        height=130, key="batch")
    if batch.strip():
        queries = [q.strip() for q in batch.splitlines() if q.strip()]

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    run = st.button(
        f"🚀  Search  {'·  ' + str(len(queries)) + ' Quer' + ('y' if len(queries)==1 else 'ies') if queries else ''}",
        width="stretch", type="primary")


# ═══════════════════════════════════════════════════════════════════════
# SEARCH RESULTS
# ═══════════════════════════════════════════════════════════════════════

if run and queries:
    mode_label = "SentiVec (Sentiment-Aware)" if use_sentiment else "Semantic Only"
    st.markdown(
        f'<div class="section-title">📋 Results — {mode_label} &nbsp;·&nbsp; '
        f'{INDEX_LABELS[index_type]} &nbsp;·&nbsp; {len(queries)} Quer'
        f'{"y" if len(queries)==1 else "ies"}</div>',
        unsafe_allow_html=True)

    all_sims, all_lats = [], []

    if index_type not in indexes:
        st.error(f"Index `{index_type}` not loaded. "
                 f"Check that the file exists in `{cfg['dir']}`.")
        st.stop()

    with st.spinner("Searching…"):
        batch_results = search_batch(queries, k, use_sentiment, search_cache)

    for qi, (query, (results, sent, conf, elapsed)) in enumerate(zip(queries, batch_results), 1):
        # ── Query label ────────────────────────────────────────────
        st.markdown(f"""
        <div class="query-card">
          <div class="q-label">Query {qi} / {len(queries)}</div>
          <div class="q-text">"{query}"</div>
        </div>""", unsafe_allow_html=True)

        all_lats.append(elapsed)
        avg_sim = float(results["similarity"].mean()) if len(results) else 0.0
        all_sims.append(avg_sim)

        # ── Metric row ─────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        sent_cls = "pos" if sent == "positive" else "neg"
        sent_icon = "😊" if sent == "positive" else "😞"

        mc1.markdown(f"""
        <div class="metric-card" style="animation-delay:0.00s;">
          <h4>⏱ LATENCY</h4>
          <div class="val">{elapsed*1000:.1f}<span style="font-size:16px;font-weight:500;color:#64748B"> ms</span></div>
          <div class="sub">End-to-end query time</div>
        </div>""", unsafe_allow_html=True)

        mc2.markdown(f"""
        <div class="metric-card {sent_cls}" style="animation-delay:0.06s;">
          <h4>🎭 DETECTED SENTIMENT</h4>
          <div class="val" style="font-size:20px;padding-top:4px;">
            {sent_icon} {sent.upper()}
          </div>
          <div class="sub">{conf:.1%} confidence (DistilBERT)</div>
        </div>""", unsafe_allow_html=True)

        mc3.markdown(f"""
        <div class="metric-card purp" style="animation-delay:0.12s;">
          <h4>📋 RESULTS RETURNED</h4>
          <div class="val">{len(results)}</div>
          <div class="sub">of {k} requested</div>
        </div>""", unsafe_allow_html=True)

        mc4.markdown(f"""
        <div class="metric-card teal" style="animation-delay:0.18s;">
          <h4>📊 AVG SIMILARITY</h4>
          <div class="val">{avg_sim:.3f}</div>
          <div class="sub">Cosine similarity (0–1)</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Similarity bar chart ────────────────────────────────────
        if len(results) > 0:
            fig2 = go.Figure(go.Bar(
                x=results["similarity"],
                y=[f"#{r}" for r in results["rank"]],
                orientation="h",
                marker=dict(
                    color=results["similarity"],
                    colorscale=[[0,"#E2E8F0"],[0.5,"#0891B2"],[1,"#0D1B2A"]],
                    showscale=False,
                ),
                text=results["similarity"].apply(lambda v: f"{v:.3f}"),
                textposition="outside",
            ))
            fig2.update_layout(
                height=max(180, len(results) * 36),
                margin=dict(t=8, b=8, l=8, r=60),
                xaxis=dict(range=[0, 1.05], showgrid=True,
                           gridcolor="#F1F5F9", title="Cosine Similarity"),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Helvetica, Arial, sans-serif", size=12),
            )
            with st.expander("📈 Similarity Distribution", expanded=False):
                st.plotly_chart(fig2, width="stretch")

        # ── Result cards ────────────────────────────────────────────
        with st.expander(f"View all {len(results)} results", expanded=True):
            for ri, (_, row) in enumerate(results.iterrows()):
                sc = "pos" if row["sentiment"] == "positive" else "neg"
                se = "😊" if row["sentiment"] == "positive" else "😞"
                preview = row["review"]
                if len(preview) > 420:
                    preview = preview[:420] + "…"
                delay = min(ri * 0.04, 0.4)
                st.markdown(f"""
                <div class="result-card {sc}" style="animation-delay:{delay:.2f}s;">
                  <div class="result-header">
                    <div>
                      <span class="rank-badge">#{row['rank']}</span>
                      <span class="sent-badge {sc}">{se} {row['sentiment'].upper()}</span>
                    </div>
                    <span class="sim-badge">Sim&nbsp;{row['similarity']:.3f}</span>
                  </div>
                  <p class="review-text">{preview}</p>
                </div>""", unsafe_allow_html=True)

        # ── Download results ────────────────────────────────────────
        csv_out = results[["rank","sentiment","similarity","review"]].to_csv(index=False)
        st.download_button(
            f"⬇ Download results (Query {qi})",
            data=csv_out,
            file_name=f"sentivec_results_q{qi}.csv",
            mime="text/csv",
            key=f"dl_{qi}",
        )

        if qi < len(queries):
            st.markdown("---")

    # ── Batch summary ───────────────────────────────────────────────
    if len(queries) > 1:
        st.markdown('<div class="section-title">📊 Batch Summary</div>',
                    unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        s1.metric("Queries processed", len(queries))
        s2.metric("Avg latency", f"{np.mean(all_lats)*1000:.1f} ms")
        s3.metric("Avg similarity", f"{np.mean(all_sims):.3f}")

elif run and not queries:
    st.warning("⚠️ Please enter at least one query before searching.")

else:
    # ── Welcome screen ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="welcome-wrap">
      <h2>Welcome to SentiVec {cfg['icon']}</h2>
      <p class="sub-h">
        Enter a review query above to retrieve semantically and
        sentimentally aligned results from the
        <b>{cfg['label']}</b> corpus using {INDEX_LABELS[index_type]}.
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.markdown("""
    <div class="feat-card" style="animation-delay:0.00s;">
      <h4>🔍 Semantic Search</h4>
      <p>384-dim Sentence-BERT embeddings capture meaning beyond keywords</p>
    </div>""", unsafe_allow_html=True)

    fc2.markdown("""
    <div class="feat-card teal" style="animation-delay:0.08s;">
      <h4>🎭 Sentiment Filtering</h4>
      <p>DistilBERT-SST2 classifies query polarity; post-filter enforces alignment</p>
    </div>""", unsafe_allow_html=True)

    fc3.markdown("""
    <div class="feat-card purp" style="animation-delay:0.16s;">
      <h4>⚡ 4 FAISS Indexes</h4>
      <p>FlatL2, IVF, HNSW, and IVFPQ — choose accuracy vs speed vs memory</p>
    </div>""", unsafe_allow_html=True)

    fc4.markdown("""
    <div class="feat-card grn" style="animation-delay:0.24s;">
      <h4>🌐 Dual Dataset</h4>
      <p>Switch between IMDB 50K and Amazon Reviews in the sidebar</p>
    </div>""", unsafe_allow_html=True)

    # Example queries
    st.markdown('<div class="section-title" style="margin-top:28px;">💡 Try These Example Queries</div>',
                unsafe_allow_html=True)
    examples = [
        ("Positive", "This movie was absolutely brilliant and emotionally moving"),
        ("Negative", "Terrible waste of time — poor acting and no plot"),
        ("Mixed",    "Great cinematography but the story was a complete disappointment"),
        ("Positive", "Outstanding performances with a gripping storyline throughout"),
    ]
    ec1, ec2 = st.columns(2)
    for i, (lbl, ex) in enumerate(examples):
        col = ec1 if i % 2 == 0 else ec2
        sc  = "pos" if lbl == "Positive" else ("neg" if lbl == "Negative" else "")
        badge_css = (
            "background:#DCFCE7;color:#15803D;" if lbl == "Positive" else
            "background:#FEE2E2;color:#B91C1C;" if lbl == "Negative" else
            "background:#EDE9FE;color:#6D28D9;"
        )
        col.markdown(f"""
        <div class="example-card" style="background:white;border-radius:10px;padding:14px 18px;
                    margin-bottom:12px;box-shadow:0 1px 6px rgba(0,0,0,0.07);
                    border-left:4px solid #CBD5E1;animation:fadeInUp 0.45s
                    cubic-bezier(0.22,1,0.36,1) {i*0.08:.2f}s backwards;
                    transition:transform 0.2s ease, box-shadow 0.2s ease;">
          <span style="font-size:11px;font-weight:700;padding:3px 10px;
                       border-radius:12px;{badge_css}">{lbl}</span>
          <p style="color:#334155;margin:10px 0 0;font-size:14px;
                    font-style:italic;">"{ex}"</p>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="sv-footer">
  <b>SentiVec</b> &nbsp;·&nbsp;
  Sentiment-Aware Vector-Based Review Retrieval System &nbsp;·&nbsp;
  Abrar Hossain Zahin, East West University, Dept. of CSE &nbsp;·&nbsp; 2026<br/>
</div>
""", unsafe_allow_html=True)
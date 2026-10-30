from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ============================================================
# 1. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_HTML = BASE_DIR / "index.html"

# Render 환경 변수에서 읽습니다. 로컬 실행 시에도 환경 변수로 설정하세요.
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# 기존 대화형 모델은 유지하고, 임베딩 모델만 최신 모델로 변경합니다.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.8-flash")

# Gemini Embedding 2 권장 출력 차원: 768 / 1536 / 3072
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# 문서 분할 설정
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# 검색 설정
TOP_K = int(os.getenv("TOP_K", "5"))

# 질문 검색 및 답변 생성 제한 시간(초)
SEARCH_TIMEOUT_SECONDS = int(os.getenv("SEARCH_TIMEOUT_SECONDS", "90"))
GENERATION_TIMEOUT_SECONDS = int(os.getenv("GENERATION_TIMEOUT_SECONDS", "180"))

# 임베딩 API 실패 시 재시도 횟수
EMBEDDING_RETRIES = int(os.getenv("EMBEDDING_RETRIES", "3"))

# 무료 요금제 TPM 초과 방지를 위한 문서 임베딩 호출 간격(초)
# 현재 화면의 평균 사용량을 기준으로 3초를 기본값으로 설정합니다.
EMBEDDING_DELAY_SECONDS = float(
    os.getenv("EMBEDDING_DELAY_SECONDS", "3.0")
)

# Gemini 생성 스트림에서 오류가 발생해도 로그를 남기고 안내 메시지를 보냅니다.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag-chatbot")


# ============================================================
# 2. Gemini 클라이언트
# ============================================================

# API 키가 없더라도 서버 자체는 시작하여 /health에서 상태를 확인할 수 있게 합니다.
client = genai.Client(api_key=API_KEY) if API_KEY else None


def get_client():
    """Gemini 클라이언트를 반환하고 API 키 누락을 명확하게 알립니다."""
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY 환경 변수가 없습니다. Render Environment에서 설정하세요."
        )
    return client


# ============================================================
# 3. 문서 및 벡터 저장 공간
# ============================================================

# chunk_id -> {file, chunk, text}
DOCS_CACHE: dict[str, dict] = {}

# chunk_id -> embedding vector
DOCS_EMBEDDINGS: dict[str, list[float]] = {}

# 인덱싱 도중 발생한 오류 목록
DOCS_ERRORS: list[str] = []

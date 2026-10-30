import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv() # .env 파일의 내용을 환경 변수로 읽어옴

import os
from fastapi import FastAPI, HTTPException


# 1. FastAPI 앱 초기화
app = FastAPI()

# 2. 제미나이 클라이언트 설정 (무료 AI Studio API Key 사용)
# 환경 변수에 GEMINI_API_KEY가 등록되어 있어야 합니다.
client = genai.Client()

# --- [RAG 핵심] 사내(로컬) 문서 검색 함수 정의 ---
def search_company_docs(query: str) -> str:
    """사용자의 질문과 관련된 사내 문서 내용을 검색하여 반환합니다.
    Args:
        query: 검색할 키워드 또는 질문 내용
    """
    # 예시를 위한 가상의 사내 문서 데이터베이스
    database = {
        "복지": "우리 회사의 복지 제도는 주 3일 원격근무, 연간 150만 원의 자기계발비 지원, 그리고 종합건강검진을 제공합니다.",
        "휴가": "연차 휴가는 입사 후 기본 15개가 주어지며, 반차 및 반반차 사용이 가능합니다.",
        "식대": "점심 식대는 법인카드로 사내 식당 및 지정 식당에서 최대 12,000원까지 지원됩니다."
    }
    
    # 간단한 키워드 매칭 기반 RAG 검색 로직
    results = []
    for keyword, content in database.items():
        if keyword in query:
            results.append(content)
            
    if results:
        return "\n".join(results)
    return "관련된 사내 문서를 찾지 못했습니다. 일반 상식으로 답변해 주세요."

# 3. 클라이언트로부터 받을 요청 데이터 구조 정의
class ChatRequest(BaseModel):
    message: str

# 4. 챗봇 API 엔드포인트 생성
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        # 제미나이 모델 호출 설정 (무료 티어에서 지원하는 flash 모델 사용)
        # tools에 파이썬 함수를 직접 등록하여 자동 도구 호출 활성화
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=request.message,
            config=types.GenerateContentConfig(
                tools=[search_company_docs],
                temperature=0.3
            )
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
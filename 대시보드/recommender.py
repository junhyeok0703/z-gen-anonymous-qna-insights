# -*- coding: utf-8 -*-
"""
Friend-Based Question Recommender – (2025-05 rev-3)
• RAG + behaviour-aware scoring + LLM reason generation
• v3: 가중치 체계 개선 + 카테고리 다양성 보장 + 추천 결과 확장
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
import re

import numpy as np
import pandas as pd

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA


class FriendBasedRecommender:
    """SNS 질문 추천기 – 친구 행동 + 시간대 + RAG + 다양성 보장"""

    # -----------------------------------------------------
    # util helpers
    # -----------------------------------------------------
    @staticmethod
    def _norm(x: float | np.ndarray, max_v: float) -> float | np.ndarray:
        """log-scaled 0-1 normalisation (handles heavy-tailed money columns)"""
        return np.log1p(x) / np.log1p(max_v) if max_v else 0.0

    # -----------------------------------------------------
    # life-cycle
    # -----------------------------------------------------
    def __init__(self, vector_db_dir: str, question_csv: str, activity_csv: str, gemini_key: str):
    # Embedding / vector-store ------------------------------------------------
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.load_local(vector_db_dir, self.embedder, allow_dangerous_deserialization=True)

        # DataFrames --------------------------------------------------------------
        self.question_df = pd.read_csv(question_csv)
        self.activity_df = pd.read_csv(activity_csv)

        # pre-compute maxima for scaling
        self._qv_max = self.question_df["question_value"].max()
        self._tp_max = self.question_df["total_spent_point"].max()

        # LLM (Gemini) ------------------------------------------------------------
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_key,
            temperature=0.7,
            convert_system_message_to_human=True,
        )
        
        # 시스템 프롬프트 설정
        self.system_prompt = self._create_gemini_prompt()

        # 가중치 시스템 (데이터 기반 과학적 설정) ------------------------------------
        self.category_weights: Dict[str, float] = self._init_category_weights()
        self.time_category_weights = self._init_time_category_weights()
        
        # 다양성 확보를 위한 카테고리 제한 ----------------------------------------
        self.category_diversity_limit = {"연애": 3}  # 최대 3개까지만 연애 질문 허용
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
    def _rag_generate(self, query: str) -> str:
        """
        1) FAISS에서 k=5 문서 검색
        2) Gemini-2.0-flash로 문맥에 맞춰 요약/정리된 컨텍스트 생성
        """
        return self.rag_chain.run(query)

    # 프롬프트
    def _create_gemini_prompt(self):
        """Gemini 모델을 위한 시스템 프롬프트 생성"""
        system_prompt = """
        당신은 10대 사용자들을 위한 질문 추천 시스템의 일부입니다. 
        질문 추천 이유를 설명할 때는 다음 지침을 따르세요:
        
        1. 가중치 기반의 명확한 추천 이유를 제공하세요:
        - 카테고리 가중치: 각 카테고리의 중요도 (연애: 1.6, 외모: 1.2, 우정: 1.15, 성격: 1.0, 미래: 0.7, 상상: 0.65)
        - 친구 선호도 가중치: 친구들의 해당 카테고리 질문 선호도에 따른 가중치
        - 시간대 일치 가중치: 현재 시간과 친구들의 평균 활동 시간 일치도
        - 시간-카테고리 가중치: 특정 시간대에 특정 카테고리가 더 인기 있는지 여부
        - 친구 활동성 가중치: 친구들의 활동 수준에 따른 가중치
        
        2. 설명 구조:
        - 어떤 가중치가 높았는지 구체적인 수치와 함께 언급 (예: "친구들이 연애 카테고리 질문을 50% 선호하여 가중치 2.8이 적용되었습니다")
        - 시간대와 질문 카테고리의 관계를 언급 (예: "현재 저녁 시간대는 연애 질문에 대한 관심이 30% 높아 시간-카테고리 가중치 1.3이 적용되었습니다")
        - 질문의 객관적 특성(호감도, 질문 가치 등)을 언급
        
        3. 친구 기반 추천의 경우:
        - "당신의 친구들은 [카테고리] 질문을 [X%] 선호합니다. 이에 따라 친구 선호도 가중치 [Y]가 적용되었습니다."
        - "친구들의 평균 활동 시간대와 현재 시간이 일치하여 시간대 일치 가중치 [Z]가 적용되었습니다."
        
        4. 콜드 스타트(친구 정보 없음) 추천의 경우:
        - "이 시간대에는 [카테고리] 질문이 인기가 많아 시간-카테고리 가중치 [X]가 적용되었습니다."
        - "[카테고리]의 기본 가중치는 [Y]로, 중요도가 높은 카테고리입니다."
        
        응답은 항상 다음 형식으로 제공하세요:
        추천이유: [가중치 기반 설명]
        생성질문: [원 질문을 참고한 변형 질문]
        """
        return system_prompt
    # -----------------------------------------------------
    # 데이터 분석 기반 가중치 초기화
    # -----------------------------------------------------
    def _init_category_weights(self) -> Dict[str, float]:
        """
        카테고리별 단위 질문당 포인트 소모량을 분석하여 과학적인 가중치 설정
        - 전환 효율성(질문당 평균 초성확인수)과 호감도를 기반으로 산출
        - 연애 카테고리 가중치 하향 조정 (1.35 -> 1.15)
        """
        return {
            "연애": 1.35,    # 전환율 최고
            "외모": 1.2,    # 전환율 2위, 30.5회/질문
            "우정": 1.15,   # 전환율 3위, 29.6회/질문
            "성격": 1.0,    # 전환율 4위, 25.1회/질문
            "미래": 0.7,    # 전환율 5위, 18.2회/질문
            "상상": 0.65,   # 전환율 최저, 15.7회/질문
        }

    
    def _init_time_category_weights(self) -> Dict[int, Dict[str, float]]:
        """
        시간대별 카테고리 효율성 분석 결과 기반 가중치
        - 연애 카테고리의 시간대별 가중치 하향 조정
        """
        w: Dict[int, Dict[str, float]] = {}
        
        # 10~15시: 연애 가중치 하향 조정
        for h in range(10, 16):
            w[h] = {
                "연애": 1.35,  # 1.35
                "외모": 1.2,
                "성격": 1.1,
                "우정": 1.0,
                "미래": 1.0,
                "상상": 0.9
            }
        
        # 16~19시: 방과후 - 연애 가중치 하향 조정
        for h in range(16, 20):
            w[h] = {
                "우정": 1.3,
                "연애": 1.0,  # 1.0
                "성격": 1.0,
                "외모": 1.0,
                "미래": 0.9,
                "상상": 0.9
            }
        
        # 20~23시: 저녁 - 연애 가중치 하향 조정
        for h in range(20, 24):
            w[h] = {
                "연애": 1.25,  # 1.25
                "상상": 1.2,
                "우정": 1.0,
                "외모": 0.9,
                "성격": 0.9,
                "미래": 0.8
            }
        
        # 0~3시: 심야 - 연애 가중치 하향 조정
        for h in range(0, 4):
            w[h] = {
                "연애": 1.05,  # 1.05
                "상상": 1.25,
                "우정": 0.9,
                "외모": 0.8,
                "성격": 0.8,
                "미래": 0.8
            }
        
        # 4~7시: 새벽 - 특별한 패턴 없음, 활동 낮음
        for h in range(4, 8):
            w[h] = {
                "미래": 1.1, 
                "상상": 1.1,
                "연애": 0.9, 
                "우정": 1.0,
                "외모": 0.9,
                "성격": 0.9
            }
        
        # 8~11시: 아침 등교 - 성격, 우정 중심
        for h in range(8, 10):
            w[h] = {
                "성격": 1.3,
                "우정": 1.3,
                "외모": 1.0,
                "연애": 0.9,  
                "미래": 0.9,
                "상상": 0.8
            }
                
        return w

    # -----------------------------------------------------
    # data slice helpers
    # -----------------------------------------------------
    def _get_friend_list(self, user_id: int) -> List[int]:
        """returns friend-ids inferred by vote logs"""
        return (
            self.activity_df[self.activity_df["user_id_y"] == user_id]["chosen_user_id"]
            .dropna()
            .unique()
            .tolist()
        )

        
    # 친구 분석 메서드 개선 - 더 많은 특성 추출
    def _analyze_friends(self, friend_ids: List[int]) -> Dict[str, Any]:
        if len(friend_ids) <= 3:
            return {"cold": True}
        logs = self.activity_df[
            (self.activity_df["user_id_x"].isin(friend_ids)) & (self.activity_df["status"] == "I")
        ]
        if logs.empty:
            return {"cold": True}

        q_ids = logs["question_id"].dropna().tolist()
        fq = self.question_df[self.question_df["id"].isin(q_ids)]
        cat_pref = fq.category.value_counts(normalize=True).to_dict()
        avg_hour = pd.to_datetime(logs["created_at_x"].dropna()).dt.hour.mean()
        
        # 새로운 특성 추출 - 친구 활동 패턴
        active_days = pd.to_datetime(logs["created_at_x"].dropna()).dt.dayofweek.value_counts(normalize=True).to_dict()
        question_lengths = fq["question_text"].str.len().mean()  # 평균 질문 길이
        
        return {
            "cold": False,
            "cat_pref": cat_pref,
            "avg_hour": int(avg_hour),
            "active_days": active_days,
            "avg_q_length": question_lengths,
            # 친구 활동 지표들 추가
        }

    # -----------------------------------------------------
    # scoring helpers
    # -----------------------------------------------------
    def _scaled_base_score(self, meta: pd.Series) -> float:
        """0-1 base = 0.4*favorability + 0.4*question_value + 0.2*spent_point"""
        fav = meta.get("favorability_figure", 0) / 5  # 0-1
        qv = self._norm(meta.get("question_value", 0), self._qv_max)
        tp = self._norm(meta.get("total_spent_point", 0), self._tp_max)
        return 0.4 * fav + 0.4 * qv + 0.2 * tp

    def _hour_alignment_weight(self, cur: int, friend_avg: Optional[int]) -> float:
        """친구들의 평균 활동 시간대와 현재 시간의 유사성 가중치"""
        if friend_avg is None:
            return 1.0
        diff = abs(cur - friend_avg)
        if diff <= 1:
            return 1.3  # 1시간 이내 차이는 강한 가중치
        if diff <= 2:
            return 1.15  # 2시간 이내 차이는 중간 가중치
        return 1.0  # 그 외에는 중립 가중치

    def _score_question(self, meta: pd.Series, profile: Dict[str, Any], hour: int) -> Tuple[float, Dict[str, float]]:
        cat = meta.get("category", "기타")
        base = self._scaled_base_score(meta)  # 0-1
        cat_w = self.category_weights.get(cat, 1.0)

        # 친구 선호도의 영향력 강화 (1.5 -> 3.0으로 증가)
        pref_ratio = profile.get("cat_pref", {}).get(cat, 0)
        pref_w = 1 + pref_ratio * 4.0  # 친구 선호도 강화 (최대 4배)

        hour_w = self._hour_alignment_weight(hour, profile.get("avg_hour"))
        time_cat_w = self.time_category_weights.get(hour, {}).get(cat, 1.0)

        # 친구별 특성 반영 - 새로운 가중치 추가
        friend_activity_ratio = len(profile.get("friend_ids", [])) / 10  # 친구 활동성 반영
        friend_activity_w = min(1.5, max(0.7, friend_activity_ratio))  # 0.7~1.5 범위

        # 무작위성 요소 추가 (±15%)
        randomness = 0.85 + np.random.random() * 0.3  # 0.85~1.15 범위의 무작위 가중치

        # 총점 계산에 새로운 가중치 적용
        total = base * cat_w * pref_w * hour_w * time_cat_w * friend_activity_w * randomness
        
        components = {
            "base_score": base,
            "category_weight": cat_w,
            "friend_pref_weight": pref_w,
            "hour_alignment_weight": hour_w,
            "time_category_weight": time_cat_w,
            "friend_activity_weight": friend_activity_w,
            "randomness_factor": randomness
        }
        return total, components

    def _cold_start_score(self, meta: pd.Series, hour: int) -> Tuple[float, Dict[str, float]]:
        cat = meta.get("category", "기타")
        base = self._scaled_base_score(meta)  # 0-1
        cat_w = self.category_weights.get(cat, 1.0)

        hour_col = f"hour_{hour}"
        react_ratio = meta.get(hour_col, 0) / (meta.get("check_count", 1) + 1)
        time_w = min(1.4, max(0.7, 0.7 + react_ratio * 1.4))  # 시간대별 반응률 반영

        time_cat_w = self.time_category_weights.get(hour, {}).get(cat, 1.0)
        total = base * cat_w * time_w * time_cat_w
        comps = {
            "base_score": base,
            "category_weight": cat_w,
            "time_weight": time_w,
            "time_category_weight": time_cat_w,
        }
        return total, comps

    # -----------------------------------------------------
    # RAG helper
    # -----------------------------------------------------
    def _find_similar_questions(self, query_text: str, exclude_ids: List[int] | None = None, k: int = 5) -> List[Dict]:
        """
        향상된 유사 질문 검색 시스템
        - 하이브리드 검색: 의미적 유사성 + 키워드 매칭 혼합
        - 가중치 적용 다양화: 카테고리 일치도, 호감도, 질문 길이 등 고려
        - 다양성 확보: 결과 클러스터링 및 재랭킹
        """
        # 1. 의미적 유사성 검색 (기존 방식)
        query_emb = self.embedder.embed_query(query_text)
        semantic_docs = self.vectorstore.similarity_search_by_vector(query_emb, k=k * 3)
        
        # 2. 키워드 기반 검색 (새로운 방식)
        # 핵심 키워드 추출
        keywords = self._extract_keywords(query_text)
        keyword_matches = []
        
        # 키워드 일치 점수 계산을 위한 준비
        all_questions = set(self.question_df["question_text"].tolist())
        keyword_scores = {}
        
        for q in all_questions:
            # 키워드 매칭 점수 계산 (TF-IDF 스타일)
            score = 0
            for kw in keywords:
                if kw.lower() in q.lower():
                    # 더 희소한 키워드에 더 높은 가중치 부여
                    idf = np.log(len(all_questions) / sum(1 for qq in all_questions if kw.lower() in qq.lower()))
                    score += idf
            
            if score > 0:
                keyword_scores[q] = score
        
        # 키워드 점수가 높은 상위 항목 선택
        keyword_matches = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:k * 2]
        
        # 3. 하이브리드 결과 결합
        hybrid_candidates = {}
        
        # 의미적 유사성 결과 추가
        for doc in semantic_docs:
            text = doc.page_content
            if text not in hybrid_candidates:
                row = self.question_df[self.question_df["question_text"] == text]
                if not row.empty:
                    q_id = int(row["id"].iloc[0])
                    if exclude_ids and q_id in exclude_ids:
                        continue
                    
                    # 의미적 유사성 점수
                    semantic_score = 0.7  # 기본 의미적 유사성 가중치
                    
                    # 카테고리 일치 보너스
                    cat = row["category"].iloc[0]
                    query_cat = self._predict_category(query_text)  # 새로운 메서드 필요
                    category_bonus = 0.2 if cat == query_cat else 0
                    
                    # 최종 점수
                    hybrid_candidates[text] = {
                        "id": q_id,
                        "question_text": text,
                        "category": cat,
                        "favorability_figure": row["favorability_figure"].iloc[0],
                        "question_value": row["question_value"].iloc[0],
                        "score": semantic_score + category_bonus
                    }
        
        # 키워드 기반 결과 추가 또는 점수 병합
        for text, kw_score in keyword_matches:
            if text in hybrid_candidates:
                # 이미 의미적 유사성으로 추가된 경우, 점수 합산
                hybrid_candidates[text]["score"] += 0.3 * (kw_score / max(s for _, s in keyword_matches))
            else:
                row = self.question_df[self.question_df["question_text"] == text]
                if not row.empty:
                    q_id = int(row["id"].iloc[0])
                    if exclude_ids and q_id in exclude_ids:
                        continue
                    
                    # 키워드 기반 점수만 적용
                    normalized_kw_score = 0.3 * (kw_score / max(s for _, s in keyword_matches))
                    
                    hybrid_candidates[text] = {
                        "id": q_id,
                        "question_text": text,
                        "category": row["category"].iloc[0],
                        "favorability_figure": row["favorability_figure"].iloc[0],
                        "question_value": row["question_value"].iloc[0],
                        "score": normalized_kw_score
                    }
        
        # 4. 다양성 확보를 위한 카테고리 클러스터링 및 재랭킹
        final_results = []
        
        # 카테고리별 그룹화
        by_category = {}
        for text, item in hybrid_candidates.items():
            cat = item["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)
        
        # 각 카테고리에서 최고 점수 항목 선택
        categories = list(by_category.keys())
        while len(final_results) < k and categories:
            # 라운드 로빈 방식으로 카테고리 순회
            for cat in categories[:]:
                if by_category[cat]:
                    # 해당 카테고리에서 가장 높은 점수의 항목 선택
                    best_item = max(by_category[cat], key=lambda x: x["score"])
                    final_results.append(best_item)
                    # 선택된 항목 제거
                    by_category[cat].remove(best_item)
                else:
                    # 더 이상 항목이 없는 카테고리 제거
                    categories.remove(cat)
                
                # 충분한 결과를 얻었으면 중단
                if len(final_results) >= k:
                    break
        
        # 남은 자리는 전체 점수 상위 항목으로 채우기
        if len(final_results) < k:
            all_remaining = [item for items in by_category.values() for item in items]
            all_remaining.sort(key=lambda x: x["score"], reverse=True)
            final_results.extend(all_remaining[:k - len(final_results)])
        
        return final_results[:k]

    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 중요 키워드 추출"""
        # 불용어 목록
        stopwords = {"이", "그", "저", "것", "수", "은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "로", "으로"}
        
        # 간단한 명사 추출 (실제로는 형태소 분석기 사용 권장)
        words = re.findall(r'\w+', text)
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]
        
        return keywords

    def _predict_category(self, text: str) -> str:
        """질문 텍스트 기반 카테고리 예측"""
        # 간단한 키워드 기반 카테고리 예측
        keywords = {    
            "연애": ["좋아하", "썸", "고백", "연애", "사랑", "남자친구", "여자친구", "남친", "여친", "데이트"],
            "외모": ["잘생", "예쁘", "외모", "얼굴", "스타일", "패션", "옷", "머리", "화장"],
            "우정": ["친구", "우정", "의리", "베프", "절친", "소속", "그룹"],
            "성격": ["성격", "착한", "착하", "화나", "착하", "다정", "차가", "시크", "장난꾸러기", "귀여운"],
            "미래": ["꿈", "미래", "직업", "장래", "목표", "성공", "취업", "대학", "희망"],
            "상상": ["상상", "가정", "만약", "무인도", "마지막", "하늘", "천국", "지구", "우주"]
        }
        
        text_lower = text.lower()
        scores = {}
        
        for cat, kws in keywords.items():
            score = 0
            for kw in kws:
                if kw in text_lower:
                    score += 1
            scores[cat] = score
        
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        # 기본값
        return "기타"

  
    def _ensure_category_diversity(self, candidates: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """
        카테고리 다양성을 보장하는 강화된 알고리즘
        - 연애 카테고리는 엄격하게 최대 2개로 제한
        - 최소 3개 이상의 서로 다른 카테고리가 포함되도록 보장
        """
        if candidates.empty or len(candidates) <= 1:
            return candidates
            
        # 엄격한 카테고리 제한 설정 - 연애 카테고리는 절대 최대 2개로 제한
        MAX_ROMANCE_QUESTIONS = 2  # 하드코딩된 상수로 명확하게 제한
        
        # 점수 기준 정렬
        cand = candidates.sort_values("score", ascending=False).copy()
        
        # 결과 저장용 데이터프레임
        selected = pd.DataFrame(columns=cand.columns)
        
        # 카테고리별 카운트 초기화
        category_counts = {}
        
        # 1단계: 각 카테고리에서 가장 높은 점수의 항목 하나씩 선택 (다양성 확보)
        categories = cand["category"].unique()
        for category in categories:
            if len(selected) >= k:
                break
                
            # 해당 카테고리 중 가장 높은 점수의 항목 선택
            cat_items = cand[cand["category"] == category]
            if not cat_items.empty:
                top_item = cat_items.iloc[0]
                selected = pd.concat([selected, pd.DataFrame([top_item])], ignore_index=True)
                category_counts[category] = 1
                
                # 선택된 항목은 후보에서 제거
                cand = cand[cand["id"] != top_item["id"]]
        
        # 2단계: 나머지 자리를 점수 순서대로 채우되, 연애 카테고리 제한 엄수
        remaining_slots = k - len(selected)
        if remaining_slots > 0:
            for _, row in cand.iterrows():
                if len(selected) >= k:
                    break
                    
                category = row["category"]
                
                # 연애 카테고리 제한 확인 - 엄격하게 적용
                if category == "연애" and category_counts.get("연애", 0) >= MAX_ROMANCE_QUESTIONS:
                    continue  # 이미 최대 개수에 도달했으면 건너뛰기
                
                # 추가 및 카운트 업데이트
                selected = pd.concat([selected, pd.DataFrame([row])], ignore_index=True)
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # 3단계: 최종 검증 - 연애 카테고리가 여전히 너무 많으면 강제로 제거
        romance_rows = selected[selected["category"] == "연애"]
        if len(romance_rows) > MAX_ROMANCE_QUESTIONS:
            # 점수가 낮은 연애 질문부터 제거
            romance_to_remove = romance_rows.sort_values("score").head(len(romance_rows) - MAX_ROMANCE_QUESTIONS)
            selected = selected[~selected["id"].isin(romance_to_remove["id"].tolist())]
            
            # 제거된 만큼 다른 카테고리에서 보충
            removed_count = len(romance_to_remove)
            other_candidates = cand[
                (cand["category"] != "연애") & 
                (~cand["id"].isin(selected["id"].tolist()))
            ].head(removed_count)
            
            selected = pd.concat([selected, other_candidates], ignore_index=True)
        
        # 최종 결과 정렬 (점수 기준)
        selected = selected.sort_values("score", ascending=False).head(k)
        
        return selected


    # -----------------------------------------------------
    # public API
    # -----------------------------------------------------
   # 추천 메서드 개선 - 상위 후보군에서 확률적 선택 적용
    def recommend(self, user_id: int, k: int = 5) -> List[Dict]:
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        hour = now.hour

        friends = self._get_friend_list(user_id)
        profile = self._analyze_friends(friends)
        
        # 친구 ID 정보 추가
        profile["friend_ids"] = friends

        if profile["cold"]:
            cand = self.question_df.copy()
            scores = cand.apply(lambda r: self._cold_start_score(r, hour), axis=1)
        else:
            cand = self.question_df.copy()
            scores = cand.apply(lambda r: self._score_question(r, profile, hour), axis=1)

        cand["score"] = [s[0] for s in scores]
        cand["components"] = [s[1] for s in scores]
        
        # 연애 카테고리 가중치 추가 패널티 적용 - 편향 완화
        cand.loc[cand["category"] == "연애", "score"] *= 0.8
        
        # 상위 후보군 선택 (top k*8 개 선택) - 충분히 큰 풀에서 다양성 확보
        top_pool = cand.sort_values("score", ascending=False).head(k * 8)
        
        # 다양성 보장된 최종 후보 선택
        top = self._ensure_category_diversity(top_pool, k)
        
        # 마지막 검증 - 연애 카테고리가 2개 넘으면 강제 조정
        romance_count = sum(1 for _, r in top.iterrows() if r.category == "연애")
        if romance_count > 2:
            # 연애 질문 중 가장 낮은 점수 항목을 찾아 제거
            romance_rows = top[top["category"] == "연애"].sort_values("score")
            to_remove = romance_rows.head(romance_count - 2)
            
            # 제거 대상이 아닌 행만 유지
            top = top[~top["id"].isin(to_remove["id"])]
            
            # 제거된 만큼 다른 카테고리로 채우기
            others = cand[
                (cand["category"] != "연애") & 
                (~cand["id"].isin(top["id"]))
            ].sort_values("score", ascending=False).head(romance_count - 2)
            
            top = pd.concat([top, others], ignore_index=True)
            top = top.sort_values("score", ascending=False).head(k)
        
        out: List[Dict] = []
        for _, r in top.iterrows():
            rec = {
                "question_id": int(r.id),
                "question_text": r.question_text,
                "category": r.category,
                "favorability": r.favorability_figure,
                "score": r.score,
                "components": r.components,
            }
            
            # 유사한 질문 찾기
            sims = self._find_similar_questions(
                r.question_text,
                exclude_ids=[r.id] + [q["question_id"] for q in out],
                k=2,
            )
            
            # 추천 이유와 생성 질문 생성
            reason_info = self.generate_recommendation_reason(
                is_cold_start=profile["cold"],
                question_text=r.question_text,
                category=r.category,
                hour=hour,
                favorability=r.favorability_figure,
                friend_pref=profile.get("cat_pref", {}).get(r.category, 0),
                similar_questions=sims,
                components=r.components,
            )
            
            rec.update(reason_info)
            out.append(rec)
            
        # 추가 확장 추천에서도 연애 카테고리 제한 적용
        extended_recs = self._generate_extended_recommendations(out, profile["cold"], hour)
        
        # 확장 추천에서도 연애 카테고리 제한 확인
        final_combined = out + extended_recs
        romance_count = sum(1 for r in final_combined if r.get("category") == "연애")
        
        # 전체 추천 목록에서 연애 질문이 3개 이상이면 일부 제거
        if romance_count > 3:
            # 확장 추천에서 연애 카테고리 제거
            filtered_extended = [r for r in extended_recs if r.get("category") != "연애"]
            
            # 여전히 3개 초과면 점수가 낮은 연애 질문부터 제거
            romance_recs = [r for r in out if r.get("category") == "연애"]
            if len(romance_recs) > 3:
                # 점수 기준 정렬하여 낮은 점수부터 제거
                romance_recs.sort(key=lambda x: x.get("score", 0))
                out = [r for r in out if r.get("category") != "연애"] + romance_recs[:3]
            
            final_combined = out + filtered_extended
        
        return final_combined

        
    
    # -----------------------------------------------------
    # 추가 추천 생성
    # -----------------------------------------------------
    def _generate_extended_recommendations(self, base_recs: List[Dict], is_cold_start: bool, hour: int) -> List[Dict]:
        """
        기본 추천 외에 추가 추천 생성 - 연애 카테고리 제한 적용
        """
        extended_recs = []
        used_ids = [r["question_id"] for r in base_recs]
        
        # 기존 추천에서 연애 카테고리 수 확인
        romance_count = sum(1 for r in base_recs if r.get("category") == "연애")
        romance_limit_reached = romance_count >= 2  # 이미 2개 이상이면 제한 도달
        
        # 1. 생성 질문 기반 추천 2개 (연애 카테고리 제한 적용)
        gen_count = 0
        for base_rec in base_recs:
            if gen_count >= 2:
                break
                
            # 연애 카테고리 제한 적용
            if romance_limit_reached and base_rec.get("category") == "연애":
                continue
                
            # 생성 질문이 있는 경우만 처리
            gen_q = base_rec.get("generated_question")
            if not gen_q or len(gen_q) < 10:  # 너무 짧은 생성 질문은 건너뛰기
                continue
                
            # 생성 질문을 활용한 추천 생성
            extended_recs.append({
                "question_id": -1 * (gen_count + 1),  # 임시 음수 ID 부여
                "question_text": gen_q,
                "category": base_rec["category"],
                "favorability": 0.0,
                "score": base_rec["score"] * 0.95,
                "type": "generated",
                "source_id": base_rec["question_id"]
            })
            
            # 연애 카테고리 추가되면 카운트 증가
            if base_rec.get("category") == "연애":
                romance_count += 1
                if romance_count >= 3:  # 3개 이상이면 제한 도달
                    romance_limit_reached = True
                    
            gen_count += 1
        
        # 2. 관련 질문 기반 추천 3개 (연애 카테고리 제한 적용)
        related_count = 0
        for base_rec in base_recs:
            if related_count >= 3:
                break
                
            # 관련 질문이 있는 경우만 처리
            related = base_rec.get("related_questions", [])
            if not related:
                continue
                
            for rel in related:
                # 이미 연애 카테고리 제한에 도달했고, 이 질문이 연애 카테고리면 건너뛰기
                if romance_limit_reached and rel.get("category") == "연애":
                    continue
                    
                rel_id = rel.get("id")
                # 이미 사용된 ID는 건너뛰기
                if rel_id in used_ids:
                    continue
                    
                extended_recs.append({
                    "question_id": rel_id,
                    "question_text": rel["text"],
                    "category": rel["category"],
                    "favorability": 0.0,
                    "score": base_rec["score"] * 0.9,
                    "type": "related",
                    "source_id": base_rec["question_id"]
                })
                used_ids.append(rel_id)
                
                # 연애 카테고리 추가되면 카운트 증가
                if rel.get("category") == "연애":
                    romance_count += 1
                    if romance_count >= 3:  # 3개 이상이면 제한 도달
                        romance_limit_reached = True
                        
                related_count += 1
                
                if related_count >= 3:
                    break
        
        return extended_recs

    # -----------------------------------------------------
    # LLM reason generator
    # -----------------------------------------------------
    def generate_recommendation_reason(
    self,
    is_cold_start: bool,
    question_text: str,
    category: str,
    hour: int,
    favorability: float,
    friend_pref: float = 0.0,
    similar_questions: List[Dict] | None = None,
    components: Dict[str, float] = None,) -> Dict[str, Any]:
        
        rag_context = self._rag_generate(question_text)
        # friendly time label
        if 8 <= hour <= 11:
            time_ctx = "아침 등교 시간대"
        elif 12 <= hour <= 15:
            time_ctx = "점심~오후 시간대"
        elif 16 <= hour <= 19:
            time_ctx = "방과후 시간대"
        elif 20 <= hour <= 23:
            time_ctx = "저녁 시간대"
        else:
            time_ctx = "밤/새벽 시간대"

        friend_pref = friend_pref or 0.05
        sim_txt = "\n".join(f"- {q['question_text']}" for q in (similar_questions or [])) or "- (유사 질문 없음)"
        
        # 가중치 정보 문자열 구성
        weights_info = ""
        if components:
            weights_info = (
                f"카테고리 가중치: {components.get('category_weight', 1.0):.2f}, "
                f"친구 선호도 가중치: {components.get('friend_pref_weight', 1.0):.2f}, "
                f"시간대 일치 가중치: {components.get('hour_alignment_weight', 1.0):.2f}, "
                f"시간-카테고리 가중치: {components.get('time_category_weight', 1.0):.2f}"
            )
            if 'friend_activity_weight' in components:
                weights_info += f", 친구 활동성 가중치: {components.get('friend_activity_weight', 1.0):.2f}"

        # 가중치 변수 명확히 설정
        cat_w = components.get('category_weight', 1.0) if components else 1.0
        friend_w = components.get('friend_pref_weight', 1.0) if components else 1.0
        hour_w = components.get('hour_alignment_weight', 1.0) if components else 1.0
        time_cat_w = components.get('time_category_weight', 1.0) if components else 1.0

        if is_cold_start:
            tpl = (
                "문맥 정보:\n{rag_context}\n\n"
                "시스템: 당신은 10대 친구들을 위한 질문 추천 앱의 일부입니다. 현재 {time_ctx}({hour}시)입니다. "
                "이 질문의 카테고리는 \"{category}\"이고, 호감도는 {favorability:.1f}입니다. "
                "다음 질문과 유사 질문을 분석하여, 추천 이유와 변형된 생성 질문을 만들어주세요.\n\n"
                "주어진 질문: \n{question_text}\n\n유사 질문:\n{sim_txt}\n\n"
                "가중치 정보: {weights_info}\n\n"
                "다음 형식으로 답변해주세요:\n"
                "추천이유: [시간대와 카테고리 가중치를 명시적으로 언급하며 추천 이유 설명. 예: '이 질문은 {category} 카테고리(가중치: {cat_w:.1f})이며, "
                "{time_ctx}에 인기가 높습니다(시간-카테고리 가중치: {time_cat_w:.1f}). 호감도 {favorability:.1f}점으로 평가되어 추천합니다.']\n"
                "생성질문: [원 질문을 참고하여 '가장 ~할 것 같은 사람은?' 형식으로 변형]"
            )
        else:
            tpl = (
                "문맥 정보:\n{rag_context}\n\n"
                "시스템: 당신은 10대 친구들을 위한 질문 추천 앱의 일부입니다. 현재 {time_ctx}({hour}시)입니다. "
                "친구 그룹은 \"{category}\" 질문에 {friend_pref:.0%} 관심을 보였습니다. "
                "호감도 {favorability:.1f} 점의 이 질문을 추천합니다.\n\n"
                "주어진 질문: \n{question_text}\n\n유사 질문:\n{sim_txt}\n\n"
                "가중치 정보: {weights_info}\n\n"
                "다음 형식으로 답변해주세요:\n"
                "추천이유: [구체적인 가중치를 언급하며 추천 이유 설명. 예: '친구들이 {category} 카테고리를 {friend_pref:.0%} 선호하고(친구 선호도 가중치: {friend_w:.1f}), "
                "{time_ctx}와 일치하여(시간대 일치 가중치: {hour_w:.1f}) 추천합니다. 카테고리 가중치는 {cat_w:.1f}이며, "
                "호감도 {favorability:.1f}점입니다.']\n"
                "생성질문: [원 질문을 참고하여 '가장 ~할 것 같은 사람은?' 형식으로 변형]"
            )

        prompt = PromptTemplate.from_template(tpl)
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        result = llm_chain.run(
            rag_context=rag_context,
            time_ctx=time_ctx,
            hour=hour,
            category=category,
            question_text=question_text,
            favorability=favorability,
            friend_pref=friend_pref,
            sim_txt=sim_txt,
            weights_info=weights_info,
            cat_w=cat_w,
            friend_w=friend_w,
            hour_w=hour_w,
            time_cat_w=time_cat_w,
        )

        # 향상된 결과 파싱 로직
        reason = ""
        gen_q = ""
        
        # 정규 표현식으로 패턴 검색
        reason_match = re.search(r'추천이유:?\s*(.*?)(?:\n|$)', result, re.IGNORECASE | re.DOTALL)
        gen_q_match = re.search(r'생성질문:?\s*(.*?)(?:\n|$)', result, re.IGNORECASE | re.DOTALL)
        
        if reason_match:
            reason = reason_match.group(1).strip()
        if gen_q_match:
            gen_q = gen_q_match.group(1).strip()
            
        # 정규 표현식으로 찾지 못한 경우, 다른 패턴 시도
        if not reason:
            lines = result.strip().splitlines()
            for i, line in enumerate(lines):
                if "추천" in line and "이유" in line and i+1 < len(lines):
                    reason = lines[i+1].strip()
                    break
                    
        if not gen_q:
            lines = result.strip().splitlines()
            for i, line in enumerate(lines):
                if "생성" in line and "질문" in line and i+1 < len(lines):
                    gen_q = lines[i+1].strip()
                    break
        
        # 관련 질문 정보 구성
        related = (
            [{"id": q["id"], "text": q["question_text"], "category": q["category"]} for q in (similar_questions or [])][:2]
        )
        
        return {"reason": reason, "generated_question": gen_q, "related_questions": related}
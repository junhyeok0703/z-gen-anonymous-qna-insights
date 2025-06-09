import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from folium.plugins import HeatMap, MarkerCluster
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# 추천기 임포트
from recommender import FriendBasedRecommender


plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우

# 페이지 기본 설정
st.set_page_config(page_title="2팀: SNOOPY 대시보드", layout="wide")

# 대시보드 데이터 불러오기
@st.cache_data
def load_main_data():
    monthly_paid_user_df = pd.read_csv("/Users/parkjunhyeok/고급프젝_10대SNS분석/대시보드/dash_monthly_paid_user_df.csv")
    accounts_user_df = pd.read_csv("/Users/parkjunhyeok/고급프젝_10대SNS분석/대시보드/dash_accounts_user_df.csv")
    tm_hackle_events_df = pd.read_csv("/Users/parkjunhyeok/고급프젝_10대SNS분석/대시보드/dash_hackle_events_df.csv")
    return monthly_paid_user_df, accounts_user_df, tm_hackle_events_df

monthly_paid_user_df, accounts_user_df, tm_hackle_events_df = load_main_data()

# 사이드바 메뉴
st.sidebar.title("📂 메뉴")
menu = st.sidebar.radio("이동할 메뉴를 선택하세요", ["📊 운영 대시보드", "🗺️ 지도로 보는 학교 통계", "🤖 결제 예측 모델", "👬 친구기반 질문 추천시스템"])

# 메뉴 1: 운영 대시보드
if menu == "📊 운영 대시보드":

    # KPI 카드 (이번 달 기준)
    latest_row = monthly_paid_user_df.iloc[-1]
    month = latest_row["year_month"]
    count = latest_row["paid_user_count"]
    target = latest_row["target"]
    achieved = latest_row["achieved"]
    rate = latest_row["달성률(%)"]

    st.title(f"📅 이번달 ({month}) KPI 요약")
    st.subheader("이번 달 목표 성과")

    # 전체 영역 분할 (왼쪽: 세로 지표, 오른쪽: 도넛)
    left, right = st.columns([1, 2])

    with left:
        st.metric("결제 유저 수", f"{int(count):,}명")
        st.metric("목표 유저 수", f"{int(target):,}명")
        st.metric("달성 여부", "✔️" if achieved else "❌")

    with right:
        import plotly.graph_objects as go

        fig = go.Figure(data=[
            go.Pie(
                labels=['달성률', '미달성'],
                values=[rate, 100 - rate],
                hole=0.5,
                marker_colors=["#4CAF50", "#E0E0E0"] if achieved else ["#FF6B6B", "#E0E0E0"],
                textinfo='label+percent'
            )
        ])
        fig.update_layout(
            annotations=[dict(text=f"{rate:.1f}%", x=0.5, y=0.5, font_size=16, showarrow=False)],
            showlegend=False,
            margin=dict(t=30, b=10),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()




    # 월별 결제 유저 수 vs 목표 (월 선택 가능)
    st.title("📊 월별 결제 유저 수 및 달성률")

    # 월 선택 (기본으로 2023-11 보이게)
    month_list = sorted(monthly_paid_user_df['year_month'].unique())
    default_index = month_list.index("2023-11") if "2023-11" in month_list else 0
    selected_month = st.selectbox("월을 선택하세요", month_list, index=default_index)


    # 선택된 월 및 이전 달 데이터 추출
    selected_idx = monthly_paid_user_df[monthly_paid_user_df["year_month"] == selected_month].index[0]
    prev_idx = selected_idx - 1 if selected_idx > 0 else None

    # 선택된 행
    row = monthly_paid_user_df.iloc[selected_idx]
    paid_user_count = int(row["paid_user_count"])
    target_user_count = int(row["target"])
    rate = row["달성률(%)"]
    achieved = row["achieved"]

    # 수치 요약 (두 개의 col)
    col1, col2 = st.columns(2)

    # 결제 유저 수에 delta 추가
    if prev_idx is not None:
        prev_value = int(monthly_paid_user_df.iloc[prev_idx]["paid_user_count"])
        diff = paid_user_count - prev_value
        delta = f"{diff:+,}명"
    else:
        delta = None  # 이전 달 없으면 표시 안 함

    # metric 표시
    col1.metric("### 결제 유저 수", f"{paid_user_count:,}명", delta)
    col2.metric("### 목표 유저 수", f"{target_user_count:,}명")

    # 두 개의 그래프 (좌: 추이, 우: 파이차트)
    left_col, right_col = st.columns(2)

    # 왼쪽: 결제 유저 수 추이
    with left_col:
        st.markdown("#### 결제 유저 수 추이")
        compare_df = monthly_paid_user_df.loc[[i for i in [prev_idx, selected_idx] if i is not None], ["year_month", "paid_user_count"]]
        compare_df = compare_df.set_index("year_month")
        st.line_chart(compare_df)

    # 오른쪽: 파이 차트
    with right_col:
        st.markdown("#### 달성률")
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Pie(
            labels=['결제 유저', '미달성'],
            values=[rate, 100 - rate],
            marker_colors=["#4CAF50", "#E0E0E0"] if achieved else ["#FF6B6B", "#E0E0E0"],
            textinfo='label+percent'
        )])
        fig.update_layout(
            width=350, height=350,
            margin=dict(t=30, b=10)
        )
        st.plotly_chart(fig)


    # 누적 결제 유저 수 추이
    st.subheader("📈 누적 결제 유저 수 추이")

    # 누적 값 컬럼 추가 (캐시 적용)
    @st.cache_data
    def compute_cumsum(df):
        df = df.copy()
        df["누적 결제 유저 수"] = df["paid_user_count"].cumsum()
        return df

    monthly_paid_user_df = compute_cumsum(monthly_paid_user_df)

    # 그래프 출력
    st.line_chart(monthly_paid_user_df.set_index("year_month")[["누적 결제 유저 수"]])

    st.divider()

    # 기능 사용량
    st.title("🔧 기능 사용량")
    selected_month = st.selectbox("월 선택", sorted(tm_hackle_events_df['year_month'].unique()))
    selected_event_df = tm_hackle_events_df[tm_hackle_events_df['year_month'] == selected_month]

    # 이벤트 수 집계 후 명확한 컬럼명 지정
    event_counts = selected_event_df["event_key"].value_counts().head(10).reset_index()
    event_counts.columns = ["event_key", "count"]

    event_counts = event_counts.sort_values(by="count", ascending=False)
    st.bar_chart(event_counts.set_index("event_key"))


# 메뉴 2: 지도로 보는 학교 통계
elif menu == "🗺️ 지도로 보는 학교 통계":
    st.title("🗺️ 지도로 보는 학교 통계")

    @st.cache_data
    def load_data():
        gdf = gpd.read_file("data/schools.geojson")
        gdf["student_count"] = gdf["student_count"].fillna(0)
        gdf["시도"] = gdf["address"].str.extract(r"(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)")
        return gdf

    gdf = load_data()

    # 필터
    school_type_map = {"전체": None, "고등학교": "H", "중학교": "M"}
    col1, col2 = st.columns(2)
    with col1:
        selected_region = st.selectbox("시도 선택", ["전체"] + sorted(gdf["시도"].dropna().unique()))
    with col2:
        selected_type = st.selectbox("학교급 선택", ["전체", "고등학교", "중학교"])

    filtered_df = gdf.copy()
    if selected_region != "전체":
        filtered_df = filtered_df[filtered_df["시도"] == selected_region]
    if school_type_map[selected_type]:
        filtered_df = filtered_df[filtered_df["school_type"] == school_type_map[selected_type]]

    # 검색창
    school_options = gdf["학교명"].dropna().unique().tolist()
    school_choice = st.selectbox("학교명 자동완성", [""] + sorted(school_options))
    manual_input = st.text_input("또는 주소 등 직접 검색어 입력")
    search_button = st.button("검색")

    if "search_triggered" not in st.session_state:
        st.session_state["search_triggered"] = False

    if search_button:
        query = manual_input.strip() if manual_input else school_choice.strip()
        if query == "":
            st.session_state["search_triggered"] = False
            st.session_state["query"] = ""
        else:
            st.session_state["search_triggered"] = True
            st.session_state["query"] = query


    if st.session_state["search_triggered"] and st.session_state.get("query", ""):
        final_query = st.session_state["query"]
        matched = filtered_df[
            filtered_df["학교명"].astype(str).str.contains(final_query, case=False, na=False) |
            filtered_df["소재지도로명주소"].astype(str).str.contains(final_query, case=False, na=False)
        ]
        if not matched.empty:
            center = [matched.iloc[0]["위도"], matched.iloc[0]["경도"]]
        else:
            center = [filtered_df["위도"].mean(), filtered_df["경도"].mean()]
            st.warning("검색 결과가 없습니다.")
    else:
        matched = pd.DataFrame()
        center = [filtered_df["위도"].mean(), filtered_df["경도"].mean()]

    # 지도 생성
    m = folium.Map(
        location=center,
        zoom_start=13 if not matched.empty else 11,
        tiles="http://xdworld.vworld.kr:8080/2d/Base/service/{z}/{x}/{y}.png",
        attr="VWorld base map"
    )
    # 결과 메시지
    if st.session_state["search_triggered"] and final_query:
        st.success(f"🔍 검색 결과: {len(matched):,}개 학교가 검색되었습니다.")
    elif selected_region != "전체" or selected_type != "전체":
        st.success(f"🔍 필터 결과: {len(filtered_df):,}개 학교가 검색되었습니다.")
    else:
        st.success(f"전체 {len(gdf):,}개 학교가 로드되었습니다.")

    # 히트맵
    heat_data = [
        [row["위도"], row["경도"], row["student_count"]]
        for _, row in filtered_df.iterrows()
        if pd.notna(row["위도"]) and pd.notna(row["경도"])
    ]
    HeatMap(
        heat_data,
        radius=15,
        blur=10,
        max_zoom=11,
        min_opacity=0.3,
        max_val=max(filtered_df["student_count"]),
        gradient={
            "0.2": 'blue',
            "0.4": 'lime',
            "0.6": 'orange',
            "1.0": 'red'
        }
    ).add_to(m)

    # 마커 클러스터
    cluster = MarkerCluster().add_to(m)

    # 마커 전부 다 추가
    for _, row in filtered_df.iterrows():
        student_display = f"{int(row['student_count'])}명" if row["student_count"] else "정보 없음"
        radius = max(4, min(12, row["student_count"] / 20))
        popup_text = f"""
        <b>{row["학교명"]}</b> (유저 수: {student_display})<br>
        <i>{row["소재지도로명주소"]}</i>
        """
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=radius,
            popup=folium.Popup(popup_text, max_width=300),
            color="blue" if row["school_type"] == "H" else "green",
            fill=True,
            fill_opacity=0.25,
            weight=0.5
        ).add_to(cluster)

    # 검색 결과 핀 (여러 개 모두 표시)
    if not matched.empty:
        for _, row in matched.iterrows():
            popup = folium.Popup(
                html=f"<b>{row['학교명']}</b><br>{row['소재지도로명주소']}",
                max_width=300
            )
            folium.Marker(
                location=[row["위도"], row["경도"]],
                popup=popup,
                icon=folium.Icon(color="red", icon="glyphicon-map-marker")
            ).add_to(m)


    # 렌더링 최적화
    st.markdown("🔹 파란/초록 마커는 학교 개별 정보, 배경 색상은 학생 밀집도를 나타냅니다.")
    st_folium(m, width=1200, height=700, returned_objects=[])


# 메뉴 3: 결제 예측 모델
elif menu == "🤖 결제 예측 모델":
    st.title("🤖 유저 행동 기반 결제 확률 예측")
    st.markdown("🔍 유저의 행동 시퀀스를 입력하면, 트랜스포머 모델이 결제 확률을 예측합니다.")

    # Transformer 클래스 정의
    class PurchaseTransformer(nn.Module):
        def __init__(self, vocab_size=44, time_size=34298, d_model=64, nhead=4, num_layers=2):
            super().__init__()
            self.event_embed = nn.Embedding(vocab_size, d_model)
            self.time_embed = nn.Embedding(time_size, d_model)

            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.classifier = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )

        def forward(self, event_seq, time_seq):
            x = self.event_embed(event_seq) + self.time_embed(time_seq)
            x = x.permute(1, 0, 2)
            x = self.encoder(x)
            x = x.mean(dim=0)
            return self.classifier(x).squeeze()
        
    @st.cache_resource
    def load_model():
        model = PurchaseTransformer(vocab_size=44, time_size=34298)
        state_dict = torch.load("purchase_transformer_best.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    model = load_model()

    # 이벤트 선택용 딕셔너리
    event_choices = [
        (0, "$session_end"), (1, "$session_start"), (3, "click_appbar_alarm_center"),
        (5, "click_appbar_friend_plus"), (7, "click_attendance"),
        (12, "click_bottom_navigation_timeline"), (21, "click_purchase"),
        (25, "click_question_start"), (31, "complete_question"),
        (41, "view_shop"), (43, "view_timeline_tap")
    ]
    event_dict = {label: eid for eid, label in event_choices}

    # 1. 행동 시퀀스 선택
    st.markdown("### 유저의 행동 시퀀스를 선택하세요 (최대 10개)")
    selected_events = st.multiselect("행동 선택 (순서대로)", options=list(event_dict.keys()), default=[], max_selections=10)
    event_seq = [event_dict[label] for label in selected_events]

    # 2. 시간 간격 생성 (자동으로 0, 1, 2, ...)
    time_diff_seq = list(range(len(event_seq)))

    # 3. 패딩 (10개 미만이면 0으로 채움)
    while len(event_seq) < 10:
        event_seq.append(0)
    while len(time_diff_seq) < 10:
        time_diff_seq.append(0)

    # 4. 텐서 변환
    event_tensor = torch.tensor([event_seq])
    time_tensor = torch.tensor([time_diff_seq])

    # 5. 예측 실행
    if st.button("🚀 결제 확률 예측"):
        with torch.no_grad():
            prob = model(event_tensor, time_tensor).item()
            is_paid_pred = int(prob > 0.45)

        st.subheader("예측 결과")
        if is_paid_pred:
            st.success(f"✅ 해당 유저는 결제 가능성이 높습니다! (확률: {prob:.2%})")
        else:
            st.warning(f"⚠️ 해당 유저는 결제 가능성이 낮습니다. (확률: {prob:.2%})")


# 메뉴 4: 친구기반 질문 추천시스템
elif menu == "👬 친구기반 질문 추천시스템":
    st.title("👬 친구 기반 질문 추천기")
    st.markdown("🔍 사용자의 ID를 입력하면(접속을 가정), 친구들의 행동 데이터를 분석하여 맞춤형 질문을 추천합니다.")

    # Sidebar: configuration
    # 메뉴 아래에 구분선 추가
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Setting")
    VECTOR_DIR = st.sidebar.text_input("FAISS Index Directory", "data/faiss_index")
    Q_CSV = st.sidebar.text_input("Questions CSV Path", "data/question_csv_df.csv")
    A_CSV = st.sidebar.text_input("Activities CSV Path", "data/merged_df_a.csv")
    GEMINI_KEY = st.sidebar.text_input("Gemini API Key", type="password")
    st.divider()
    st.header("👤 사용자 입력")
    user_id = st.number_input("사용자 ID", min_value=1, step=1)
    generate = st.button("🚀 추천 생성")

    # FAISS 파일 검사
    faiss_path = Path(VECTOR_DIR)
    index_file = faiss_path / 'index.faiss'

    if generate:
        if not faiss_path.exists() or not index_file.exists():
            st.error(f"❌ FAISS 인덱스 파일을 찾을 수 없습니다:\n  경로: {index_file}")
        else:
            with st.spinner("🎯 추천을 생성하는 중... 잠시만 기다려주세요..."):
                try:
                    recommender = FriendBasedRecommender(VECTOR_DIR, Q_CSV, A_CSV, GEMINI_KEY)
                    recs = recommender.recommend(int(user_id))
                except Exception as e:
                    st.error(f"⚠️ 추천기 로드 실패: {e}")
                    recs = []

            if recs:
                cols = st.columns(2)
                for idx, r in enumerate(recs):
                    col = cols[idx % 2]
                    with col:
                        # 카드 레이아웃
                        st.markdown(
                            """
                            <div style='padding:16px; margin-bottom:16px; border-radius:12px; background:#f0f4f8; box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
                            """, unsafe_allow_html=True)
                        st.markdown(f"#### {idx+1}. {r['question_text']}")

                        # 생성된 질문 vs 관련 질문 vs 기본 추천
                        if r.get('type') == 'generated':
                            st.write(f"**카테고리**: {r['category']}")
                        elif r.get('type') == 'related':
                            st.write(f"**ID**: {r['question_id']}   |   **카테고리**: {r['category']}")
                        else:
                            st.write(f"**ID**: {r['question_id']}   |   **카테고리**: {r['category']}   |   **점수**: {r['score']:.4f}")
                            if r.get('reason'):
                                with st.expander("🔍 추천 이유 보기"):
                                    st.write(r['reason'])
                            if r.get('generated_question'):
                                with st.expander("✏️ 생성된 질문 확인"):
                                    st.write(r['generated_question'])
                            if r.get('related_questions'):
                                with st.expander("🔗 관련 질문 목록"):
                                    for rel in r['related_questions']:
                                        st.write(f"- {rel['text']}   (ID: {rel['id']}, 카테고리: {rel.get('category','-')})")

                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("추천 결과가 없습니다. 입력값을 다시 확인해주세요.")
    else:
        st.info("사이드바에서 설정을 완료한 뒤 ‘🚀 추천 생성’ 버튼을 눌러주세요.")

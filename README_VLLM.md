# Module Analyzer with Local LLM (vLLM + Llama)

Python 코드베이스를 **로컬 LLM**을 사용하여 자동으로 분석하고 문서화하는 도구

## 🌟 주요 특징

### 🤖 로컬 LLM 통합
- **vLLM** 기반 고속 추론 서버
- **Llama 3.2** 모델 지원 (3B, 8B 등)
- **완전한 프라이버시**: 코드가 외부로 전송되지 않음
- **무료**: API 비용 없음

### ✨ 핵심 기능
- 자동 코드 구조 분석 (클래스, 메서드, 함수)
- LLM 기반 모듈 역할 및 책임 분석
- 복잡도 계산 및 의존성 분석
- 지능형 캐싱 (변경된 파일만 재분석)
- 병렬 처리로 빠른 분석 속도

## 📦 설치

### 1. 기본 요구사항
```bash
# Python 3.11+
python --version

# httpx 설치 (HTTP 클라이언트)
pip install httpx
```

### 2. vLLM 설치 (옵션 A: pip)
```bash
# CUDA 12.1+ 환경
pip install vllm

# 또는 특정 CUDA 버전
pip install vllm-nccl-cu12
```

### 3. vLLM 설치 (옵션 B: Docker)
```bash
# vLLM Docker 이미지 사용
docker pull vllm/vllm-openai:latest
```

## 🚀 사용법

### Step 1: vLLM 서버 시작

#### 옵션 A: 직접 실행
```bash
# Llama 3.2 3B 모델 (권장 - 빠르고 메모리 적음)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --port 8000 \
    --max-model-len 4096

# Llama 3.2 8B 모델 (더 정확하지만 느림)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-8B-Instruct \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

#### 옵션 B: Docker 실행
```bash
docker run --gpus all \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --port 8000
```

**서버가 준비되면** `http://localhost:8000/health` 가 200을 반환합니다.

### Step 2: 코드 분석 실행

```bash
# 기본 사용 (현재 디렉토리)
python module_analyzer_vllm.py .

# 특정 프로젝트 분석
python module_analyzer_vllm.py /path/to/project

# 특정 디렉토리만 분석
python module_analyzer_vllm.py /path/to/project --target src

# 다른 포트의 vLLM 서버 사용
python module_analyzer_vllm.py . --llm-url http://localhost:9000

# AI 없이 구조만 분석
python module_analyzer_vllm.py . --no-ai

# 캐시 비활성화
python module_analyzer_vllm.py . --no-cache
```

## 📊 실행 예시

```bash
$ python module_analyzer_vllm.py .

🤖 LLM: meta-llama/Llama-3.2-3B-Instruct
🔗 서버: http://localhost:8000
📁 프로젝트: /home/user/myproject
📄 출력: /home/user/myproject/docs/architecture
💾 캐시: 활성화

🔌 LLM 서버 연결 테스트...
   ✓ 연결 성공

🔍 Step 1: Python 파일 스캔...
   발견: 25개 파일

🔗 Step 2: 모듈 구조 분석...
   [1/25] myapp/core/engine.py 🤖
   [2/25] myapp/api/views.py 🤖
   [3/25] myapp/utils/helpers.py 💾  (캐시)
   ...

📊 Step 3: 의존성 관계 분석...

📝 Step 4: 문서 생성...
   ✓ index.json
   ✓ 25개 모듈 문서
   ✓ dependencies.json

✅ 완료!
   📊 분석: 25개 모듈
   ⏱️  소요: 18.3초
   🤖 AI: 활성화
   📂 출력: docs/architecture

💡 확인: docs/architecture/index.json
```

## 🎯 LLM 분석 결과 예시

### AI가 생성한 모듈 설명
```json
{
  "module_id": "myapp.core.engine",
  "summary": {
    "one_liner": "핵심 데이터 처리 엔진 구현",
    "description": "다양한 데이터 소스로부터 데이터를 수집하고 변환하는 핵심 엔진. 비동기 처리와 에러 핸들링 포함.",
    "responsibilities": [
      "데이터 수집 및 검증",
      "비동기 변환 파이프라인 실행",
      "에러 처리 및 로깅"
    ],
    "key_patterns": ["Factory Pattern", "Async/Await", "Error Handling"],
    "complexity_level": "high",
    "importance": "critical"
  },
  "metrics": {
    "total_lines": 450,
    "num_classes": 3,
    "num_methods": 25,
    "total_complexity": 78
  }
}
```

## 📈 성능 비교

| 모델 | 속도 | 메모리 | 품질 | 권장 용도 |
|------|------|--------|------|----------|
| Llama 3.2 3B | ⚡⚡⚡ | 6GB | ⭐⭐⭐ | 빠른 분석, 대형 프로젝트 |
| Llama 3.2 8B | ⚡⚡ | 16GB | ⭐⭐⭐⭐ | 정확한 분석, 중소 프로젝트 |

**벤치마크** (50개 파일 프로젝트):
- **3B 모델**: ~20초 (약 0.4초/파일)
- **8B 모델**: ~35초 (약 0.7초/파일)
- **캐시 활성화 시**: 90% 이상 빠름

## 🔧 고급 설정

### vLLM 서버 최적화
```bash
# GPU 메모리 최적화
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --enforce-eager

# 멀티 GPU 사용
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-8B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000
```

### Config 커스터마이징
```python
# module_analyzer_vllm.py 내부
class Config:
    CODE_PREVIEW_LENGTH = 1500  # LLM에 보낼 코드 길이
    LLM_RETRY_COUNT = 3         # 재시도 횟수
    LLM_TIMEOUT = 60.0          # 타임아웃 (초)
    LLM_MAX_TOKENS = 600        # 최대 토큰
```

## 🐛 문제 해결

### 1. LLM 서버 연결 실패
```bash
# 서버 상태 확인
curl http://localhost:8000/health

# 방화벽 확인
telnet localhost 8000
```

### 2. JSON 파싱 오류
- **원인**: LLM이 잘못된 형식으로 응답
- **해결**: 재시도 로직이 자동으로 처리 (3회 시도)
- **대안**: `--no-ai` 플래그로 구조만 분석

### 3. CUDA OOM (Out of Memory)
```bash
# 더 작은 모델 사용
--model meta-llama/Llama-3.2-1B-Instruct

# 또는 GPU 메모리 제한
--gpu-memory-utilization 0.7
```

### 4. 느린 분석 속도
- **캐싱 활성화**: 변경되지 않은 파일은 자동으로 스킵
- **작은 모델 사용**: 3B 모델 권장
- **배치 크기 조정**: vLLM 서버 설정

## 📁 출력 구조

```
docs/architecture/
├── index.json                    # 전체 프로젝트 개요
├── modules/                      # 모듈별 상세 문서
│   ├── myapp_core_engine.json
│   └── ...
├── relationships/                # 의존성 정보
│   └── dependencies.json
└── .cache/                       # 캐시 (자동 생성)
    └── ...
```

## 🆚 비교: AI 유무

| 항목 | AI 없이 (--no-ai) | AI 포함 |
|------|-------------------|---------|
| 속도 | ⚡⚡⚡ (1-2초) | ⚡⚡ (20-30초) |
| 설명 | 기본 (클래스/함수 수) | 상세 (역할, 책임, 패턴) |
| 중요도 | 기본 (normal) | 지능형 (low~critical) |
| 패턴 탐지 | ❌ | ✅ |
| 권장 사용 | CI/CD, 빠른 스캔 | 문서화, 리뷰 |

## 🔐 프라이버시

✅ **완전한 로컬 실행**
- 코드가 외부로 전송되지 않음
- vLLM 서버는 로컬에서 실행
- 인터넷 연결 불필요 (모델 다운로드 후)

✅ **보안**
- Symlink 공격 방지
- 경로 탐색 공격 차단
- 파일 크기 제한 (10MB)

## 📚 추가 리소스

### vLLM 문서
- [vLLM 공식 문서](https://docs.vllm.ai/)
- [Llama 모델](https://huggingface.co/meta-llama)

### Module Analyzer
- `module_analyzer.py`: AI 없는 기본 버전
- `module_analyzer_vllm.py`: vLLM 통합 버전 (현재 파일)

## 🎓 예제 워크플로우

### 1. 신규 프로젝트 분석
```bash
# 1. vLLM 서버 시작 (별도 터미널)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --port 8000

# 2. 전체 분석
python module_analyzer_vllm.py /path/to/project

# 3. 결과 확인
cat docs/architecture/index.json | jq .
```

### 2. 증분 업데이트
```bash
# 캐시 덕분에 변경된 파일만 재분석
python module_analyzer_vllm.py /path/to/project

# 출력:
#   [1/50] module1.py 💾 (캐시)
#   [2/50] module2.py 🤖 (새로 분석)
#   ...
```

### 3. CI/CD 통합
```bash
# AI 없이 빠르게 구조 변경 감지
python module_analyzer_vllm.py . --no-ai --no-cache

# 결과를 git에 커밋
git add docs/architecture/
git commit -m "Update architecture docs"
```

## 🤝 기여

문제 보고 또는 개선 제안:
- GitHub Issues: [dd-0321/DD_CONCEPTS](https://github.com/dd-0321/DD_CONCEPTS/issues)

## 📝 라이선스

MIT License

## 👤 저자

DD - Concept & Simulation Creator

---

**Note**: 이 도구는 코드 분석 및 문서화 목적으로만 사용하세요. 로컬 LLM을 사용하여 완전한 프라이버시를 보장합니다.

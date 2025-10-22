# Module Analyzer

Python 코드베이스를 자동으로 분석하고 문서화하는 도구

## 🎯 주요 기능

### ✨ 핵심 기능
- **자동 코드 분석**: Python 파일의 클래스, 메서드, 함수 자동 추출
- **복잡도 계산**: Cyclomatic Complexity 자동 계산
- **의존성 분석**: 모듈 간 import 관계 및 순환 의존성 탐지
- **JSON 문서 생성**: 구조화된 JSON 형식으로 문서 자동 생성
- **캐싱 지원**: 변경되지 않은 파일은 캐시에서 로드하여 성능 향상

### 🚀 개선 사항 (v2.0)

#### 버그 수정
- ✅ 순환 참조 탐지 알고리즘 수정
- ✅ 내부 모듈 판별 로직 개선
- ✅ 파일 인코딩 처리 강화 (UTF-8/Latin-1 fallback)
- ✅ 예외 처리 개선 (bare except 제거)

#### 성능 최적화
- ✅ 비동기 병렬 파일 분석
- ✅ AST 순회 최적화 (중복 제거)
- ✅ 병렬 JSON 파일 저장
- ✅ 캐싱 메커니즘 추가

#### 보안 강화
- ✅ Symlink 공격 방지
- ✅ 경로 탐색 공격 방지
- ✅ 파일 크기 제한 (10MB)
- ✅ 안전한 파일 읽기

#### 코드 품질
- ✅ 매직 넘버 상수화
- ✅ 데이터클래스 불변성 개선
- ✅ 타입 힌트 강화

## 📦 설치

```bash
# Python 3.11+ 필요
python --version

# 의존성 없음 (표준 라이브러리만 사용)
```

## 🚀 사용법

### 기본 사용
```bash
# 전체 프로젝트 분석
python module_analyzer.py /path/to/project --no-ai

# 특정 디렉토리만 분석
python module_analyzer.py /path/to/project --target src --no-ai

# 캐시 비활성화
python module_analyzer.py /path/to/project --no-ai --no-cache

# 특정 패턴 제외
python module_analyzer.py /path/to/project --no-ai --exclude tests docs migrations
```

### AI 분석 (향후 지원)
```bash
# AI API 설정 후 사용 가능
python module_analyzer.py /path/to/project
```

## 📊 출력 구조

```
docs/architecture/
├── index.json                    # 전체 프로젝트 개요
├── modules/                      # 모듈별 상세 문서
│   ├── module_name.json
│   └── ...
├── relationships/                # 의존성 정보
│   └── dependencies.json
└── .cache/                       # 캐시 파일
    └── ...
```

### index.json 구조
```json
{
  "generated_at": "2025-10-22T01:56:24Z",
  "project": "MyProject",
  "total_modules": 42,
  "total_loc": 15000,
  "layers": {
    "core": {"modules": [...], "count": 10},
    "api": {"modules": [...], "count": 8}
  },
  "modules": [
    {
      "id": "myapp.core.engine",
      "role": "핵심 엔진 로직",
      "complexity": "high",
      "importance": "critical",
      "loc": 450,
      "detail_file": "modules/myapp_core_engine.json"
    }
  ],
  "statistics": {
    "total_classes": 150,
    "total_functions": 320,
    "total_imports": 280,
    "circular_dependencies": 2
  },
  "warnings": [
    {
      "type": "circular_dependency",
      "count": 2,
      "severity": "medium",
      "cycles": [["A", "B", "A"]]
    }
  ]
}
```

### 모듈 상세 문서 구조
```json
{
  "module_id": "myapp.core.engine",
  "file_path": "myapp/core/engine.py",
  "summary": {
    "one_liner": "핵심 엔진 클래스",
    "description": "...",
    "responsibilities": [...],
    "complexity_level": "high",
    "importance": "critical"
  },
  "structure": {
    "classes": [
      {
        "name": "Engine",
        "docstring": "...",
        "methods": [
          {
            "name": "process",
            "signature": "def process(self, data: Dict) -> Result",
            "complexity": 12,
            "is_async": false,
            "calls": [...]
          }
        ]
      }
    ],
    "functions": [...],
    "imports": [...],
    "constants": {...}
  },
  "metrics": {
    "total_lines": 450,
    "num_classes": 3,
    "num_functions": 5,
    "num_methods": 25,
    "total_complexity": 78
  }
}
```

## 🔧 설정

### Config 클래스
```python
class Config:
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MIN_INIT_FILE_SIZE = 50
    CODE_PREVIEW_LENGTH = 2000
    HIGH_COMPLEXITY_THRESHOLD = 10
    CYCLOMATIC_COMPLEXITY_WARNING = 15
    MAX_WORKERS = 4
```

## 📈 성능

| 프로젝트 크기 | 파일 수 | 분석 시간 | 메모리 |
|--------------|---------|----------|--------|
| Small        | ~10     | < 1초    | ~50MB  |
| Medium       | ~100    | ~5초     | ~200MB |
| Large        | ~1000   | ~30초    | ~500MB |

*캐시 활성화 시 재분석 시간 90% 감소*

## 🎓 예시

### 분석 실행
```bash
$ python module_analyzer.py . --no-ai

📁 프로젝트: /home/user/myproject
📄 출력: /home/user/myproject/docs/architecture
💾 캐시: 활성화

🔍 Step 1: Python 파일 스캔...
   발견: 42개 파일

🔗 Step 2: 모듈 구조 분석...
   [1/42] myapp/core/engine.py ✓
   [2/42] myapp/api/views.py 💾
   ...

📊 Step 3: 의존성 관계 분석...

📝 Step 4: 문서 생성...
   ✓ index.json
   ✓ 42개 모듈 문서
   ✓ dependencies.json

✅ 완료!
   📊 분석: 42개 모듈
   ⏱️  소요: 2.5초
   📂 출력: docs/architecture

💡 확인: docs/architecture/index.json
```

## 🛡️ 보안 고려사항

- Symlink 자동 무시
- 프로젝트 외부 경로 접근 차단
- 10MB 이상 파일 자동 제외
- 안전한 AST 파싱 (타임아웃 없음)

## 🔮 향후 계획

- [ ] AI 분석 통합 (GPT-4, Claude API)
- [ ] HTML/Markdown 문서 생성
- [ ] 시각화 그래프 생성 (mermaid, graphviz)
- [ ] VS Code 확장 개발
- [ ] 변경 이력 추적
- [ ] 테스트 커버리지 통합

## 📝 라이선스

MIT License

## 👤 저자

DD - Concept & Simulation Creator

---

**Note**: 이 도구는 코드 분석 및 문서화 목적으로만 사용하세요. 악성 코드 분석이나 취약점 발견 목적으로 사용하지 마세요.

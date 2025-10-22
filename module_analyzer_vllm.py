"""
Module Analyzer with Local LLM (vLLM + Llama)
코드베이스 자동 분석 및 문서화 도구 - 로컬 LLM 버전

사용법:
    # vLLM 서버 먼저 시작 (별도 터미널):
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --port 8000

    # 분석 실행:
    python module_analyzer_vllm.py /path/to/project
    python module_analyzer_vllm.py /path/to/project --llm-url http://localhost:8000
    python module_analyzer_vllm.py /path/to/project --no-ai
"""

import ast
import json
import asyncio
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging

# HTTP 클라이언트
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    logging.warning("⚠️  httpx가 설치되지 않음. AI 기능 비활성화. pip install httpx")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# LLM 클라이언트
# =============================================================================

class LLMClient:
    """로컬 LLM 클라이언트 (vLLM + Llama)"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        timeout: float = 60.0
    ):
        if not HAS_HTTPX:
            raise ImportError("httpx 필요. pip install httpx")

        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout

        logger.info(f"🤖 LLM: {model}")
        logger.info(f"🔗 서버: {base_url}")

    async def test_connection(self) -> bool:
        """서버 연결 테스트"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ LLM 서버 연결 실패: {e}")
            return False

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 600,
        temperature: float = 0.1,
        stop: Optional[List[str]] = None
    ) -> str:
        """LLM 완성 요청"""

        if stop is None:
            stop = ["```\n", "\n\n\n\n", "<|eot_id|>"]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # OpenAI-compatible API
                response = await client.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": 0.95,
                        "stop": stop
                    }
                )

                response.raise_for_status()
                result = response.json()

                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['text'].strip()
                else:
                    raise ValueError(f"Invalid response format: {result}")

            except httpx.TimeoutException:
                raise Exception("LLM 서버 응답 시간 초과")
            except httpx.HTTPStatusError as e:
                raise Exception(f"LLM 서버 오류: {e.response.status_code}")
            except Exception as e:
                raise Exception(f"LLM 호출 실패: {str(e)}")


# =============================================================================
# 상수 정의
# =============================================================================

class Config:
    """분석 도구 설정 상수"""

    # 파일 크기 제한
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MIN_INIT_FILE_SIZE = 50  # bytes

    # 코드 분석
    CODE_PREVIEW_LENGTH = 1500  # LLM 컨텍스트 절약
    HIGH_COMPLEXITY_THRESHOLD = 10
    CYCLOMATIC_COMPLEXITY_WARNING = 15

    # 성능
    MAX_WORKERS = 4
    CACHE_DIR = ".cache"

    # 인코딩
    PRIMARY_ENCODING = 'utf-8'
    FALLBACK_ENCODING = 'latin-1'

    # LLM
    LLM_RETRY_COUNT = 3
    LLM_TIMEOUT = 60.0
    LLM_MAX_TOKENS = 600


# =============================================================================
# 데이터 클래스
# =============================================================================

@dataclass
class MethodInfo:
    """메서드 정보"""
    name: str
    signature: str
    docstring: Optional[str]
    params: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    complexity: int = 1
    is_async: bool = False
    calls: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ClassInfo:
    """클래스 정보"""
    name: str
    docstring: Optional[str]
    line_start: int
    line_end: int
    methods: List[MethodInfo] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)


@dataclass
class ModuleStructure:
    """모듈 구조"""
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[MethodInfo] = field(default_factory=list)
    imports: List[Dict[str, Any]] = field(default_factory=list)
    constants: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 메인 분석 클래스
# =============================================================================

class ModuleAnalyzer:
    """코드베이스 자동 분석 도구 (vLLM 통합 버전)"""

    def __init__(
        self,
        project_root: Path,
        use_ai: bool = True,
        use_cache: bool = True,
        llm_url: str = "http://localhost:8000",
        llm_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    ):
        self.project_root = Path(project_root)
        self.use_ai = use_ai
        self.use_cache = use_cache
        self.output_dir = self.project_root / "docs" / "architecture"
        self.cache_dir = self.output_dir / Config.CACHE_DIR

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "modules").mkdir(exist_ok=True)
        (self.output_dir / "relationships").mkdir(exist_ok=True)

        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)

        # LLM 클라이언트 초기화
        self.llm = None
        if self.use_ai and HAS_HTTPX:
            try:
                self.llm = LLMClient(base_url=llm_url, model=llm_model)
            except Exception as e:
                logger.warning(f"⚠️  LLM 초기화 실패: {e}")
                self.use_ai = False

        logger.info(f"📁 프로젝트: {self.project_root}")
        logger.info(f"📄 출력: {self.output_dir}")
        if self.use_cache:
            logger.info(f"💾 캐시: 활성화")
        if not self.use_ai:
            logger.info(f"🚫 AI: 비활성화 (구조만 분석)")

    async def analyze_project(
        self,
        target_dir: str = ".",
        exclude_patterns: Optional[List[str]] = None
    ):
        """전체 프로젝트 분석"""

        if exclude_patterns is None:
            exclude_patterns = [
                "__pycache__", ".venv", "venv", ".env", ".git",
                "node_modules", "tests", "test_", "dist", "build"
            ]

        # LLM 서버 연결 테스트
        if self.use_ai and self.llm:
            logger.info("\n🔌 LLM 서버 연결 테스트...")
            if await self.llm.test_connection():
                logger.info("   ✓ 연결 성공")
            else:
                logger.warning("   ⚠️  연결 실패 - AI 분석 비활성화")
                self.use_ai = False

        start_time = time.time()

        logger.info("\n🔍 Step 1: Python 파일 스캔...")
        python_files = self._find_python_files(
            self.project_root / target_dir,
            exclude_patterns
        )

        logger.info(f"   발견: {len(python_files)}개 파일")

        if not python_files:
            logger.warning("⚠️  Python 파일을 찾을 수 없습니다!")
            return

        # 모듈별 분석
        logger.info("\n🔗 Step 2: 모듈 구조 분석...")
        all_modules = await self._analyze_all_files(python_files)

        # 의존성 분석
        logger.info("\n📊 Step 3: 의존성 관계 분석...")
        relationships = self._analyze_relationships(all_modules)

        # index.json 생성
        logger.info("\n📝 Step 4: 문서 생성...")
        index = self._create_index(all_modules, relationships)

        # 파일 저장
        self._save_documentation(all_modules, relationships, index)

        elapsed = time.time() - start_time

        logger.info(f"\n✅ 완료!")
        logger.info(f"   📊 분석: {len(all_modules)}개 모듈")
        logger.info(f"   ⏱️  소요: {elapsed:.1f}초")
        if self.use_ai and self.llm:
            logger.info(f"   🤖 AI: 활성화")
        logger.info(f"   📂 출력: {self.output_dir}")
        logger.info(f"\n💡 확인: {self.output_dir / 'index.json'}")

    async def _analyze_all_files(
        self,
        python_files: List[Path]
    ) -> Dict[str, Dict[str, Any]]:
        """모든 파일을 병렬로 분석"""

        tasks = [self.analyze_file(file_path) for file_path in python_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_modules = {}
        for i, (file_path, result) in enumerate(zip(python_files, results), 1):
            rel_path = file_path.relative_to(self.project_root)

            if isinstance(result, Exception):
                logger.error(f"   [{i}/{len(python_files)}] {rel_path} ❌ {result}")
                continue

            status = "💾" if result.get('from_cache') else ("🤖" if self.use_ai else "✓")
            logger.info(f"   [{i}/{len(python_files)}] {rel_path} {status}")

            result.pop('from_cache', None)
            all_modules[result['module_id']] = result

        return all_modules

    def _find_python_files(
        self,
        directory: Path,
        exclude_patterns: List[str]
    ) -> List[Path]:
        """Python 파일 찾기"""

        python_files = []

        try:
            resolved_dir = directory.resolve()
        except (OSError, RuntimeError) as e:
            logger.error(f"❌ 디렉토리 접근 실패: {e}")
            return []

        for py_file in directory.rglob("*.py"):
            # Symlink 체크
            if py_file.is_symlink():
                continue

            # 경로 탐색 공격 방지
            try:
                py_file.resolve().relative_to(resolved_dir)
            except ValueError:
                continue

            # 제외 패턴 체크
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue

            # 파일 크기 체크
            try:
                file_size = py_file.stat().st_size

                if file_size > Config.MAX_FILE_SIZE:
                    logger.warning(f"   ⚠️  파일 너무 큼: {py_file.name}")
                    continue

                # __init__.py는 비어있으면 제외
                if py_file.name == "__init__.py":
                    if file_size < Config.MIN_INIT_FILE_SIZE:
                        continue

            except OSError:
                continue

            python_files.append(py_file)

        return sorted(python_files)

    def _get_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""

    async def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """단일 파일 분석"""

        rel_path = file_path.relative_to(self.project_root)
        module_id = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')

        # 캐시 확인
        if self.use_cache:
            cache_file = self.cache_dir / f"{module_id.replace('.', '_')}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached = json.load(f)

                    current_hash = self._get_file_hash(file_path)
                    if cached.get('file_hash') == current_hash:
                        cached['from_cache'] = True
                        return cached
                except Exception:
                    pass

        # 코드 읽기
        code = self._read_file_safe(file_path)
        if code is None:
            return self._create_error_doc(module_id, str(rel_path), "Encoding error")

        # AST 파싱
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return self._create_error_doc(module_id, str(rel_path), str(e))

        # 구조 추출
        structure = self._extract_structure(tree, code)

        # AI 분석
        if self.use_ai and self.llm:
            try:
                ai_description = await self._ask_ai_about_module(
                    code, structure, module_id
                )
            except Exception as e:
                logger.debug(f"      AI 실패: {e}")
                ai_description = self._create_default_description(structure)
        else:
            ai_description = self._create_default_description(structure)

        # 최종 문서 생성
        module_doc = self._create_module_doc(
            module_id=module_id,
            file_path=str(rel_path),
            structure=structure,
            ai_description=ai_description,
            code=code
        )

        module_doc['file_hash'] = self._get_file_hash(file_path)

        # 캐시 저장
        if self.use_cache:
            try:
                cache_file = self.cache_dir / f"{module_id.replace('.', '_')}.json"
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(module_doc, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        return module_doc

    def _read_file_safe(self, file_path: Path) -> Optional[str]:
        """안전한 파일 읽기"""
        try:
            with open(file_path, 'r', encoding=Config.PRIMARY_ENCODING) as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding=Config.FALLBACK_ENCODING) as f:
                    return f.read()
            except Exception:
                return None

    def _extract_structure(self, tree: ast.AST, code: str) -> ModuleStructure:
        """AST에서 구조 추출"""

        classes = []
        functions = []
        imports = []
        constants = {}

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(self._extract_class(node))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._extract_method(node))

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'type': 'import'
                    })

            elif isinstance(node, ast.ImportFrom):
                imports.append({
                    'module': node.module or '',
                    'items': [alias.name for alias in node.names],
                    'type': 'from'
                })

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        try:
                            constants[target.id] = ast.literal_eval(node.value)
                        except (ValueError, SyntaxError):
                            constants[target.id] = "<complex>"

        return ModuleStructure(
            classes=classes,
            functions=functions,
            imports=imports,
            constants=constants
        )

    def _extract_class(self, node: ast.ClassDef) -> ClassInfo:
        """클래스 정보 추출"""

        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_method(item))

        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                try:
                    base_classes.append(ast.unparse(base))
                except Exception:
                    base_classes.append("<complex>")

        return ClassInfo(
            name=node.name,
            docstring=ast.get_docstring(node),
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            methods=methods,
            base_classes=base_classes
        )

    def _extract_method(self, node: ast.FunctionDef) -> MethodInfo:
        """메서드/함수 정보 추출"""

        # 파라미터
        params = []
        for arg in node.args.args:
            param = {'name': arg.arg}
            if arg.annotation:
                try:
                    param['type'] = ast.unparse(arg.annotation)
                except Exception:
                    param['type'] = '<complex>'
            params.append(param)

        # 리턴 타입
        return_type = None
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except Exception:
                return_type = '<complex>'

        # 시그니처
        param_strs = []
        for p in params:
            if 'type' in p:
                param_strs.append(f"{p['name']}: {p['type']}")
            else:
                param_strs.append(p['name'])

        sig = f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {node.name}({', '.join(param_strs)})"
        if return_type:
            sig += f" -> {return_type}"

        # 호출 및 복잡도를 한 번의 순회로 추출
        calls, complexity = self._extract_calls_and_complexity(node)

        return MethodInfo(
            name=node.name,
            signature=sig,
            docstring=ast.get_docstring(node),
            params=params,
            return_type=return_type,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            complexity=complexity,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            calls=calls
        )

    def _extract_calls_and_complexity(
        self,
        node: ast.FunctionDef
    ) -> Tuple[List[Dict[str, str]], int]:
        """호출 및 복잡도를 한 번에 추출 (최적화)"""

        calls = []
        seen: Set[str] = set()
        complexity = 1

        for n in ast.walk(node):
            # 복잡도 계산
            if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(n, ast.BoolOp):
                complexity += len(n.values) - 1

            # 호출 추출
            if isinstance(n, ast.Call):
                call_info = None
                if isinstance(n.func, ast.Attribute):
                    if isinstance(n.func.value, ast.Name) and n.func.value.id == 'self':
                        call_info = {'type': 'method', 'name': n.func.attr}
                    else:
                        try:
                            call_info = {'type': 'external', 'name': ast.unparse(n.func)}
                        except Exception:
                            pass
                elif isinstance(n.func, ast.Name):
                    call_info = {'type': 'function', 'name': n.func.id}

                if call_info:
                    key = f"{call_info['type']}:{call_info['name']}"
                    if key not in seen:
                        calls.append(call_info)
                        seen.add(key)

        return calls, complexity

    async def _ask_ai_about_module(
        self,
        code: str,
        structure: ModuleStructure,
        module_id: str
    ) -> Dict[str, Any]:
        """LLM에게 모듈 분석 요청 (개선된 프롬프트)"""

        # 코드 미리보기
        code_preview = code[:Config.CODE_PREVIEW_LENGTH]

        # 요약 정보
        class_names = [c.name for c in structure.classes[:5]]
        method_summary = []
        for cls in structure.classes[:3]:
            for method in cls.methods[:3]:
                method_summary.append(f"{cls.name}.{method.name}()")

        import_modules = [imp['module'] for imp in structure.imports[:8]]

        # Llama에 최적화된 프롬프트
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Python code analyzer. Analyze the module and respond with JSON only.<|eot_id|><|start_header_id|>user<|end_header_id|>

Module ID: {module_id}
Classes: {', '.join(class_names) if class_names else 'None'}
Key Methods: {', '.join(method_summary[:5]) if method_summary else 'None'}
Imports: {', '.join(import_modules) if import_modules else 'None'}

Code snippet:
```python
{code_preview}
```

Respond with ONLY valid JSON (no markdown, no extra text):
{{
  "one_liner": "Brief summary in Korean (max 50 chars)",
  "description": "Detailed description in Korean (100-200 chars)",
  "responsibilities": ["responsibility1", "responsibility2", "responsibility3"],
  "key_patterns": ["pattern1", "pattern2"],
  "complexity_level": "low|medium|high",
  "importance": "low|normal|high|critical"
}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        # LLM 호출 (개선된 재시도 로직)
        last_error = None
        for attempt in range(Config.LLM_RETRY_COUNT):
            try:
                response = await self.llm.complete(
                    prompt=prompt,
                    max_tokens=Config.LLM_MAX_TOKENS,
                    temperature=0.1
                )

                # JSON 정제
                response = response.strip()

                # 마크다운 코드 블록 제거
                if '```json' in response:
                    response = response.split('```json', 1)[1]
                if '```' in response:
                    response = response.split('```', 1)[0]

                # 불필요한 텍스트 제거
                response = response.strip()

                # JSON 파싱
                try:
                    result = json.loads(response)
                except json.JSONDecodeError:
                    # 첫 번째 {부터 마지막 }까지만 추출
                    start = response.find('{')
                    end = response.rfind('}')
                    if start != -1 and end != -1:
                        response = response[start:end+1]
                        result = json.loads(response)
                    else:
                        raise

                # 필수 필드 검증
                required = ['one_liner', 'description', 'responsibilities',
                           'complexity_level', 'importance']

                if not all(k in result for k in required):
                    raise ValueError(f"Missing required fields. Got: {list(result.keys())}")

                # 타입 검증
                if not isinstance(result['responsibilities'], list):
                    result['responsibilities'] = []
                if 'key_patterns' not in result:
                    result['key_patterns'] = []
                elif not isinstance(result['key_patterns'], list):
                    result['key_patterns'] = []

                return result

            except json.JSONDecodeError as e:
                last_error = f"JSON 파싱 실패: {e}"
                if attempt < Config.LLM_RETRY_COUNT - 1:
                    logger.debug(f"      재시도 {attempt+1}/{Config.LLM_RETRY_COUNT}")
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
            except Exception as e:
                last_error = str(e)
                if attempt < Config.LLM_RETRY_COUNT - 1:
                    logger.debug(f"      LLM 오류 (재시도 {attempt+1}): {e}")
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue

        raise Exception(f"LLM 분석 실패 ({Config.LLM_RETRY_COUNT}회 시도): {last_error}")

    def _create_default_description(self, structure: ModuleStructure) -> Dict[str, Any]:
        """기본 설명 생성 (AI 없이)"""

        if structure.classes:
            class_names = [c.name for c in structure.classes]
            one_liner = f"{', '.join(class_names[:2])} 클래스 포함"

            responsibilities = []
            for cls in structure.classes:
                if cls.methods:
                    responsibilities.append(f"{cls.name}: {len(cls.methods)}개 메서드")
        else:
            one_liner = f"{len(structure.functions)}개 함수 포함"
            responsibilities = [f.name for f in structure.functions[:3]]

        total_complexity = sum(m.complexity for c in structure.classes for m in c.methods)
        total_complexity += sum(f.complexity for f in structure.functions)

        if total_complexity > 50:
            complexity_level = "high"
        elif total_complexity > 20:
            complexity_level = "medium"
        else:
            complexity_level = "low"

        return {
            "one_liner": one_liner,
            "description": f"이 모듈은 {len(structure.classes)}개 클래스와 {len(structure.functions)}개 함수를 포함합니다.",
            "responsibilities": responsibilities[:5],
            "key_patterns": [],
            "complexity_level": complexity_level,
            "importance": "normal"
        }

    def _create_module_doc(
        self,
        module_id: str,
        file_path: str,
        structure: ModuleStructure,
        ai_description: Dict[str, Any],
        code: str
    ) -> Dict[str, Any]:
        """최종 모듈 문서 생성"""

        # 클래스 변환
        classes_dict = []
        for cls in structure.classes:
            cls_dict = {
                'name': cls.name,
                'docstring': cls.docstring,
                'line_start': cls.line_start,
                'line_end': cls.line_end,
                'base_classes': cls.base_classes,
                'methods': []
            }

            for method in cls.methods:
                cls_dict['methods'].append({
                    'name': method.name,
                    'signature': method.signature,
                    'docstring': method.docstring,
                    'params': method.params,
                    'return_type': method.return_type,
                    'line_start': method.line_start,
                    'line_end': method.line_end,
                    'complexity': method.complexity,
                    'is_async': method.is_async,
                    'calls': method.calls
                })

            classes_dict.append(cls_dict)

        # 함수 변환
        functions_dict = []
        for func in structure.functions:
            functions_dict.append({
                'name': func.name,
                'signature': func.signature,
                'docstring': func.docstring,
                'params': func.params,
                'return_type': func.return_type,
                'line_start': func.line_start,
                'line_end': func.line_end,
                'complexity': func.complexity,
                'is_async': func.is_async,
                'calls': func.calls
            })

        doc = {
            'module_id': module_id,
            'file_path': file_path,
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),

            'summary': {
                'one_liner': ai_description['one_liner'],
                'description': ai_description['description'],
                'responsibilities': ai_description['responsibilities'],
                'complexity_level': ai_description['complexity_level'],
                'importance': ai_description['importance']
            },

            'structure': {
                'classes': classes_dict,
                'functions': functions_dict,
                'imports': structure.imports,
                'constants': structure.constants
            },

            'metrics': {
                'total_lines': len(code.split('\n')),
                'num_classes': len(structure.classes),
                'num_functions': len(structure.functions),
                'num_methods': sum(len(c.methods) for c in structure.classes),
                'total_complexity': sum(
                    m.complexity for c in structure.classes for m in c.methods
                ) + sum(f.complexity for f in structure.functions)
            }
        }

        # 고복잡도 경고
        high_complexity_methods = []
        for cls in structure.classes:
            for method in cls.methods:
                if method.complexity > Config.HIGH_COMPLEXITY_THRESHOLD:
                    high_complexity_methods.append({
                        'class': cls.name,
                        'method': method.name,
                        'complexity': method.complexity,
                        'lines': [method.line_start, method.line_end]
                    })

        if high_complexity_methods:
            doc['warnings'] = {'high_complexity': high_complexity_methods}

        return doc

    def _create_error_doc(
        self,
        module_id: str,
        file_path: str,
        error: str
    ) -> Dict[str, Any]:
        """에러 문서 생성"""

        return {
            'module_id': module_id,
            'file_path': file_path,
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'error': error,
            'summary': {
                'one_liner': '분석 실패',
                'description': f'오류: {error}',
                'responsibilities': [],
                'complexity_level': 'unknown',
                'importance': 'unknown'
            },
            'structure': {
                'classes': [],
                'functions': [],
                'imports': [],
                'constants': {}
            },
            'metrics': {
                'total_lines': 0,
                'num_classes': 0,
                'num_functions': 0,
                'num_methods': 0,
                'total_complexity': 0
            }
        }

    def _analyze_relationships(
        self,
        all_modules: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """모듈 간 관계 분석"""

        import_graph = {}

        for module_id, module_doc in all_modules.items():
            imports = []
            for imp in module_doc['structure']['imports']:
                imp_module = imp['module']
                if self._is_internal_module(imp_module, all_modules):
                    imports.append(imp_module)

            import_graph[module_id] = imports

        cycles = self._find_cycles(import_graph)

        return {
            'import_graph': import_graph,
            'cycles': cycles,
            'statistics': {
                'total_imports': sum(len(v) for v in import_graph.values()),
                'num_cycles': len(cycles)
            }
        }

    def _is_internal_module(
        self,
        module_name: str,
        all_modules: Dict[str, Dict[str, Any]]
    ) -> bool:
        """내부 모듈인지 확인"""

        if module_name in all_modules:
            return True

        mod_parts = module_name.split('.')
        for mod_id in all_modules.keys():
            mod_id_parts = mod_id.split('.')
            min_len = min(len(mod_parts), len(mod_id_parts))
            if mod_parts[:min_len] == mod_id_parts[:min_len]:
                return True

        return False

    def _find_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """순환 의존성 찾기"""

        cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                try:
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:]

                    min_idx = cycle.index(min(cycle))
                    normalized = cycle[min_idx:] + cycle[:min_idx]

                    if normalized not in cycles:
                        cycles.append(normalized)
                except ValueError:
                    pass
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                dfs(neighbor, path + [neighbor])

            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [node])

        return cycles

    def _create_index(
        self,
        all_modules: Dict[str, Dict[str, Any]],
        relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """index.json 생성"""

        layers = {}
        for module_id in all_modules.keys():
            parts = module_id.split('.')
            if len(parts) >= 2:
                layer_name = parts[0]
                if layer_name not in layers:
                    layers[layer_name] = {'modules': [], 'count': 0}
                layers[layer_name]['modules'].append(module_id)
                layers[layer_name]['count'] += 1

        modules_summary = []
        for module_id, module_doc in all_modules.items():
            modules_summary.append({
                'id': module_id,
                'role': module_doc['summary']['one_liner'],
                'complexity': module_doc['summary']['complexity_level'],
                'importance': module_doc['summary']['importance'],
                'loc': module_doc['metrics']['total_lines'],
                'detail_file': f"modules/{module_id.replace('.', '_')}.json"
            })

        importance_order = {'critical': 0, 'high': 1, 'normal': 2, 'low': 3}
        modules_summary.sort(
            key=lambda m: (importance_order.get(m['importance'], 9), -m['loc'])
        )

        index = {
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'project': self.project_root.name,
            'total_modules': len(all_modules),
            'total_loc': sum(m['metrics']['total_lines'] for m in all_modules.values()),

            'layers': layers,
            'modules': modules_summary,

            'statistics': {
                'total_classes': sum(m['metrics']['num_classes'] for m in all_modules.values()),
                'total_functions': sum(m['metrics']['num_functions'] for m in all_modules.values()),
                'total_imports': relationships['statistics']['total_imports'],
                'circular_dependencies': len(relationships['cycles'])
            },

            'warnings': []
        }

        if relationships['cycles']:
            index['warnings'].append({
                'type': 'circular_dependency',
                'count': len(relationships['cycles']),
                'severity': 'medium',
                'cycles': relationships['cycles']
            })

        high_complexity_modules = [
            m['module_id'] for m in all_modules.values()
            if m['summary']['complexity_level'] == 'high'
        ]
        if high_complexity_modules:
            index['warnings'].append({
                'type': 'high_complexity',
                'modules': high_complexity_modules,
                'severity': 'low'
            })

        return index

    def _save_documentation(
        self,
        all_modules: Dict[str, Dict[str, Any]],
        relationships: Dict[str, Any],
        index: Dict[str, Any]
    ):
        """문서 저장"""

        def save_json(path: Path, data: Dict[str, Any]) -> None:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = []

            # index.json
            index_path = self.output_dir / "index.json"
            futures.append(executor.submit(save_json, index_path, index))

            # 모듈 상세
            for module_id, module_doc in all_modules.items():
                filename = f"{module_id.replace('.', '_')}.json"
                module_path = self.output_dir / "modules" / filename
                futures.append(executor.submit(save_json, module_path, module_doc))

            # 의존성
            rel_path = self.output_dir / "relationships" / "dependencies.json"
            futures.append(executor.submit(save_json, rel_path, relationships))

            for future in futures:
                future.result()

        logger.info(f"   ✓ index.json")
        logger.info(f"   ✓ {len(all_modules)}개 모듈 문서")
        logger.info(f"   ✓ dependencies.json")


# =============================================================================
# CLI 진입점
# =============================================================================

async def main():
    """CLI 진입점"""

    import argparse

    parser = argparse.ArgumentParser(
        description='Python 코드베이스 자동 분석 도구 (vLLM + Llama)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # vLLM 서버 먼저 시작:
  python -m vllm.entrypoints.openai.api_server \\
      --model meta-llama/Llama-3.2-3B-Instruct \\
      --port 8000

  # 분석 실행:
  python module_analyzer_vllm.py /path/to/project
  python module_analyzer_vllm.py /path/to/project --target src
  python module_analyzer_vllm.py /path/to/project --llm-url http://localhost:8000
  python module_analyzer_vllm.py /path/to/project --no-ai  # AI 없이
        """
    )

    parser.add_argument('project_root', help='프로젝트 루트 디렉토리')
    parser.add_argument('--target', default='.', help='분석할 하위 디렉토리')
    parser.add_argument('--no-ai', action='store_true', help='AI 분석 비활성화')
    parser.add_argument('--no-cache', action='store_true', help='캐싱 비활성화')
    parser.add_argument('--exclude', nargs='+', help='제외할 패턴')
    parser.add_argument('--llm-url', default='http://localhost:8000', help='vLLM 서버 URL')
    parser.add_argument('--llm-model', default='meta-llama/Llama-3.2-3B-Instruct', help='모델명')

    args = parser.parse_args()

    # 분석 실행
    analyzer = ModuleAnalyzer(
        project_root=args.project_root,
        use_ai=not args.no_ai,
        use_cache=not args.no_cache,
        llm_url=args.llm_url,
        llm_model=args.llm_model
    )

    await analyzer.analyze_project(
        target_dir=args.target,
        exclude_patterns=args.exclude
    )


if __name__ == '__main__':
    asyncio.run(main())

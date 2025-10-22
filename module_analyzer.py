"""
Module Analyzer: 코드베이스 자동 분석 및 문서화 도구

사용법:
    python module_analyzer.py /path/to/project
    python module_analyzer.py /path/to/project --target v1
    python module_analyzer.py /path/to/project --no-ai
    python module_analyzer.py /path/to/project --no-cache
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

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 상수 정의
# =============================================================================

class Config:
    """분석 도구 설정 상수"""

    # 파일 크기 제한
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MIN_INIT_FILE_SIZE = 50  # bytes

    # 코드 분석
    CODE_PREVIEW_LENGTH = 2000
    HIGH_COMPLEXITY_THRESHOLD = 10
    CYCLOMATIC_COMPLEXITY_WARNING = 15

    # 성능
    MAX_WORKERS = 4
    CACHE_DIR = ".cache"

    # 인코딩
    PRIMARY_ENCODING = 'utf-8'
    FALLBACK_ENCODING = 'latin-1'


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
    """코드베이스 자동 분석 도구 (개선 버전)"""

    def __init__(
        self,
        project_root: Path,
        use_ai: bool = True,
        use_cache: bool = True
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

        logger.info(f"📁 프로젝트: {self.project_root}")
        logger.info(f"📄 출력: {self.output_dir}")
        if self.use_cache:
            logger.info(f"💾 캐시: 활성화")

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

        # 모듈별 분석 (병렬 처리)
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
        logger.info(f"   📂 출력: {self.output_dir}")
        logger.info(f"\n💡 확인: {self.output_dir / 'index.json'}")

    async def _analyze_all_files(
        self,
        python_files: List[Path]
    ) -> Dict[str, Dict[str, Any]]:
        """모든 파일을 병렬로 분석"""

        # 비동기 작업 생성
        tasks = [self.analyze_file(file_path) for file_path in python_files]

        # 병렬 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 수집
        all_modules = {}
        for i, (file_path, result) in enumerate(zip(python_files, results), 1):
            rel_path = file_path.relative_to(self.project_root)

            if isinstance(result, Exception):
                logger.error(f"   [{i}/{len(python_files)}] {rel_path} ❌ {result}")
                continue

            status = "💾" if result.get('from_cache') else "✓"
            logger.info(f"   [{i}/{len(python_files)}] {rel_path} {status}")

            # 캐시 플래그 제거
            result.pop('from_cache', None)
            all_modules[result['module_id']] = result

        return all_modules

    def _find_python_files(
        self,
        directory: Path,
        exclude_patterns: List[str]
    ) -> List[Path]:
        """Python 파일 찾기 (보안 강화)"""

        python_files = []

        try:
            resolved_dir = directory.resolve()
        except (OSError, RuntimeError) as e:
            logger.error(f"❌ 디렉토리 접근 실패: {e}")
            return []

        for py_file in directory.rglob("*.py"):
            # Symlink 체크 (보안)
            if py_file.is_symlink():
                logger.warning(f"   ⚠️  Symlink 무시: {py_file}")
                continue

            # 경로 탐색 공격 방지
            try:
                py_file.resolve().relative_to(resolved_dir)
            except ValueError:
                logger.warning(f"   ⚠️  경로 벗어남: {py_file}")
                continue

            # 제외 패턴 체크
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue

            # 파일 크기 체크
            try:
                file_size = py_file.stat().st_size

                if file_size > Config.MAX_FILE_SIZE:
                    logger.warning(
                        f"   ⚠️  파일 너무 큼: {py_file.name} "
                        f"({file_size / 1024 / 1024:.1f}MB)"
                    )
                    continue

                # __init__.py는 비어있으면 제외
                if py_file.name == "__init__.py":
                    if file_size < Config.MIN_INIT_FILE_SIZE:
                        continue

            except OSError as e:
                logger.warning(f"   ⚠️  파일 접근 실패: {py_file} - {e}")
                continue

            python_files.append(py_file)

        return sorted(python_files)

    def _get_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산 (캐싱용)"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""

    async def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """단일 파일 분석 (캐싱 지원)"""

        # 모듈 ID 생성
        rel_path = file_path.relative_to(self.project_root)
        module_id = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')

        # 캐시 확인
        if self.use_cache:
            cache_file = self.cache_dir / f"{module_id.replace('.', '_')}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached = json.load(f)

                    # 파일 해시 비교
                    current_hash = self._get_file_hash(file_path)
                    if cached.get('file_hash') == current_hash:
                        cached['from_cache'] = True
                        return cached
                except Exception as e:
                    logger.debug(f"   캐시 읽기 실패: {e}")

        # 코드 읽기 (인코딩 처리 강화)
        code = self._read_file_safe(file_path)
        if code is None:
            return self._create_error_doc(module_id, str(rel_path), "Encoding error")

        # AST 파싱
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.debug(f"      ⚠️  구문 오류: {e}")
            return self._create_error_doc(module_id, str(rel_path), str(e))

        # 구조 추출
        structure = self._extract_structure(tree, code)

        # AI 분석 (옵션)
        if self.use_ai:
            try:
                ai_description = await self._ask_ai_about_module(
                    code, structure, module_id
                )
            except Exception as e:
                logger.debug(f"      ⚠️  AI 분석 실패: {e}")
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

        # 파일 해시 추가
        module_doc['file_hash'] = self._get_file_hash(file_path)

        # 캐시 저장
        if self.use_cache:
            try:
                cache_file = self.cache_dir / f"{module_id.replace('.', '_')}.json"
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(module_doc, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.debug(f"   캐시 저장 실패: {e}")

        return module_doc

    def _read_file_safe(self, file_path: Path) -> Optional[str]:
        """안전한 파일 읽기 (인코딩 처리)"""

        # UTF-8 시도
        try:
            with open(file_path, 'r', encoding=Config.PRIMARY_ENCODING) as f:
                return f.read()
        except UnicodeDecodeError:
            pass

        # Fallback 인코딩 시도
        try:
            with open(file_path, 'r', encoding=Config.FALLBACK_ENCODING) as f:
                return f.read()
        except Exception as e:
            logger.warning(f"      ⚠️  인코딩 오류: {file_path.name} - {e}")
            return None

    def _extract_structure(self, tree: ast.AST, code: str) -> ModuleStructure:
        """AST에서 구조 추출"""

        classes = []
        functions = []
        imports = []
        constants = {}

        code_lines = code.split('\n')

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_info = self._extract_class(node, code_lines)
                classes.append(class_info)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._extract_method(node, code_lines)
                functions.append(func_info)

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
                    if isinstance(target, ast.Name):
                        if target.id.isupper():
                            try:
                                value = ast.literal_eval(node.value)
                                constants[target.id] = value
                            except (ValueError, SyntaxError):
                                constants[target.id] = "<complex>"

        return ModuleStructure(
            classes=classes,
            functions=functions,
            imports=imports,
            constants=constants
        )

    def _extract_class(self, node: ast.ClassDef, code_lines: List[str]) -> ClassInfo:
        """클래스 정보 추출"""

        methods = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self._extract_method(item, code_lines)
                methods.append(method_info)

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

    def _extract_method(
        self,
        node: ast.FunctionDef,
        code_lines: List[str]
    ) -> MethodInfo:
        """메서드/함수 정보 추출 (최적화: 한 번의 walk)"""

        # 파라미터
        params = []
        for arg in node.args.args:
            param = {'name': arg.arg}
            if arg.annotation:
                try:
                    param['type'] = ast.unparse(arg.annotation)
                except Exception:
                    param['type'] = "<complex>"
            params.append(param)

        # 리턴 타입
        return_type = None
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except Exception:
                return_type = "<complex>"

        # 시그니처 생성
        param_strs = []
        for p in params:
            if 'type' in p:
                param_strs.append(f"{p['name']}: {p['type']}")
            else:
                param_strs.append(p['name'])

        sig = f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {node.name}({', '.join(param_strs)})"
        if return_type:
            sig += f" -> {return_type}"

        # 한 번의 walk로 calls와 complexity 추출
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
        """함수 내부 호출 및 복잡도를 한 번의 순회로 계산 (최적화)"""

        calls = []
        seen_calls: Set[str] = set()
        complexity = 1

        for n in ast.walk(node):
            # Complexity 계산
            if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(n, ast.BoolOp):
                complexity += len(n.values) - 1

            # Calls 추출
            if isinstance(n, ast.Call):
                call_info = None

                if isinstance(n.func, ast.Attribute):
                    if isinstance(n.func.value, ast.Name) and n.func.value.id == 'self':
                        call_info = {'type': 'method', 'name': n.func.attr}
                    else:
                        try:
                            call_info = {'type': 'external', 'name': ast.unparse(n.func)}
                        except Exception:
                            call_info = {'type': 'external', 'name': '<unparseable>'}

                elif isinstance(n.func, ast.Name):
                    call_info = {'type': 'function', 'name': n.func.id}

                if call_info:
                    key = f"{call_info['type']}:{call_info['name']}"
                    if key not in seen_calls:
                        calls.append(call_info)
                        seen_calls.add(key)

        return calls, complexity

    async def _ask_ai_about_module(
        self,
        code: str,
        structure: ModuleStructure,
        module_id: str
    ) -> Dict[str, Any]:
        """AI에게 모듈 역할 물어보기"""

        code_preview = code[:Config.CODE_PREVIEW_LENGTH] if len(code) > Config.CODE_PREVIEW_LENGTH else code

        class_names = [c.name for c in structure.classes]
        method_summary = []
        for cls in structure.classes:
            for method in cls.methods[:5]:
                method_summary.append(f"{cls.name}.{method.name}()")

        import_modules = [imp['module'] for imp in structure.imports[:10]]

        prompt = f"""다음 Python 모듈을 분석하고 역할을 설명하세요.

**모듈**: {module_id}
**클래스**: {class_names}
**주요 메서드**: {method_summary}
**Import**: {import_modules}

**코드 일부**:
```python
{code_preview}
```

다음 JSON 형식으로만 답변하세요 (다른 텍스트 없이):
{{
  "one_liner": "한 줄 요약 (50자 이내)",
  "description": "상세 설명 (100-200자)",
  "responsibilities": ["책임1", "책임2", "책임3"],
  "key_patterns": ["패턴1", "패턴2"],
  "complexity_level": "low|medium|high",
  "importance": "low|normal|high|critical"
}}

중요: JSON만 출력하세요."""

        # API 호출 시뮬레이션 (실제로는 구현 필요)
        raise Exception("AI API not configured - use --no-ai flag")

    def _create_default_description(self, structure: ModuleStructure) -> Dict[str, Any]:
        """AI 없이 기본 설명 생성"""

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
                method_dict = {
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
                }
                cls_dict['methods'].append(method_dict)

            classes_dict.append(cls_dict)

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

        # 복잡도 경고
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

        for func in structure.functions:
            if func.complexity > Config.HIGH_COMPLEXITY_THRESHOLD:
                high_complexity_methods.append({
                    'function': func.name,
                    'complexity': func.complexity,
                    'lines': [func.line_start, func.line_end]
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
        """에러 발생 시 기본 문서"""

        return {
            'module_id': module_id,
            'file_path': file_path,
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'error': error,
            'summary': {
                'one_liner': '분석 실패',
                'description': f'구문 오류: {error}',
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
        call_graph = {}

        for module_id, module_doc in all_modules.items():
            imports = []
            for imp in module_doc['structure']['imports']:
                imp_module = imp['module']
                if self._is_internal_module(imp_module, all_modules):
                    imports.append(imp_module)

            import_graph[module_id] = imports

            calls = []
            for cls in module_doc['structure']['classes']:
                for method in cls['methods']:
                    for call in method['calls']:
                        if call['type'] == 'method':
                            calls.append({
                                'from': f"{module_id}.{cls['name']}.{method['name']}",
                                'to': call['name'],
                                'type': 'method_call'
                            })

            call_graph[module_id] = calls

        # 순환 의존성 찾기 (개선된 알고리즘)
        cycles = self._find_cycles(import_graph)

        return {
            'import_graph': import_graph,
            'call_graph': call_graph,
            'cycles': cycles,
            'statistics': {
                'total_imports': sum(len(v) for v in import_graph.values()),
                'total_calls': sum(len(v) for v in call_graph.values()),
                'num_cycles': len(cycles)
            }
        }

    def _is_internal_module(
        self,
        module_name: str,
        all_modules: Dict[str, Dict[str, Any]]
    ) -> bool:
        """내부 모듈인지 확인 (개선된 로직)"""

        # 직접 매칭
        if module_name in all_modules:
            return True

        # 부모/자식 관계 체크
        mod_parts = module_name.split('.')

        for mod_id in all_modules.keys():
            mod_id_parts = mod_id.split('.')

            # 정확한 prefix 매칭
            min_len = min(len(mod_parts), len(mod_id_parts))
            if mod_parts[:min_len] == mod_id_parts[:min_len]:
                return True

        return False

    def _find_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """순환 의존성 찾기 (개선된 알고리즘)"""

        cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # 순환 발견
                try:
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:]

                    # 정규화된 형태로 저장 (중복 제거)
                    # 순환의 최소 노드부터 시작하도록 정렬
                    min_idx = cycle.index(min(cycle))
                    normalized = cycle[min_idx:] + cycle[:min_idx]

                    # 이미 존재하는지 확인
                    is_duplicate = False
                    for existing_cycle in cycles:
                        if len(existing_cycle) == len(normalized):
                            # 회전된 형태도 체크
                            for i in range(len(normalized)):
                                rotated = normalized[i:] + normalized[:i]
                                if existing_cycle == rotated:
                                    is_duplicate = True
                                    break
                        if is_duplicate:
                            break

                    if not is_duplicate:
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

        layers = self._categorize_layers(all_modules)

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

    def _categorize_layers(
        self,
        all_modules: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """모듈을 레이어별로 분류"""

        layers = {}

        for module_id in all_modules.keys():
            parts = module_id.split('.')

            if len(parts) >= 2:
                layer_name = parts[0]

                if layer_name not in layers:
                    layers[layer_name] = {
                        'modules': [],
                        'count': 0
                    }

                layers[layer_name]['modules'].append(module_id)
                layers[layer_name]['count'] += 1

        return layers

    def _save_documentation(
        self,
        all_modules: Dict[str, Dict[str, Any]],
        relationships: Dict[str, Any],
        index: Dict[str, Any]
    ):
        """문서 파일 저장 (병렬 처리)"""

        def save_json(path: Path, data: Dict[str, Any]) -> None:
            """JSON 파일 저장 헬퍼"""
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = []

            # 1. index.json
            index_path = self.output_dir / "index.json"
            futures.append(executor.submit(save_json, index_path, index))

            # 2. 각 모듈 상세
            for module_id, module_doc in all_modules.items():
                filename = f"{module_id.replace('.', '_')}.json"
                module_path = self.output_dir / "modules" / filename
                futures.append(executor.submit(save_json, module_path, module_doc))

            # 3. 관계 정보
            rel_path = self.output_dir / "relationships" / "dependencies.json"
            futures.append(executor.submit(save_json, rel_path, relationships))

            # 모든 작업 완료 대기
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
        description='Python 코드베이스 자동 분석 도구 (개선 버전)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python module_analyzer.py /path/to/project
  python module_analyzer.py /path/to/project --target src
  python module_analyzer.py /path/to/project --no-ai --no-cache
  python module_analyzer.py /path/to/project --exclude tests docs
        """
    )

    parser.add_argument(
        'project_root',
        help='프로젝트 루트 디렉토리'
    )

    parser.add_argument(
        '--target',
        default='.',
        help='분석할 하위 디렉토리 (기본: 전체)'
    )

    parser.add_argument(
        '--no-ai',
        action='store_true',
        help='AI 분석 비활성화 (구조만 추출)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='캐싱 비활성화'
    )

    parser.add_argument(
        '--exclude',
        nargs='+',
        help='제외할 패턴 (예: tests __pycache__)'
    )

    args = parser.parse_args()

    # 분석 실행
    analyzer = ModuleAnalyzer(
        project_root=args.project_root,
        use_ai=not args.no_ai,
        use_cache=not args.no_cache
    )

    await analyzer.analyze_project(
        target_dir=args.target,
        exclude_patterns=args.exclude
    )


if __name__ == '__main__':
    asyncio.run(main())

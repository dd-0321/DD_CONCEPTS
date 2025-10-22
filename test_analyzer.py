"""
Module Analyzer 테스트
"""

import asyncio
from pathlib import Path
from module_analyzer import ModuleAnalyzer


async def test_basic_functionality():
    """기본 기능 테스트"""

    print("=" * 60)
    print("Module Analyzer 테스트 시작")
    print("=" * 60)

    # 현재 디렉토리를 분석 대상으로
    current_dir = Path(__file__).parent

    # 분석기 생성
    analyzer = ModuleAnalyzer(
        project_root=current_dir,
        use_ai=False,  # AI 없이 테스트
        use_cache=True
    )

    print("\n1. Python 파일 찾기 테스트...")
    python_files = analyzer._find_python_files(
        current_dir,
        exclude_patterns=["__pycache__", ".venv", "venv"]
    )
    print(f"   ✓ {len(python_files)}개 파일 발견")
    for f in python_files[:5]:
        print(f"     - {f.name}")

    print("\n2. 단일 파일 분석 테스트...")
    if python_files:
        test_file = python_files[0]
        print(f"   분석 대상: {test_file.name}")

        result = await analyzer.analyze_file(test_file)

        print(f"   ✓ 모듈 ID: {result['module_id']}")
        print(f"   ✓ 클래스: {result['metrics']['num_classes']}개")
        print(f"   ✓ 함수: {result['metrics']['num_functions']}개")
        print(f"   ✓ 복잡도: {result['summary']['complexity_level']}")

    print("\n3. 순환 의존성 탐지 테스트...")
    # 테스트 그래프
    test_graph = {
        'A': ['B'],
        'B': ['C'],
        'C': ['A'],  # A -> B -> C -> A 순환
        'D': ['E'],
        'E': ['D']   # D -> E -> D 순환
    }

    cycles = analyzer._find_cycles(test_graph)
    print(f"   ✓ {len(cycles)}개 순환 발견:")
    for cycle in cycles:
        print(f"     - {' -> '.join(cycle)}")

    print("\n4. 내부 모듈 판별 테스트...")
    test_modules = {
        'myapp.core.engine': {},
        'myapp.utils.helpers': {},
        'myapp.api.views': {}
    }

    test_cases = [
        ('myapp.core', True),
        ('myapp.core.engine', True),
        ('external.lib', False),
        ('myapp', True),
    ]

    for module_name, expected in test_cases:
        result = analyzer._is_internal_module(module_name, test_modules)
        status = "✓" if result == expected else "✗"
        print(f"   {status} '{module_name}' -> {result} (예상: {expected})")

    print("\n" + "=" * 60)
    print("✅ 모든 테스트 완료!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(test_basic_functionality())

"""
vLLM Module Analyzer 테스트
"""

import asyncio
from pathlib import Path
from module_analyzer_vllm import ModuleAnalyzer, LLMClient, HAS_HTTPX


async def test_basic_functionality():
    """기본 기능 테스트 (AI 없이)"""

    print("=" * 60)
    print("Module Analyzer vLLM - 기본 테스트")
    print("=" * 60)

    current_dir = Path(__file__).parent

    # AI 없이 분석기 생성
    analyzer = ModuleAnalyzer(
        project_root=current_dir,
        use_ai=False,
        use_cache=False
    )

    print("\n1. Python 파일 찾기 테스트...")
    python_files = analyzer._find_python_files(
        current_dir,
        exclude_patterns=["__pycache__", ".venv", "test_"]
    )
    print(f"   ✓ {len(python_files)}개 파일 발견")
    for f in python_files[:3]:
        print(f"     - {f.name}")

    print("\n2. 단일 파일 분석 테스트 (AI 없이)...")
    if python_files:
        test_file = [f for f in python_files if f.name == "module_analyzer.py"]
        if test_file:
            test_file = test_file[0]
            print(f"   분석 대상: {test_file.name}")

            result = await analyzer.analyze_file(test_file)

            print(f"   ✓ 모듈 ID: {result['module_id']}")
            print(f"   ✓ 클래스: {result['metrics']['num_classes']}개")
            print(f"   ✓ 함수: {result['metrics']['num_functions']}개")
            print(f"   ✓ 복잡도: {result['summary']['complexity_level']}")
            print(f"   ✓ 설명: {result['summary']['one_liner']}")

    print("\n" + "=" * 60)
    print("✅ 기본 테스트 완료!")
    print("=" * 60)


async def test_llm_connection():
    """LLM 서버 연결 테스트"""

    print("\n" + "=" * 60)
    print("LLM 서버 연결 테스트")
    print("=" * 60)

    if not HAS_HTTPX:
        print("\n❌ httpx가 설치되지 않았습니다.")
        print("   설치: pip install httpx")
        return

    print("\n🔌 http://localhost:8000 연결 시도...")

    try:
        llm = LLMClient(
            base_url="http://localhost:8000",
            model="meta-llama/Llama-3.2-3B-Instruct"
        )

        # 연결 테스트
        if await llm.test_connection():
            print("   ✅ 연결 성공!")

            # 간단한 완성 테스트
            print("\n📝 간단한 프롬프트 테스트...")
            prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Say "Hello from vLLM!" in JSON format:
{"message": "..."}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            response = await llm.complete(
                prompt=prompt,
                max_tokens=50,
                temperature=0.1
            )

            print(f"   응답: {response[:100]}...")

            print("\n✅ LLM 서버가 정상 작동 중입니다!")
            print("   이제 AI 분석을 사용할 수 있습니다:")
            print("   python module_analyzer_vllm.py .")

        else:
            print("   ❌ 연결 실패")
            print("\n💡 vLLM 서버를 먼저 시작하세요:")
            print("   python -m vllm.entrypoints.openai.api_server \\")
            print("       --model meta-llama/Llama-3.2-3B-Instruct \\")
            print("       --port 8000")

    except Exception as e:
        print(f"   ❌ 오류: {e}")
        print("\n💡 vLLM 서버를 먼저 시작하세요:")
        print("   python -m vllm.entrypoints.openai.api_server \\")
        print("       --model meta-llama/Llama-3.2-3B-Instruct \\")
        print("       --port 8000")

    print("=" * 60)


async def main():
    """메인 테스트 실행"""

    # 1. 기본 기능 테스트
    await test_basic_functionality()

    # 2. LLM 연결 테스트
    await test_llm_connection()

    print("\n" + "=" * 60)
    print("모든 테스트 완료!")
    print("=" * 60)
    print("\n다음 단계:")
    print("1. vLLM 서버 시작:")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("       --model meta-llama/Llama-3.2-3B-Instruct \\")
    print("       --port 8000")
    print("\n2. AI 분석 실행:")
    print("   python module_analyzer_vllm.py .")
    print("\n3. 결과 확인:")
    print("   cat docs/architecture/index.json | jq .")


if __name__ == '__main__':
    asyncio.run(main())

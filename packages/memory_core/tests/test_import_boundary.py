"""边界守卫：memory_core 不得依赖 Web 框架或宿主层。

这是 P0 抽包的硬性架构不变量（见 docs/derivative/01-架构改造方案.md §4）。
违反即测试失败，等价于 CI import-lint。
"""

import ast
from pathlib import Path

import memory_core

PACKAGE_ROOT = Path(memory_core.__file__).parent

# 内核绝对不允许出现的顶层依赖
FORBIDDEN_TOP_LEVEL = {
    "fastapi",
    "slowapi",
    "starlette",
    "uvicorn",
    "routers",
    "services",  # 裸 services = backend 宿主的包，内核只能用 memory_core.services
    "main",
}


def _iter_import_roots(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]


def _scan(root: Path) -> list[str]:
    violations = []
    for py_file in sorted(root.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for imported in _iter_import_roots(tree):
            if imported in FORBIDDEN_TOP_LEVEL:
                violations.append(f"{py_file.relative_to(root)}: import {imported}")
    return violations


def test_memory_core_has_no_web_or_host_imports():
    violations = _scan(PACKAGE_ROOT)
    assert not violations, "内核出现禁止依赖：\n" + "\n".join(violations)


def test_kernel_tests_have_no_web_or_host_imports():
    # 内核测试也必须是「纯内核」——依赖 fastapi/routers/main 的测试属于 backend 宿主层，
    # 不应留在 memory_core/tests（arch-3：边界守卫此前漏扫 tests 目录）。
    tests_root = Path(__file__).parent
    violations = _scan(tests_root)
    assert not violations, "内核测试出现禁止依赖：\n" + "\n".join(violations)

# Project Overview
Persona 5 Royal Generative Social Links. We are injecting a local LLM into P5R using a C# Reloaded-II mod (memory hooking) and a Python background server running custom Triton kernels for 4-bit quantized inference.

## The Micro-Commit Loop (MANDATORY)
You are acting as an expert systems engineering mentor and pair programmer. For every single task requested, you MUST execute the following 4 steps in order. Do not skip steps.

1. **TEACH:** Append a detailed, technical explanation of the concept we are about to build to `learning.md`. Explain the "why" behind the architecture, memory layouts, or C# pointer arithmetic.
2. **CODE:** Write or modify the specific, isolated piece of code required for the current micro-step.
3. **TEST:** Run the appropriate test, linter, or compiler command (e.g., `dotnet build` or `python -m pytest`). If it fails, fix it before proceeding.
4. **COMMIT:** Once the code passes, immediately execute a Git commit.

## Git Protocol
- Commit messages must follow Conventional Commits (e.g., `feat:`, `fix:`, `refactor:`, `docs:`).
- Commits must be atomic. Do not bundle multiple logical changes into one commit.
- Automatically execute `git add` and `git commit`. Do not wait for user approval to commit if the tests pass.

## Coding Standards
- **Python:** Use strict type hinting. No generic `Exception` catching.
- **Triton:** Heavily comment block pointer math and `tl.load` mask logic.
- **C#:** Use `unsafe` blocks explicitly when dealing with game memory pointers.
# Main promotion status

**Status (2026-07-23): complete.**
Git `main` is now the canonical branch for the DS-Conv2D-H1 source-safe deploy path.

Do not use `research/fpga-1dcnn-90acc`, `research/ast-teacher-*`, or `reproduce/abdoli-*` for new Phase 1 runs unless the owner explicitly asks to inspect historical work.

## Current server checkout

```bash
cd ~/Dung_TDTU/AI_accelerator_sound_source_tests
git fetch origin
git checkout main
git pull origin main
test -f configs/main/student_ds_conv2d_h1_pyramid_sourcegroup.json && echo OK_MAIN_CONFIG
test -f docs/main/README.md && echo OK_DOCS
```

## Historical note

This file used to describe the promotion procedure from `research/fpga-1dcnn-90acc` to `main`. That procedure is no longer an active task. New work should branch from or run directly on the updated `main` according to [SERVER_POLICY.md](SERVER_POLICY.md).

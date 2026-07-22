$ErrorActionPreference = "Stop"

$PythonExe = "C:\Users\Dawin\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

& $PythonExe tools/run_ast_teacher_multifold.py `
    --exp_name local_ast_teacher_full10_12ep `
    --folds 1-10 `
    --epochs 12 `
    --batch_size 4 `
    --eval_batch_size 8 `
    --accum_steps 4 `
    --encoder_lr 1e-5 `
    --head_lr 5e-4 `
    --early_stop_warmup 6 `
    --early_stop_patience 5 `
    --num_workers 0 `
    --local_files_only `
    --skip_existing

param(
    [ValidateSet(
        "rmsnorm_shared",
        "rmsnorm_warp",
        "rmsnorm_half2",
        "fused_shared",
        "fused_warp",
        "fused_half2"
    )]
    [string]$Op = "rmsnorm_warp",

    [ValidateSet("float32", "float16", "bfloat16")]
    [string]$DType = "float16",

    [int]$Batch = 32,
    [int]$SeqLen = 1,
    [int]$HiddenSize = 4096,
    [int]$Warmup = 10,
    [int]$Repeat = 20,
    [string]$Output = "profiles\ncu_rmsnorm"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python venv not found: $Python"
}

$Ncu = Get-Command ncu -ErrorAction SilentlyContinue
if (-not $Ncu) {
    $NcuExe = Get-ChildItem `
        -Path "C:\Program Files\NVIDIA Corporation" `
        -Recurse `
        -Filter ncu.exe `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1

    if (-not $NcuExe) {
        throw "ncu was not found. Install NVIDIA Nsight Compute or add ncu.exe to PATH."
    }

    $NcuPath = $NcuExe.FullName
}
else {
    $NcuPath = $Ncu.Source
}

Push-Location $ProjectRoot
try {
    New-Item -ItemType Directory -Force -Path "profiles" | Out-Null

    & $NcuPath `
        --set=roofline `
        --launch-skip $Warmup `
        --launch-count 1 `
        --force-overwrite `
        --export=$Output `
        $Python benchmarks\profile_rmsnorm.py `
        --op $Op `
        --dtype $DType `
        --batch $Batch `
        --seq-len $SeqLen `
        --hidden-size $HiddenSize `
        --warmup $Warmup `
        --repeat $Repeat
}
finally {
    Pop-Location
}

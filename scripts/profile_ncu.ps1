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
    throw "ncu was not found in PATH. Install NVIDIA Nsight Compute or add it to PATH."
}

Push-Location $ProjectRoot
try {
    New-Item -ItemType Directory -Force -Path "profiles" | Out-Null

    & $Ncu.Source `
        --set=roofline `
        --launch-skip $Warmup `
        --launch-count 1 `
        --force-overwrite `
        --export=$Output `
        $Python benchmarks\profile_rmsnorm.py `
        --op $Op `
        --dtype $DType `
        --batch $Batch `
        --hidden-size $HiddenSize `
        --warmup $Warmup `
        --repeat $Repeat
}
finally {
    Pop-Location
}

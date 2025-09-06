Param(
    [int]$Threads = 4,
    [int]$Nctx = 2048
)

Write-Host "Setting LLAMA_THREADS=$Threads and LLAMA_N_CTX=$Nctx for this session..."
$env:LLAMA_THREADS = $Threads
$env:LLAMA_N_CTX = $Nctx

$dataDir = Join-Path -Path (Get-Location) -ChildPath "data"
if (-not (Test-Path $dataDir)) {
    Write-Host "Creating data directory at $dataDir"
    New-Item -ItemType Directory -Path $dataDir | Out-Null
} else {
    Write-Host "Data directory exists: $dataDir"
}

Write-Host "LLM_BACKEND = $env:LLM_BACKEND"
Write-Host "LLM_MODEL  = $env:LLM_MODEL"
Write-Host "CHROMA_PERSIST_DIR = $env:CHROMA_PERSIST_DIR"
Write-Host "TTS_ENGINE = $env:TTS_ENGINE"

Write-Host "Launching app.py..."
python app.py

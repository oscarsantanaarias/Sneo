# arma el entorno de python para entrenar: torch con cuda, transformers, peft y lo que pide el convertidor a gguf
$ErrorActionPreference = "Continue"
$here = $PSScriptRoot
$venv = Join-Path $here "venv"
if(-not (Test-Path "$venv\Scripts\python.exe")){
    Write-Host "Creating the Python environment..."
    & py -3.12 -m venv $venv
}
$python = "$venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
Write-Host "Installing PyTorch with CUDA (about 3 GB)..."
& $python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
Write-Host "Installing the training libraries..."
& $python -m pip install transformers peft accelerate datasets bitsandbytes safetensors sentencepiece protobuf numpy requests
# los kernels rapidos de la atencion lineal de qwen3.5 son opcionales: sin ellos entrena igual pero mas lento
& $python -m pip install triton-windows flash-linear-attention 2>$null
if($LASTEXITCODE -ne 0){ Write-Host "Fast linear attention kernels not installed, training still works (slower)." }
& $python -W ignore -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| peft', peft.__version__)"
Write-Host "Ready."

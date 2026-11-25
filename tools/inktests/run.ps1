param([string]$Tag = "ink", [int]$Context = 16384, [string]$Ask = "ask1.mjs", [switch]$Keep, [string]$Model = "F:\AI Projects\Strata\Models\Inkling-2-Mini\Inkling-2-Mini-UD-Q4_K_XL.gguf", [string]$Dump = "", [switch]$Vision, [switch]$Profile)
$here = $PSScriptRoot
$tree = (Resolve-Path "$here\..\..").Path
$exe = "$tree\build\master_ink.exe"
$api = "http://127.0.0.1:8010"
$log = "$here\logs\$Tag.log"
$busy = @(Get-Process | Where-Object { ($_.Path -match '[\\/]sneo\.exe$' -and $_.Path -notlike 'F:\Sneo\*') -or ($_.Path -match '[\\/]master\.exe$') -or $_.Path -match 'Strata' })
if ($busy) { "not testing, running: $(($busy | ForEach-Object { $_.Path }) -join ', ')"; exit 1 }
if (-not (Test-Path "$here\build\master.exe")) { "build first with build.bat"; exit 1 }
New-Item -ItemType Directory -Force "$here\logs" | Out-Null
Copy-Item "$here\build\master.exe" $exe -Force
$pool = [ordered]@{acceptRequests = $false; vision = [bool]$Vision; secondGpu = $false; helperGpu = -1; context = $Context; speculate = $false; fastRouting = $false; holdRam = $true}
($pool | ConvertTo-Json -Compress) | Set-Content -Encoding ascii "$here\logs\pool.json"
$env:CUDA_VISIBLE_DEVICES = "GPU-486c00fa-7213-019b-2a6e-6f2dd91f1f4d"
if ($Dump) { $env:SNEO_INK_DUMP = $Dump }
if ($Profile) { $env:SNEO_INK_PROFILE = "1" }
$proc = Start-Process -FilePath $exe -ArgumentList @("8010", "-", "-", "`"$tree\modelos`"", "`"$Model`"", "--pool", "`"$here\logs\pool.json`"") -WorkingDirectory $tree -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
Remove-Item Env:CUDA_VISIBLE_DEVICES
if ($Dump) { Remove-Item Env:SNEO_INK_DUMP }
if ($Profile) { Remove-Item Env:SNEO_INK_PROFILE }
$began = Get-Date
try {
    $loaded = $false
    for ($i = 0; $i -lt 600; $i++) {
        Start-Sleep 2
        if ($proc.HasExited) { break }
        try { $state = Invoke-RestMethod -TimeoutSec 5 "$api/api/models" } catch { continue }
        if ($state.ready) { $loaded = $true; break }
    }
    "load {0} after {1:N0} s" -f ($(if ($loaded) { "ok" } else { "FAILED" }), ((Get-Date) - $began).TotalSeconds)
    if ($loaded) { & node "$here\$Ask" 8010 }
} finally {
    if (-not $Keep -and -not $proc.HasExited) { Get-Process -Id $proc.Id -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe } | Stop-Process -Force -Confirm:$false }
}
Get-Content $log | Select-String "generated|kernels over|inkling2_loop:|failed" | ForEach-Object { $_.Line }

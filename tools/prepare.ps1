$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$root = Split-Path -Parent $PSScriptRoot
$ui = Join-Path $root "ui"
$tools = Join-Path $root "tools"

if(-not (Get-Command node.exe -ErrorAction SilentlyContinue)){
    $local = Get-ChildItem (Join-Path $tools "node") -Filter node.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if(-not $local){
        Write-Host "Node.js is not installed, downloading a private copy for Sneo..."
        try {
            $release = Invoke-RestMethod "https://nodejs.org/dist/index.json" | Where-Object { $_.lts } | Select-Object -First 1
            $name = "node-$($release.version)-win-x64"
            $zip = Join-Path $env:TEMP "$name.zip"
            Invoke-WebRequest "https://nodejs.org/dist/$($release.version)/$name.zip" -OutFile $zip -UseBasicParsing
            Expand-Archive $zip (Join-Path $tools "node") -Force
            Remove-Item $zip -Force
        } catch {
            Write-Host "Node.js could not be downloaded: $($_.Exception.Message)"
            Write-Host "Install it from https://nodejs.org and run this again."
            exit 1
        }
        $local = Get-ChildItem (Join-Path $tools "node") -Filter node.exe -Recurse | Select-Object -First 1
    }
    $env:Path = $local.DirectoryName + ";" + $env:Path
}

$installed = Join-Path $ui "node_modules\.package-lock.json"
if(-not (Test-Path $installed) -or (Get-Item (Join-Path $ui "package.json")).LastWriteTime -gt (Get-Item $installed).LastWriteTime){
    Write-Host "Installing the Sneo UI packages..."
    $env:PUPPETEER_SKIP_DOWNLOAD = "true"
    Push-Location $ui
    & npm.cmd install --no-audit --no-fund
    $failed = $LASTEXITCODE -ne 0
    if($failed){
        Write-Host "Clearing the browser download cache and trying once more..."
        Remove-Item -Recurse -Force (Join-Path $env:USERPROFILE ".cache\puppeteer") -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force (Join-Path $ui "node_modules\puppeteer") -ErrorAction SilentlyContinue
        & npm.cmd install --no-audit --no-fund
        $failed = $LASTEXITCODE -ne 0
    }
    Pop-Location
    if($failed){
        Write-Host "npm install failed, check the internet connection and run this again."
        exit 1
    }
    if(Test-Path $installed){
        (Get-Item $installed).LastWriteTime = Get-Date
    }
}

$page = Join-Path $ui "dist\index.html"
$newest = Get-ChildItem (Join-Path $ui "src") -Recurse -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if(-not (Test-Path $page) -or $newest.LastWriteTime -gt (Get-Item $page).LastWriteTime){
    Write-Host "Building the Sneo page..."
    Push-Location $ui
    & npm.cmd run build
    $failed = $LASTEXITCODE -ne 0
    Pop-Location
    if($failed){
        Write-Host "The Sneo page could not be built."
        exit 1
    }
}

$runtime = Join-Path $root "runtime"
if(-not (Test-Path (Join-Path $runtime "nvrtc64_120_0.dll"))){
    Write-Host "Downloading the NVIDIA kernel compiler (NVRTC)..."
    try {
        $info = Invoke-RestMethod "https://pypi.org/pypi/nvidia-cuda-nvrtc-cu12/12.9.86/json"
        $wheel = $info.urls | Where-Object { $_.filename -like "*win_amd64.whl" } | Select-Object -First 1
        $zip = Join-Path $env:TEMP "sneo_nvrtc.zip"
        $unpacked = Join-Path $env:TEMP "sneo_nvrtc"
        Invoke-WebRequest $wheel.url -OutFile $zip -UseBasicParsing
        Expand-Archive $zip $unpacked -Force
        New-Item -ItemType Directory -Force $runtime | Out-Null
        Copy-Item (Join-Path $unpacked "nvidia\cuda_nvrtc\bin\nvrtc64_120_0.dll") $runtime -Force
        Copy-Item (Join-Path $unpacked "nvidia\cuda_nvrtc\bin\nvrtc-builtins64_129.dll") $runtime -Force
        Remove-Item $zip -Force
        Remove-Item $unpacked -Recurse -Force
    } catch {
        Write-Host "NVRTC could not be downloaded ($($_.Exception.Message)), the engine will run on the CPU."
    }
}
$nvidia = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*NVIDIA*" }
if($nvidia -and -not (Test-Path (Join-Path $runtime "cublasLt64_12.dll"))){
    Write-Host "Downloading the NVIDIA matrix library (cuBLAS, about 530 MB) for faster prompt reading..."
    try {
        $info = Invoke-RestMethod "https://pypi.org/pypi/nvidia-cublas-cu12/12.9.2.10/json"
        $wheel = $info.urls | Where-Object { $_.filename -like "*win_amd64.whl" } | Select-Object -First 1
        $zip = Join-Path $env:TEMP "sneo_cublas.zip"
        $unpacked = Join-Path $env:TEMP "sneo_cublas"
        Invoke-WebRequest $wheel.url -OutFile $zip -UseBasicParsing
        if((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower() -ne $wheel.digests.sha256){ throw "the download is damaged" }
        Expand-Archive $zip $unpacked -Force
        New-Item -ItemType Directory -Force $runtime | Out-Null
        Copy-Item (Join-Path $unpacked "nvidia\cublas\bin\cublasLt64_12.dll") $runtime -Force
        Copy-Item (Join-Path $unpacked "nvidia\cublas\bin\cublas64_12.dll") $runtime -Force
        Remove-Item $zip -Force
        Remove-Item $unpacked -Recurse -Force
    } catch {
        Write-Host "cuBLAS could not be downloaded ($($_.Exception.Message)), prompts are read with the engine's own kernels."
    }
}

$master = Join-Path $root "master.exe"
$worker = Join-Path $root "worker.exe"
$sources = @(Get-Item (Join-Path $root "master.cpp"), (Join-Path $root "worker.cpp") -ErrorAction SilentlyContinue) + @(Get-ChildItem (Join-Path $root "src") -Filter *.h -File -ErrorAction SilentlyContinue)
$sourceNewest = $sources | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$engineStale = $sourceNewest -and (Test-Path $master) -and (Test-Path $worker) -and ($sourceNewest.LastWriteTime -gt (Get-Item $master).LastWriteTime -or $sourceNewest.LastWriteTime -gt (Get-Item $worker).LastWriteTime)
if(-not (Test-Path $master) -or -not (Test-Path $worker) -or $engineStale){
    if(-not (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe")){
        if(-not (Test-Path $master)){
            Write-Host "master.exe is missing and Visual Studio with C++ is not installed to build it."
            exit 1
        }
        Write-Host "The engine code is newer than master.exe but Visual Studio with C++ is not installed, keeping the current engine."
    } else {
        foreach($running in Get-Process master, worker -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $master -or $_.Path -eq $worker }){
            Write-Host "Stopping the running engine to update it..."
            Stop-Process -Id $running.Id -Force -Confirm:$false
        }
        foreach($i in 1..40){
            if(-not (Get-Process master, worker -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $master -or $_.Path -eq $worker })){ break }
            Start-Sleep -Milliseconds 500
        }
        Write-Host "Building the engine..."
        & cmd /c "`"$root\build.bat`" nopause"
        if(-not (Test-Path $master)){
            Write-Host "The engine could not be built."
            exit 1
        }
    }
}

foreach($running in Get-Process master -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $master }){
    if((Get-Item $master).LastWriteTime -gt $running.StartTime){
        Write-Host "The engine was rebuilt, restarting it..."
        Stop-Process -Id $running.Id -Force -Confirm:$false
        Start-Sleep -Milliseconds 800
    }
}

$serverNewest = Get-ChildItem (Join-Path $ui "server") -Filter *.js -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
foreach($held in Get-NetTCPConnection -LocalPort 3001 -State Listen -ErrorAction SilentlyContinue){
    $owner = Get-Process -Id $held.OwningProcess -ErrorAction SilentlyContinue
    $line = (Get-CimInstance Win32_Process -Filter "ProcessId=$($held.OwningProcess)").CommandLine
    if($owner -and $line -like "*server/server.js*" -and $serverNewest.LastWriteTime -gt $owner.StartTime){
        Write-Host "The Sneo server was updated, restarting it..."
        Stop-Process -Id $owner.Id -Force -Confirm:$false
        Start-Sleep -Milliseconds 800
    }
}

if(-not (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)){
    Start-Process $master -ArgumentList "8000","-","-","modelos" -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $root "master.log") -RedirectStandardError (Join-Path $root "master.log.err")
}

function Test-Sneo {
    try { return (Invoke-WebRequest "http://127.0.0.1:3001/api/sneo" -UseBasicParsing -TimeoutSec 2).Content -like '*"sneo":true*' } catch { return $false }
}

if(-not (Test-Sneo)){
    Start-Process (Get-Command node.exe).Source -ArgumentList "server/server.js" -WorkingDirectory $ui -WindowStyle Hidden -RedirectStandardOutput (Join-Path $ui "server.log") -RedirectStandardError (Join-Path $ui "server.log.err")
}

$app = Join-Path $ui "src-tauri\target\release\sneo.exe"
$appNewest = Get-ChildItem (Join-Path $ui "src-tauri\src"), (Join-Path $ui "src-tauri\Cargo.toml"), (Join-Path $ui "src-tauri\tauri.conf.json") -Recurse -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$appStale = (Test-Path $app) -and $appNewest -and $appNewest.LastWriteTime -gt (Get-Item $app).LastWriteTime
if(-not (Test-Path $app) -or $appStale){
    $cargoHome = Join-Path $env:USERPROFILE ".cargo\bin"
    if(-not (Get-Command cargo.exe -ErrorAction SilentlyContinue) -and (Test-Path (Join-Path $cargoHome "cargo.exe"))){
        $env:Path = $cargoHome + ";" + $env:Path
    }
    if(-not (Get-Command cargo.exe -ErrorAction SilentlyContinue)){
        Write-Host "Rust is needed once to build the desktop app, downloading it..."
        try {
            $setup = Join-Path $env:TEMP "rustup-init.exe"
            Invoke-WebRequest "https://win.rustup.rs/x86_64" -OutFile $setup -UseBasicParsing
            & $setup -y --default-toolchain stable --profile minimal
            $env:Path = $cargoHome + ";" + $env:Path
        } catch {
            Write-Host "Rust could not be installed: $($_.Exception.Message)"
        }
    }
    if(Get-Command cargo.exe -ErrorAction SilentlyContinue){
        Write-Host "Building the desktop app, it takes a few minutes the first time..."
        Push-Location $ui
        & npm.cmd run tauri build -- --no-bundle
        Pop-Location
    }
}

if(Test-Path $app){
    foreach($i in 1..30){
        if(Test-Sneo){ break }
        Start-Sleep -Milliseconds 500
    }
    Start-Process $app
    exit 0
}

Write-Host "The desktop app could not be built here, opening Sneo in the browser instead."
foreach($i in 1..30){
    if(Test-Sneo){ break }
    Start-Sleep 1
}
Start-Process "http://127.0.0.1:3001"
exit 0

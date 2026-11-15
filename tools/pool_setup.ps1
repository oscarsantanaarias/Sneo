$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if(-not $admin){
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

Write-Host "Preparing this PC for the Sneo pool"
Write-Host ""
foreach($network in Get-NetConnectionProfile | Where-Object { $_.IPv4Connectivity -eq "Internet" -or $_.IPv4Connectivity -eq "LocalNetwork" }){
    if($network.NetworkCategory -eq "Private"){
        Write-Host "$($network.Name) is already a private network"
        continue
    }
    Set-NetConnectionProfile -InterfaceIndex $network.InterfaceIndex -NetworkCategory Private
    Write-Host "$($network.Name) is now a private network"
}

$root = Split-Path -Parent $PSScriptRoot
foreach($program in @("worker.exe", "master.exe")){
    $path = Join-Path $root $program
    foreach($blocked in Get-NetFirewallRule -Direction Inbound -Action Block -ErrorAction SilentlyContinue | Where-Object { ($_ | Get-NetFirewallApplicationFilter).Program -like "*\$program" }){
        Remove-NetFirewallRule -Name $blocked.Name
        Write-Host "Removed a Windows rule that blocked $program ($($blocked.DisplayName))"
    }
    if(-not (Get-NetFirewallRule -DisplayName "Sneo $program" -ErrorAction SilentlyContinue)){
        New-NetFirewallRule -DisplayName "Sneo $program" -Direction Inbound -Program $path -Action Allow -Profile Private | Out-Null
        Write-Host "$program may receive connections on private networks"
    }
}

foreach($other in Get-CimInstance -Namespace root/SecurityCenter2 -ClassName FirewallProduct -ErrorAction SilentlyContinue){
    Write-Host "Another firewall is active: $($other.displayName). Allow worker.exe and master.exe (ports 9090 and 9097) in it too."
}

foreach($rule in @(@{Name = "Sneo pool door"; Port = 9097}, @{Name = "Sneo worker"; Port = 9090})){
    if(Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue){
        Write-Host "Port $($rule.Port) ($($rule.Name)) was already open"
        continue
    }
    New-NetFirewallRule -DisplayName $rule.Name -Direction Inbound -Protocol TCP -LocalPort $rule.Port -Action Allow -Profile Private | Out-Null
    Write-Host "Port $($rule.Port) ($($rule.Name)) is open on private networks"
}

Write-Host ""
foreach($address in Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -match '^(192\.168|10\.|172\.(1[6-9]|2[0-9]|3[01]))' -and $_.InterfaceAlias -notmatch 'VirtualBox|vEthernet|VMware' }){
    Write-Host "Address of this PC on $($address.InterfaceAlias): $($address.IPAddress)"
}
Write-Host ""
Read-Host "Done, press Enter to close"

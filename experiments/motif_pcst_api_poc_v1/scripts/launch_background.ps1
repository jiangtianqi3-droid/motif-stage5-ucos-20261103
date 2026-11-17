[CmdletBinding()]
param([ValidateSet('all','pilot')][string]$Stage = 'pilot')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$windowsDrivePath = $root.Replace('\', '/')
$wslRoot = (& wsl.exe -d Ubuntu-24.04 -- wslpath -a -- "$windowsDrivePath" | Out-String).Trim()
if (!$wslRoot -or $wslRoot -notmatch '^/mnt/[a-z]/[a-zA-Z0-9/_.-]+$') {
    throw "Could not resolve a safe WSL path for the experiment"
}
$logDir = Join-Path $root 'results/logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$worker = Start-Process -FilePath 'wsl.exe' -ArgumentList @(
    '-d','Ubuntu-24.04','--','bash',"$wslRoot/scripts/run.sh",'-m','src.pipeline','--stage',$Stage
) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "$Stage-$stamp.log") `
  -RedirectStandardError (Join-Path $logDir "$Stage-$stamp.err.log") -PassThru
[pscustomobject]@{ WindowsPid=$worker.Id; Stage=$Stage; Directory=$root; LogDirectory=$logDir }

[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$winPath=$root.Replace('\','/')
$work=(& wsl.exe -d Ubuntu-24.04 -- wslpath -a -- "$winPath" | Out-String).Trim()
if ($work -notmatch '^/mnt/[a-z]/[a-zA-Z0-9/_.-]+$') { throw 'Invalid WSL path' }
$log=Join-Path $root 'logs'
New-Item -ItemType Directory -Path $log -Force | Out-Null
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$job=Start-Process -FilePath wsl.exe -ArgumentList @(
  '-d','Ubuntu-24.04','--','bash',"$work/scripts/run.sh",'-m','p1.run','--mode','full'
) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $log "p1-$stamp.log") `
  -RedirectStandardError (Join-Path $log "p1-$stamp.err.log") -PassThru
[pscustomobject]@{WindowsPid=$job.Id; Directory=$root; LogDirectory=$log}

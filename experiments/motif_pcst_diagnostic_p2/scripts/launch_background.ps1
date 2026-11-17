$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$win=$root.Replace('\','/')
$work=(& wsl.exe -d Ubuntu-24.04 -- wslpath -a -- "$win" | Out-String).Trim()
if ($work -notmatch '^/mnt/[a-z]/[a-zA-Z0-9/_.-]+$') { throw 'Invalid WSL path' }
$log=Join-Path $root 'logs'
New-Item -ItemType Directory -Path $log -Force | Out-Null
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$out=Join-Path $log "p2-$stamp.log"
$err=Join-Path $log "p2-$stamp.err.log"
$job=Start-Process -FilePath wsl.exe -ArgumentList @(
 '-d','Ubuntu-24.04','--','bash',"$work/scripts/run.sh",'-m','p2.run','--mode','full'
) -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
[pscustomobject]@{WindowsPid=$job.Id; Directory=$root; Stdout=$out; Stderr=$err}

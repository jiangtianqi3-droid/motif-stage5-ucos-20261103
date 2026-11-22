$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$status=Join-Path $root 'outputs/status.json'
if(Test-Path -LiteralPath $status){
  $current=Get-Content -LiteralPath $status -Raw | ConvertFrom-Json
  if($current.state -eq 'running'){
    Write-Output 'Checking existing worker lock before launch; the WSL entry rejects a duplicate.'
  }
}
$windowsPath=$root.Replace('\','/')
$work=(& wsl.exe -d Ubuntu-24.04 -- wslpath -a -- "$windowsPath" | Out-String).Trim()
if($work -notmatch '^/mnt/[a-z]/[a-zA-Z0-9/_.-]+$'){throw 'Invalid WSL experiment path'}
$logs=Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$out=Join-Path $logs "formal-$stamp.log"
$err=Join-Path $logs "formal-$stamp.err.log"
$start=Get-Date
$job=Start-Process -FilePath wsl.exe -ArgumentList @(
 '-d','Ubuntu-24.04','--','bash',"$work/scripts/run.sh",'-m','v21.run'
) -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
$manifest=[pscustomobject]@{WindowsPid=$job.Id;StartTime=$start.ToString('o');Directory=$root;Stdout=$out;Stderr=$err}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'manifests/background_launch.json') -Encoding utf8
$manifest

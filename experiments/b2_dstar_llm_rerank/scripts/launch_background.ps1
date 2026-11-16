$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$LinuxRoot = '/mnt/d/motif-phase4-work/b2-dstar-local-llm-rerank/experiments/b2_dstar_llm_rerank'
$Process = Start-Process -FilePath 'wsl.exe' -ArgumentList @('-e', 'bash', "$LinuxRoot/scripts/supervisor.sh") -WindowStyle Hidden -PassThru
Write-Output "Background process: $($Process.Id)"
Write-Output "Progress: $Root/status.json"

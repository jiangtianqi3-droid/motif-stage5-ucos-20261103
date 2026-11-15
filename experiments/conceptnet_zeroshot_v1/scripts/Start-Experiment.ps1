$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $experimentRoot 'results/status.json'
$smokeState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
if ($smokeState.stage -ne 'smoke_passed') { throw 'A successful smoke test is required before launching.' }
$logPath = Join-Path $experimentRoot 'logs/full.log'
$errorPath = Join-Path $experimentRoot 'logs/full.stderr.log'
if ((Test-Path -LiteralPath $logPath) -or (Test-Path -LiteralPath $errorPath)) { throw 'Existing full-run logs must not be overwritten. Inspect the existing run first.' }
$linuxScript = '/mnt/d/motif-phase4-work/conceptnet-zeroshot-boost/experiments/conceptnet_zeroshot_v1/scripts/run.sh'
$started = Get-Date
$worker = Start-Process -FilePath 'wsl.exe' -ArgumentList @('-e', 'bash', $linuxScript, 'scripts/run_conceptnet_boost.py') -WindowStyle Hidden -RedirectStandardOutput $logPath -RedirectStandardError $errorPath -PassThru
$launch = [ordered]@{ windows_pid=$worker.Id; started_at=$started.ToString('o'); log=$logPath; stderr=$errorPath; status=$statePath; estimate_minutes='30-50' }
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentRoot 'results/launch.json') -Encoding utf8
for ($attempt=0; $attempt -lt 28; $attempt++) {
    Start-Sleep -Seconds 2
    $worker.Refresh()
    if ($worker.HasExited) { throw "Background launcher exited; inspect $errorPath" }
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.mode -eq 'full' -and $state.stage -ne 'smoke_passed') {
        if ($state.stage -eq 'failed') { throw "Experiment failed: $($state.error)" }
        $launch['linux_pid']=$state.pid
        $launch['first_status']=$state.stage
        $launch['confirmed_at']=(Get-Date).ToString('o')
        $launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentRoot 'results/launch.json') -Encoding utf8
        $launch | ConvertTo-Json
        exit 0
    }
}
throw "Process launched but first status not yet confirmed. Windows PID: $($worker.Id)"

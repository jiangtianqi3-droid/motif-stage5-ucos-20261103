param([ValidatePattern('^[0-9a-f]{20}$')][string]$ResumeFrom)
$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
$statusPath = Join-Path $experimentRoot 'results\status.json'
$launchTime = Get-Date
$windowsEntry = Join-Path $PSScriptRoot 'background_entry.sh'
$linuxEntry = & wsl.exe -e wslpath -a -u $windowsEntry
if ($LASTEXITCODE -ne 0 -or !$linuxEntry) { throw 'Could not resolve the experiment launcher in WSL.' }
$workerArgs = @('-e', 'bash', ('"' + $linuxEntry + '"'))
if ($ResumeFrom) { $workerArgs += @('--resume-from', $ResumeFrom) }
$worker = Start-Process -FilePath 'wsl.exe' -ArgumentList $workerArgs -WindowStyle Hidden -PassThru
# Only the startup handshake is observed. Never wait for evaluation completion.
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Seconds 1
    if (Test-Path -LiteralPath $statusPath) {
        try { $state = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json } catch { continue }
        $stamp = (Get-Item -LiteralPath $statusPath).LastWriteTime
        if ($state.mode -eq 'full' -and $stamp -ge $launchTime -and $state.full_run_started) {
            if ($state.stage -eq 'failed') { throw "Background startup failed: $($state.error)" }
            [pscustomobject]@{
                windowsPid = $worker.Id
                linuxPid = $state.pid
                stage = $state.stage
                status = $statusPath
                log = (Join-Path $experimentRoot 'logs\full.log')
                startedAt = $state.updated_at
            } | ConvertTo-Json
            exit 0
        }
    }
    if ($worker.HasExited) { throw "Background host exited before startup acknowledgement: $($worker.ExitCode)" }
}
throw 'No startup acknowledgement within 30 seconds; inspect launch status before retrying.'

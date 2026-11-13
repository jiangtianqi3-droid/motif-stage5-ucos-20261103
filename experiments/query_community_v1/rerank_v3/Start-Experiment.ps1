param([switch]$SkipLlm)
$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$launchRecord = Join-Path $experimentRoot 'launch.json'
if (Test-Path -LiteralPath $launchRecord) {
    $previousLaunch = Get-Content -LiteralPath $launchRecord -Raw | ConvertFrom-Json
    $previousProcess = Get-Process -Id $previousLaunch.pid -ErrorAction SilentlyContinue
    if ($previousProcess -and $previousProcess.StartTime.ToUniversalTime().ToString('o') -eq $previousLaunch.processStartUtc) {
        Write-Output "Experiment already running. Status: $experimentRoot/workflow_status.json"
        exit 0
    }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutPath = Join-Path $experimentRoot "logs/launcher_$stamp.log"
$stderrPath = Join-Path $experimentRoot "logs/launcher_${stamp}_stderr.log"
$launchArguments = @('-d', 'Ubuntu-24.04', '--', 'bash', '/mnt/d/motif-phase4-work/route-a-repair-20261112/run_all.sh')
if ($SkipLlm) { $launchArguments += '--skip-llm' }
$workerProcess = Start-Process -FilePath "$env:WINDIR/System32/wsl.exe" -ArgumentList $launchArguments -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
@{
    pid = $workerProcess.Id
    processStartUtc = $workerProcess.StartTime.ToUniversalTime().ToString('o')
    launchedAt = (Get-Date).ToString('o')
    stdout = $stdoutPath
    stderr = $stderrPath
    skipLlm = [bool]$SkipLlm
} | ConvertTo-Json | Set-Content -LiteralPath $launchRecord -Encoding UTF8
Write-Output "Experiment started in background. Status: $experimentRoot/workflow_status.json"
Write-Output "Report appears after verified completion: $experimentRoot/report.md"

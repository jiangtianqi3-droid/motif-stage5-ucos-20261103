$statusPath = Join-Path $PSScriptRoot 'workflow_status.json'
if (-not (Test-Path -LiteralPath $statusPath)) {
    Write-Output 'Experiment has not written a status yet.'
    exit 0
}
$state = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
$state | Select-Object state, stage, updated, completed_steps, log, error, report | Format-List

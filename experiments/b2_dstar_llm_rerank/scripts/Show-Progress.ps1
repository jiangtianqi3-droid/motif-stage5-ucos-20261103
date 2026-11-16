$Root = Split-Path -Parent $PSScriptRoot
while ($true) {
    if (Test-Path -LiteralPath "$Root/status.json") {
        $Status = Get-Content -LiteralPath "$Root/status.json" -Raw | ConvertFrom-Json
        $Percent = if ($Status.total -gt 0) { [math]::Min(100, 100 * $Status.done / $Status.total) } else { 0 }
        Write-Progress -Activity "$($Status.graph) | $($Status.stage) | seed $($Status.seed)" -Status "$($Status.done)/$($Status.total); ETA $([int]$Status.eta_seconds)s; API $($Status.api_calls)" -PercentComplete $Percent
        if ($Status.state -in @('failed','completed')) { $Status | Format-List; break }
    }
    Start-Sleep -Seconds 2
}

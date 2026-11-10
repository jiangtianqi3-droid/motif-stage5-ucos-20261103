function Read-ProgressJson {
    param([string]$Path)
    try { return ([IO.File]::ReadAllText($Path, [Text.Encoding]::UTF8) | ConvertFrom-Json) }
    catch { return $null }
}

function Get-ProgressView {
    param([string]$RunDirectory, $Protocol, $Status)
    $finished = Read-ProgressJson (Join-Path $RunDirectory 'completion.json')
    $isComplete = $finished -and $finished.state -eq 'complete'
    $weightSum = 0.0; $weighted = 0.0; $completed = 0; $units = 0
    foreach ($dataset in $Protocol.datasets) {
        $audit = $null
        foreach ($family in $Protocol.families) {
            $audit = Read-ProgressJson (Join-Path $RunDirectory "$dataset\$family\split_leakage_audit.json")
            if ($audit) { break }
        }
        # Relative work is estimated from the fixed number of query/method records.
        $weight = 1.0
        if ($audit) {
            $weight = 8.0 * ([double]$audit.test_facts + [Math]::Min([double]$Protocol.validation_max_facts, [double]$audit.valid_facts))
        }
        foreach ($family in $Protocol.families) {
            $units++; $work = Join-Path $RunDirectory "$dataset\$family"
            $fraction = 0.0
            $done = Read-ProgressJson (Join-Path $work 'completion.json')
            if ($done -and $done.state -eq 'complete') {
                $fraction = 1.0; $completed++
            } else {
                if (Test-Path -LiteralPath (Join-Path $work 'retrieval_completion.json')) { $fraction = .94 }
                if (Test-Path -LiteralPath (Join-Path $work 'selected_structure.json')) { $fraction = .98 }
                if (Test-Path -LiteralPath (Join-Path $work 'selected_llm.json')) { $fraction = .99 }
                if ($Status -and $Status.dataset -eq $dataset -and $Status.family -eq $family) {
                    $ratio = 0.0
                    if ($Status.total -and [double]$Status.total -gt 0) {
                        $ratio = [Math]::Max(0.0, [Math]::Min(1.0, [double]$Status.done / [double]$Status.total))
                    }
                    switch ($Status.stage) {
                        'dataset_start' { $fraction = [Math]::Max($fraction, .005) }
                        'frozen_inference' { $fraction = [Math]::Max($fraction, .01 + .01 * $ratio) }
                        'structural_evidence' { $fraction = [Math]::Max($fraction, .02 + .92 * $ratio) }
                        'validation_selection' {
                            $methodIndex = @('R1','R2','R3','R4').IndexOf([string]$Status.method)
                            $fraction = [Math]::Max($fraction, .94 + .01 * ($methodIndex + 1))
                        }
                        'llm_evidence' { $fraction = [Math]::Max($fraction, .985) }
                        'dataset_complete' { $fraction = 1.0 }
                    }
                }
            }
            $weightSum += $weight; $weighted += $weight * $fraction
        }
    }
    $overall = 0.0
    if ($weightSum -gt 0) { $overall = [Math]::Min(99.5, 100.0 * $weighted / $weightSum) }
    if ($isComplete) { $overall = 100.0 }
    $stagePercent = 0.0; $indeterminate = $true
    if ($Status -and $Status.total -and [double]$Status.total -gt 0) {
        $stagePercent = [Math]::Min(100.0, [Math]::Max(0.0, 100.0 * [double]$Status.done / [double]$Status.total))
        $indeterminate = $false
    }
    if ($isComplete -or ($Status -and $Status.stage -eq 'dataset_complete')) {
        $stagePercent = 100.0; $indeterminate = $false
    }
    $stageNames = @{
        preflight='准备'; dataset_start='准备'; frozen_inference='模型评分'
        structural_evidence='证据提取'; validation_selection='验证选参'
        llm_evidence='证据核验'; dataset_complete='本组完成'; finished='已完成'; exception='中断'
    }
    $datasetNames = @{
        FB15k237Inductive_v1='FB15k237'; WN18RRInductive_v1='WN18RR'
        NELLInductive_v1='NELL'; FBIngram_25='FBIngram'
    }
    $caption = '正在准备'
    if ($Status -and $Status.dataset) {
        $name = $datasetNames[[string]$Status.dataset]
        if (-not $name) { $name = [string]$Status.dataset }
        $stageName = $stageNames[[string]$Status.stage]
        if (-not $stageName) { $stageName = '处理中' }
        $caption = "$name  /  $($Status.family)  ·  $stageName"
    }
    if ($isComplete) { $caption = '实验已完成' }
    elseif ($completed -eq $units -and $units -gt 0) { $caption = '正在汇总报告' }
    return [pscustomobject]@{
        Overall=$overall; Current=$stagePercent; Indeterminate=$indeterminate
        Caption=$caption; Complete=[bool]$isComplete; Completed=$completed; Units=$units
    }
}

function Get-ExperimentHealth {
    param([string]$State, [bool]$Complete, $Alive, [int]$MissingCount, [double]$AgeSeconds, [double]$StaleMinutes=15)
    if ($Complete) { return 'complete' }
    if ($State -eq 'failed') { return 'interrupted' }
    if ($null -ne $Alive -and -not [bool]$Alive) {
        if ($MissingCount -ge 2) { return 'interrupted' }
        return 'checking'
    }
    if ($null -eq $Alive) { return 'checking' }
    if ($AgeSeconds -gt $StaleMinutes * 60) { return 'stalled' }
    return 'running'
}

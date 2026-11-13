function Read-ProgressJson {
    param([string]$Path)
    try {
        # WSL publishes each status via rename. Readers must allow that replacement.
        $share = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
        $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, $share)
        try {
            $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8, $true)
            try { $content = $reader.ReadToEnd() }
            finally { $reader.Dispose() }
        } finally { $stream.Dispose() }
        $content | ConvertFrom-Json
    }
    catch { $null }
}

function Convert-ExperimentPath {
    param([string]$Path)
    if ($Path -match '^/mnt/([a-z])/(.*)$') { return ('{0}:/{1}' -f $Matches[1], $Matches[2]) }
    return $Path
}

function Get-UpdateTime {
    param([string]$Path)
    if ($Path -and (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return [DateTimeOffset]::new([IO.File]::GetLastWriteTimeUtc($Path))
    }
    return [DateTimeOffset]::MinValue
}

function Get-TrainingView {
    param([string]$Root, $Workflow, $View)
    $protocol = Read-ProgressJson (Join-Path $Root 'training/protocol.json')
    if (-not $protocol -or [int]$protocol.steps -le 0) { return }
    $stepsPerSeed = [int]$protocol.steps
    $steps = @{}
    foreach ($seed in $protocol.seeds) {
        $tag = "T4D_strict_s$seed"
        $saved = Read-ProgressJson (Join-Path $Root "training/results/training_${tag}.json")
        $steps[$tag] = if ($saved -and [int]$saved.steps -eq $stepsPerSeed) { $stepsPerSeed } else { 0 }
    }
    $activeTag = ''
    $validating = $false
    $log = Convert-ExperimentPath $Workflow.log
    $currentAttempt = $false
    if ($log -and (Test-Path -LiteralPath $log -PathType Leaf)) {
        foreach ($line in (Get-Content -LiteralPath $log -Encoding UTF8 -Tail 128 -ErrorAction SilentlyContinue)) {
            if ($line -match '^--- (\S+) start train ---$') {
                $currentAttempt = [DateTimeOffset]::Parse($Matches[1]) -ge [DateTimeOffset]::Parse($Workflow.updated)
                continue
            }
            if (-not $currentAttempt -or $line -notmatch '^\{') { continue }
            try { $event = $line | ConvertFrom-Json } catch { continue }
            if (-not $event.tag -or -not $steps.ContainsKey([string]$event.tag)) { continue }
            switch ($event.event) {
                train_begin { $activeTag = [string]$event.tag; $steps[$activeTag] = [Math]::Max($steps[$activeTag], [int]$event.start - 1); $validating = $false }
                train_step { $activeTag = [string]$event.tag; $steps[$activeTag] = [int]$event.step; $validating = $steps[$activeTag] -in $protocol.validation_steps }
                source_validation { $validating = $false }
                train_complete { $steps[[string]$event.tag] = $stepsPerSeed; $activeTag = ''; $validating = $false }
            }
        }
    }
    $View.CurrentTotal = $stepsPerSeed * $steps.Count
    $View.CurrentDone = [int](($steps.Values | Measure-Object -Sum).Sum)
    $View.Counter = "$($View.CurrentDone) / $($View.CurrentTotal) 步"
    if ($activeTag) {
        $seed = $activeTag -replace '^T4D_strict_s', ''
        $View.Detail = "种子 $seed · $($steps[$activeTag]) / $stepsPerSeed 步"
        if ($validating) { $View.Detail += ' · 源验证' }
    }
}

function Get-EvaluationView {
    param([string]$Root, $Workflow, $View)
    $activeFile = Join-Path $Root 'results/active_run.json'
    if ((Get-UpdateTime $activeFile) -lt [DateTimeOffset]::Parse($Workflow.updated)) { return }
    $active = Read-ProgressJson $activeFile
    if (-not $active.run) { return }
    $run = Convert-ExperimentPath $active.run
    $protocol = Read-ProgressJson (Join-Path $run 'protocol.json')
    $smoke = $Workflow.stage -eq 'gpu_smoke'
    if (-not $protocol -or [bool]$protocol.smoke -ne $smoke) { return }
    $terminal = if ($smoke) { 'smoke_complete' } else { 'complete' }
    $completed = 0
    foreach ($dataset in $protocol.datasets) {
        foreach ($family in $protocol.families) {
            $record = Read-ProgressJson (Join-Path $run "$dataset/$family/completion.json")
            if ($record.state -eq $terminal) { $completed++ }
        }
    }
    $total = @($protocol.datasets).Count * @($protocol.families).Count
    $View.Detail = "评估组 $completed / $total"
    $View.Run = Split-Path -Leaf $run
    $status = Read-ProgressJson (Join-Path $run 'status.json')
    if (-not $status.at -or [DateTimeOffset]::Parse($status.at) -lt [DateTimeOffset]::Parse($Workflow.updated)) { return }
    $at = [DateTimeOffset]::Parse($status.at)
    if ($at -gt $View.Updated) { $View.Updated = $at }
    if ($status.state -eq 'failed') { $View.State = 'failed'; $View.Error = [string]$status.error; return }
    if ($status.stage -eq 'finished') { $View.Detail += ' · 正在交接核验'; return }
    $labels = @{ dataset_start='准备'; frozen_inference='本轮待算评分'; structural_evidence='证据提取'; validation_selection='验证选参'; llm_evidence='LLM 核验'; dataset_complete='本组完成' }
    $label = $labels[[string]$status.stage]
    if ($status.dataset) { $View.Detail += " · $($status.dataset) / $($status.family) · $label" }
    if ($status.split) { $View.Detail += $(if ($status.split -eq 'valid') { '（验证）' } else { '（测试）' }) }
    if ([int]$status.total -gt 0) {
        $View.CurrentDone = [int]$status.done
        $View.CurrentTotal = [int]$status.total
        $View.Counter = "$($View.CurrentDone) / $($View.CurrentTotal)"
    } elseif ($status.stage -eq 'llm_evidence') {
        $View.Counter = "已处理 $($status.processed)"
    } elseif ($status.stage -eq 'validation_selection' -and $status.method) {
        $View.Counter = [string]$status.method
    }
}

function Get-ProgressView {
    param([string]$Root, $Workflow)
    $phaseNames = @{ tests='代码测试'; cached_integration='缓存链路检查'; train='三种子训练'; training_verification='训练隔离核验'; gpu_smoke='小规模 GPU 检查'; full_evaluation='全量评估'; final_verification='结果核验'; verified='实验已完成'; code_checks_complete='代码检查已通过' }
    $plan = $null
    if ($Workflow.workflow -match '^[0-9a-f]{16}$') {
        $plan = Read-ProgressJson (Join-Path $Root "results/workflow_plan_$($Workflow.workflow).json")
    }
    $planned = if ($Workflow.check_only) { 2 } elseif ($plan.steps) { @($plan.steps).Count } else { 7 }
    $completed = @($Workflow.completed_steps | Where-Object { $_ }).Count
    $receipt = Read-ProgressJson (Join-Path $Root 'repair_completion.json')
    $verified = $Workflow.state -eq 'complete' -and $receipt.state -eq 'complete' -and
        $receipt.workflow -eq $Workflow.workflow -and $receipt.started -eq $Workflow.started
    $updated = if ($Workflow.updated) { [DateTimeOffset]::Parse($Workflow.updated) } else { [DateTimeOffset]::Now }
    $logUpdated = Get-UpdateTime (Convert-ExperimentPath $Workflow.log)
    if ($logUpdated -gt $updated) { $updated = $logUpdated }
    $caption = $phaseNames[[string]$Workflow.stage]
    if (-not $caption) { $caption = '等待实验状态' }
    $view = [pscustomobject]@{
        Stage=[string]$Workflow.stage; State=[string]$Workflow.state; Complete=[bool]$verified; Error=[string]$Workflow.error
        PhasesDone=$completed; PhasesTotal=$planned; Caption=$caption; Detail=''; Counter=''
        CurrentDone=0; CurrentTotal=0; Updated=$updated; Run=''
    }
    switch ($Workflow.stage) {
        train { Get-TrainingView $Root $Workflow $view }
        gpu_smoke { Get-EvaluationView $Root $Workflow $view }
        full_evaluation { Get-EvaluationView $Root $Workflow $view }
    }
    if ($Workflow.state -eq 'complete' -and -not $verified) { $view.Caption = '正在保存完成记录' }
    return $view
}

function Get-ExperimentHealth {
    param($View, $Alive, [int]$MissingCount, [double]$StaleMinutes=15)
    if ($View.Complete) { return 'complete' }
    if ($View.State -eq 'validated') { return 'validated' }
    if ($View.State -eq 'failed') { return 'interrupted' }
    if (-not $View.State -or $View.State -eq 'complete') { return 'checking' }
    if ($null -ne $Alive -and -not $Alive -and $MissingCount -ge 2) { return 'interrupted' }
    if ($null -eq $Alive -or -not $Alive) { return 'checking' }
    if (([DateTimeOffset]::UtcNow - $View.Updated).TotalMinutes -gt $StaleMinutes) { return 'stalled' }
    return 'running'
}

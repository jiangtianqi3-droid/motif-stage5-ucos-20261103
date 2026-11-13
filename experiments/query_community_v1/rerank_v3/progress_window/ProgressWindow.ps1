param([string]$OutputDirectory=(Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'ProgressModel.ps1')
Add-Type -AssemblyName PresentationFramework,PresentationCore,WindowsBase,System.Windows.Forms,System.Drawing
[Windows.Media.RenderOptions]::ProcessRenderMode = [Windows.Interop.RenderMode]::SoftwareOnly

$monitorDirectory = Join-Path $OutputDirectory 'monitor'
[void][IO.Directory]::CreateDirectory($monitorDirectory)
$created = $false
$mutex = [Threading.Mutex]::new($true, 'Local\RouteARepair20261112Progress', [ref]$created)
if (-not $created) { $mutex.Dispose(); return }

[xml]$xaml = @'
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
 xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
 Title="路线 A 修复实验进度" Width="600" SizeToContent="Height" MinHeight="260"
 ResizeMode="CanMinimize" WindowStartupLocation="CenterScreen" Background="#FAFBFD"
 FontFamily="Microsoft YaHei UI" ShowInTaskbar="True">
 <StackPanel Margin="28,22,28,24">
  <Grid>
   <TextBlock Text="流程阶段 · 等权计数" FontSize="13" Foreground="#64748B" VerticalAlignment="Center"/>
   <TextBlock x:Name="OverallText" Text="--" FontFamily="Segoe UI" FontSize="30" FontWeight="SemiBold" Foreground="#142238" HorizontalAlignment="Right"/>
  </Grid>
  <ProgressBar x:Name="OverallBar" Margin="0,10,0,22" Height="10" Maximum="100" Foreground="#367BF5" Background="#E4EAF3" BorderThickness="0"/>
  <Grid>
   <TextBlock x:Name="StageText" Text="正在读取进度" FontSize="14" Foreground="#142238"/>
   <TextBlock x:Name="StageCounter" FontSize="13" Foreground="#536176" HorizontalAlignment="Right"/>
  </Grid>
  <TextBlock x:Name="StageDetail" Margin="0,5,0,0" TextWrapping="Wrap" FontSize="12" Foreground="#536176"/>
  <ProgressBar x:Name="StageBar" Margin="0,10,0,20" Height="6" Maximum="100" Foreground="#8CAFF5" Background="#E4EAF3" BorderThickness="0"/>
  <Grid>
   <StackPanel Orientation="Horizontal">
    <Ellipse x:Name="StateDot" Width="7" Height="7" Fill="#94A3B8" Margin="0,0,8,0"/>
    <TextBlock x:Name="StateText" Text="正在检查" FontSize="12" Foreground="#64748B"/>
   </StackPanel>
   <TextBlock x:Name="UpdatedText" FontSize="11" Foreground="#64748B" HorizontalAlignment="Right"/>
  </Grid>
 </StackPanel>
</Window>
'@

$window = [Windows.Markup.XamlReader]::Load([Xml.XmlNodeReader]::new($xaml))
foreach ($name in @('OverallText','OverallBar','StageText','StageCounter','StageDetail','StageBar','StateDot','StateText','UpdatedText')) {
    Set-Variable -Scope Script -Name $name -Value $window.FindName($name)
}
$tray = [Windows.Forms.NotifyIcon]::new()
$tray.Icon = [Drawing.SystemIcons]::Information
$tray.Text = '路线 A 修复实验进度'
$tray.Visible = $true
$script:lastWorkflow = $null
$script:launchIdentity = ''
$script:missing = 0
$script:lastNotice = ''
$script:lastWrite = [DateTimeOffset]::MinValue

function Get-LauncherHealth {
    param($Workflow)
    $launch = Read-ProgressJson (Join-Path $OutputDirectory 'launch.json')
    if (-not $launch.pid -or -not $launch.processStartUtc -or -not $Workflow.started) { return $null }
    $start = [DateTimeOffset]::Parse($launch.processStartUtc)
    $launchAge = ([DateTimeOffset]::Parse($Workflow.started) - $start).TotalSeconds
    if ($launchAge -lt 0 -or $launchAge -gt 120) { return $null }
    $identity = "$($launch.pid)|$start"
    if ($identity -ne $script:launchIdentity) { $script:launchIdentity = $identity; $script:missing = 0 }
    $process = Get-Process -Id ([int]$launch.pid) -ErrorAction SilentlyContinue
    $alive = $process -and $process.ProcessName -eq 'wsl' -and
        [Math]::Abs(($process.StartTime.ToUniversalTime() - $start.UtcDateTime).TotalMilliseconds) -lt 1
    if ($alive) { $script:missing = 0 } else { $script:missing++ }
    return [bool]$alive
}

function Show-ExperimentNotice {
    param([string]$Health)
    if ($script:lastNotice -eq $Health) { return }
    $script:lastNotice = $Health
    switch ($Health) {
        interrupted {
            $tray.ShowBalloonTip(6000, '实验已中断', '可查看实验状态文件和当前阶段日志。', [Windows.Forms.ToolTipIcon]::Error)
            [Media.SystemSounds]::Exclamation.Play()
        }
        stalled { $tray.ShowBalloonTip(6000, '实验进度暂未更新', '启动进程仍在运行，状态或日志已超过 15 分钟没有更新。', [Windows.Forms.ToolTipIcon]::Warning) }
        complete { $tray.ShowBalloonTip(6000, '实验已完成', '全量评估和结果核验已完成。', [Windows.Forms.ToolTipIcon]::Info) }
    }
}

function Update-ProgressWindow {
    try {
        $workflow = Read-ProgressJson (Join-Path $OutputDirectory 'workflow_status.json')
        if ($workflow) { $script:lastWorkflow = $workflow }
        $workflow = $script:lastWorkflow
        $view = Get-ProgressView $OutputDirectory $workflow
        $alive = Get-LauncherHealth $workflow
        $health = Get-ExperimentHealth $view $alive $script:missing
        $OverallBar.Value = 100.0 * $view.PhasesDone / $view.PhasesTotal
        $OverallText.Text = "$($view.PhasesDone) / $($view.PhasesTotal)"
        $StageText.Text = $view.Caption
        $StageDetail.Text = $view.Detail
        $StageDetail.Visibility = if ($view.Detail) { 'Visible' } else { 'Collapsed' }
        $StageCounter.Text = $view.Counter
        $StageBar.IsIndeterminate = $view.CurrentTotal -eq 0 -and $health -in @('running','checking','stalled')
        $StageBar.Value = if ($view.CurrentTotal -gt 0) { 100.0 * $view.CurrentDone / $view.CurrentTotal } elseif ($view.Complete) { 100 } else { 0 }
        $UpdatedText.Text = if ($workflow) { '更新 ' + $view.Updated.ToOffset([TimeSpan]::FromHours(8)).ToString('HH:mm:ss') } else { '' }
        $color = '#367BF5'; $dot = '#22A06B'; $label = '正常运行'
        switch ($health) {
            checking { $dot = '#D69B26'; $label = '正在确认运行状态' }
            stalled { $color = '#D69B26'; $dot = $color; $label = '进程仍在 · 进度暂未更新' }
            interrupted { $color = '#DC4B55'; $dot = $color; $label = '实验已中断' }
            complete { $color = '#22A06B'; $dot = $color; $label = '全部完成' }
            validated { $label = '代码检查通过' }
        }
        $brushes = [Windows.Media.BrushConverter]::new()
        $OverallBar.Foreground = $brushes.ConvertFromString($color)
        $StateDot.Fill = $brushes.ConvertFromString($dot)
        $StateText.Text = $label
        $StateText.ToolTip = if ($view.Error) { $view.Error } else { Convert-ExperimentPath $workflow.log }
        Show-ExperimentNotice $health
        if (([DateTimeOffset]::Now - $script:lastWrite).TotalSeconds -ge 6) {
            $state = [ordered]@{pid=$PID; workflow=$workflow.workflow; stage=$view.Stage; run=$view.Run; health=$health
                phases_done=$view.PhasesDone; phases_total=$view.PhasesTotal; current_done=$view.CurrentDone; current_total=$view.CurrentTotal
                caption=$view.Caption; detail=$view.Detail; counter=$view.Counter; checked_at=[DateTimeOffset]::Now.ToString('o')}
            [IO.File]::WriteAllText((Join-Path $monitorDirectory 'window_state.json'), ($state | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
            $script:lastWrite = [DateTimeOffset]::Now
        }
    } catch {
        $StateText.Text = '暂时无法读取 · 自动重试'
        $StateText.ToolTip = $_.Exception.Message
        [IO.File]::AppendAllText((Join-Path $monitorDirectory 'window_errors.log'), ([DateTimeOffset]::Now.ToString('o') + ' ' + $_.Exception.ToString() + [Environment]::NewLine))
    }
}

$timer = [Windows.Threading.DispatcherTimer]::new()
$timer.Interval = [TimeSpan]::FromSeconds(2)
$timer.Add_Tick({ Update-ProgressWindow })
$window.Add_Loaded({ Update-ProgressWindow; $timer.Start() })
$window.Add_ContentRendered({ $window.Topmost = $true; [void]$window.Activate(); $window.Topmost = $false })
$window.Add_Closed({ $timer.Stop(); $tray.Visible = $false; $tray.Dispose(); $mutex.ReleaseMutex(); $mutex.Dispose() })
[void]$window.ShowDialog()

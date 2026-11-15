$ErrorActionPreference = 'Stop'
$experiment = Split-Path -Parent $PSScriptRoot
$statusPath = Join-Path $experiment 'results\status.json'
$logPath = Join-Path $experiment 'logs\full.log'
$totalGraphs = 11
$created = $false
$mutex = [Threading.Mutex]::new($true, 'Local\Universal11GraphProgress', [ref]$created)
if (-not $created) { $mutex.Dispose(); exit }

try {
    Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase
    [Windows.Media.RenderOptions]::ProcessRenderMode = [Windows.Interop.RenderMode]::SoftwareOnly
    [xml]$xaml = @'
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="11 图实验进度" Width="530" SizeToContent="Height" ResizeMode="CanMinimize"
        WindowStartupLocation="CenterScreen" Background="#FAFBFD" FontFamily="Microsoft YaHei UI">
    <StackPanel Margin="28,22,28,22">
        <DockPanel LastChildFill="False">
            <TextBlock Text="图谱完成进度" VerticalAlignment="Center" FontSize="14" Foreground="#536176"/>
            <TextBlock x:Name="CountLabel" DockPanel.Dock="Right" FontFamily="Segoe UI" FontSize="28" Foreground="#142238"/>
        </DockPanel>
        <ProgressBar x:Name="OverallBar" Height="10" Margin="0,8,0,20" Foreground="#367BF5" Background="#E4EAF3" BorderThickness="0"/>
        <TextBlock x:Name="GraphLabel" FontSize="16" FontWeight="SemiBold" Foreground="#142238"/>
        <TextBlock x:Name="StageLabel" Margin="0,5,0,0" FontSize="13" Foreground="#536176" TextWrapping="Wrap"/>
        <ProgressBar x:Name="StageBar" Height="5" Margin="0,10,0,14" Foreground="#8CAFF5" Background="#E4EAF3" BorderThickness="0"/>
        <TextBlock x:Name="TimeLabel" FontSize="12" Foreground="#536176"/>
        <Button x:Name="OpenLog" Content="查看日志" HorizontalAlignment="Right" Padding="11,4" Margin="0,12,0,0"/>
    </StackPanel>
</Window>
'@
    $reader = [Xml.XmlNodeReader]::new($xaml)
    $window = [Windows.Markup.XamlReader]::Load($reader)
    foreach ($name in 'CountLabel','OverallBar','GraphLabel','StageLabel','StageBar','TimeLabel','OpenLog') {
        Set-Variable -Name $name -Value $window.FindName($name)
    }
    $OverallBar.Maximum = $totalGraphs
    $OpenLog.Add_Click({ Start-Process -FilePath 'notepad.exe' -ArgumentList ('"' + $logPath + '"') })

    $script:lastStatus = $null
    function Update-Progress {
        try {
            $snapshot = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($snapshot.mode -eq 'full') { $script:lastStatus = $snapshot }
        } catch {
            # The evaluation replaces the status file atomically. Retry transient read errors.
        }
        $state = $script:lastStatus
        $done = @($state.completed_graphs | Select-Object -Unique).Count
        $CountLabel.Text = "$done / $totalGraphs"
        $OverallBar.Value = $done
        $GraphLabel.Text = if ($state.graph) { [string]$state.graph } else { '11 图全量评测' }
        $StageBar.IsIndeterminate = $false
        $StageBar.Value = 0
        $StageBar.Foreground = '#8CAFF5'
        $OverallBar.Foreground = '#367BF5'
        $StageLabel.Foreground = '#536176'

        $label = switch ($state.stage) {
            'starting' { '正在启动实验' }
            'preparing' { '正在核对实验资产' }
            'graph_start' { '正在准备当前图谱' }
            'auxiliary_features' { "正在计算辅助特征 · $($state.done) / $($state.total) 条事实" }
            'backbone_features' { "正在计算主干分数 · $($state.family) · 种子 $($state.seed)" }
            'weight_search' {
                $phase = if ($state.validation) { '验证集校准' } else { '自监督权重搜索' }
                "$phase · $($state.backbone) · 已检查 $($state.candidates) 个候选"
            }
            'p1_validation' { '正在准备验证集校准' }
            'family_complete' { "当前种子评测完成 · $($state.family) · $($state.seed)" }
            'graph_complete' { if ($done -eq $totalGraphs) { '正在汇总指标并生成报告' } else { '当前图谱评测完成' } }
            'completed' { '评测完成，报告已生成' }
            'failed' { '实验失败 · ' + [string]$state.error }
            default { '等待实验状态' }
        }
        if ($state.stage -eq 'completed') {
            $OverallBar.Foreground = '#22A06B'
            $StageBar.Foreground = '#22A06B'
            $StageBar.Value = 100
        } elseif ($state.stage -eq 'failed') {
            $StageLabel.Foreground = '#BC2C37'
            $OverallBar.Foreground = '#BC2C37'
        } elseif ($state.stage -eq 'auxiliary_features' -and $state.total -gt 0) {
            $StageBar.Value = 100 * $state.done / $state.total
        } elseif ($state) {
            $StageBar.IsIndeterminate = $true
        }

        if ($state.updated_at) {
            $updated = [DateTimeOffset]::Parse([string]$state.updated_at)
            $localTime = $updated.ToLocalTime()
            $elapsed = [TimeSpan]::FromSeconds([double]$state.elapsed_seconds)
            $TimeLabel.Text = '已运行 {0} 小时 {1:D2} 分钟   ·   更新于 {2:HH:mm:ss}' -f [Math]::Floor($elapsed.TotalHours), $elapsed.Minutes, $localTime
            if ($state.stage -notin 'completed','failed' -and ([DateTimeOffset]::Now - $updated).TotalMinutes -gt 15) {
                $StageBar.IsIndeterminate = $false
                $StageLabel.Foreground = '#986800'
                $label = '进度超过 15 分钟未更新 · ' + $label
            }
        }
        $StageLabel.Text = $label
    }

    $timer = [Windows.Threading.DispatcherTimer]::new()
    $timer.Interval = [TimeSpan]::FromSeconds(2)
    $timer.Add_Tick({ Update-Progress })
    $window.Add_Loaded({ Update-Progress; $timer.Start() })
    $window.Add_ContentRendered({ $window.Topmost = $true; [void]$window.Activate(); $window.Topmost = $false })
    $window.Add_Closed({ $timer.Stop() })
    [void]$window.ShowDialog()
} catch {
    [IO.File]::AppendAllText((Join-Path $PSScriptRoot 'window_errors.log'), ($_.ToString() + [Environment]::NewLine))
    throw
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}

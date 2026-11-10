param(
    [string]$OutputDirectory='D:\motif-phase4-work\route-a-rerank-20261109',
    [string]$RunId='',
    [double]$StaleMinutes=15,
    [switch]$TestMode
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'ProgressModel.ps1')
Add-Type -AssemblyName PresentationFramework,PresentationCore,WindowsBase,System.Windows.Forms,System.Drawing
[Windows.Media.RenderOptions]::ProcessRenderMode=[Windows.Interop.RenderMode]::SoftwareOnly

$receipt=Read-ProgressJson (Join-Path $OutputDirectory 'launch_receipt.json')
if (-not $RunId -and $receipt) { $RunId=[string]$receipt.run }
if ($RunId -notmatch '^[a-zA-Z0-9_-]+$') { throw 'A valid experiment run ID is required.' }
$script:runDirectory=Join-Path $OutputDirectory "results\runs\$RunId"
$script:protocol=Read-ProgressJson (Join-Path $script:runDirectory 'protocol.json')
if (-not $script:protocol) { throw 'Experiment protocol was not found.' }
$monitorDirectory=Join-Path $OutputDirectory 'monitor'
[void][IO.Directory]::CreateDirectory($monitorDirectory)
$script:viewFile=Join-Path $monitorDirectory $(if ($TestMode) {'test_window_state.json'} else {'window_state.json'})
$mutexName='Local\RouteAProgress_' + $RunId + $(if ($TestMode) {'_test'} else {''})
$created=$false
$script:mutex=[Threading.Mutex]::new($true,$mutexName,[ref]$created)
if (-not $created) { $script:mutex.Dispose(); exit }

[xml]$xaml=@'
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
 xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
 Title="实验进度" Width="520" Height="278" MinWidth="460" MinHeight="260"
 ResizeMode="CanMinimize" WindowStartupLocation="CenterScreen" Background="#FAFBFD"
 FontFamily="Microsoft YaHei UI" ShowInTaskbar="True">
 <Grid Margin="28,22,28,20">
  <Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="Auto"/><RowDefinition Height="Auto"/><RowDefinition Height="Auto"/><RowDefinition Height="Auto"/><RowDefinition Height="*"/></Grid.RowDefinitions>
  <Grid><TextBlock Text="总体进度 · 估算" FontSize="13" Foreground="#64748B" VerticalAlignment="Center"/>
   <TextBlock x:Name="OverallText" Text="--" FontFamily="Segoe UI" FontSize="32" FontWeight="SemiBold" Foreground="#142238" HorizontalAlignment="Right"/></Grid>
  <ProgressBar x:Name="OverallBar" Grid.Row="1" Margin="0,10,0,24" Height="10" Minimum="0" Maximum="100" Foreground="#367BF5" Background="#E4EAF3" BorderThickness="0"/>
  <Grid Grid.Row="2"><TextBlock x:Name="StageText" Text="正在读取进度" FontSize="12" Foreground="#536176"/>
   <TextBlock x:Name="StagePercent" Text="" FontSize="12" FontFamily="Segoe UI" Foreground="#536176" HorizontalAlignment="Right"/></Grid>
  <ProgressBar x:Name="StageBar" Grid.Row="3" Margin="0,10,0,23" Height="6" Minimum="0" Maximum="100" Foreground="#8CAFF5" Background="#E4EAF3" BorderThickness="0"/>
  <Grid Grid.Row="4"><StackPanel Orientation="Horizontal"><Ellipse x:Name="StateDot" Width="7" Height="7" Fill="#94A3B8" Margin="0,0,8,0"/>
   <TextBlock x:Name="StateText" Text="正在检查" FontSize="12" Foreground="#64748B"/></StackPanel>
   <TextBlock x:Name="CompletedText" FontSize="11" Foreground="#94A3B8" HorizontalAlignment="Right"/></Grid>
 </Grid>
</Window>
'@
$reader=[Xml.XmlNodeReader]::new($xaml)
$script:window=[Windows.Markup.XamlReader]::Load($reader)
if ($TestMode) { $script:window.Title='实验进度 · 提醒测试' }
foreach ($name in @('OverallText','OverallBar','StageText','StagePercent','StageBar','StateDot','StateText','CompletedText')) {
    Set-Variable -Scope Script -Name $name -Value $script:window.FindName($name)
}
$script:tray=[Windows.Forms.NotifyIcon]::new()
$script:tray.Icon=[Drawing.SystemIcons]::Information
$script:tray.Text='实验进度'; $script:tray.Visible=$true
$script:probe=$null; $script:probeOutput=$null; $script:probeStarted=[DateTime]::MinValue
$script:lastProbe=[DateTime]::MinValue; $script:alive=$null; $script:missing=0
$script:processIdentity=''; $script:mainProcessId=0; $script:lastAlert=''
$script:lastStatus=$null; $script:health='checking'; $script:lastStateWrite=[DateTime]::MinValue

function Show-ExperimentNotice {
    param([string]$Health)
    if ($TestMode) { return }
    if ($script:lastAlert -eq $Health) { return }
    if ($Health -in @('running','checking')) { $script:lastAlert=''; return }
    $script:lastAlert=$Health
    $title='实验进度'; $message=''; $icon=[Windows.Forms.ToolTipIcon]::Info
    switch ($Health) {
        interrupted { $title='实验已中断'; $message='实验进程已退出或报告失败，已完成的进度仍保留。'; $icon=[Windows.Forms.ToolTipIcon]::Error }
        stalled { $title='实验进度暂未更新'; $message='进程仍在，但进度已较长时间没有更新，请检查。'; $icon=[Windows.Forms.ToolTipIcon]::Warning }
        complete { $title='实验已完成'; $message='所有实验已完成，报告与结果已保存。' }
    }
    if ($message) {
        $script:tray.ShowBalloonTip(6000,$title,$message,$icon)
        if ($Health -ne 'complete') { [Media.SystemSounds]::Exclamation.Play() }
    }
}

function Update-ProcessProbe {
    $now=[DateTime]::UtcNow
    if ($script:probe) {
        if ($script:probe.HasExited) {
            $line=''
            if ($script:probeOutput.IsCompleted) { $line=$script:probeOutput.Result.Trim() }
            if ($script:probe.ExitCode -eq 0 -and $line -match 'run_route_a_rerank\.py' -and $line -notmatch '\s--worker\s') {
                if (-not $script:processIdentity) { $script:processIdentity=$line }
                $script:alive=($line -eq $script:processIdentity)
            } elseif ($script:probe.ExitCode -eq 1 -and -not $line) {
                $script:alive=$false
            } else {
                # A failed WSL connection is not evidence that the experiment stopped.
                $script:alive=$null
            }
            if ($null -ne $script:alive -and -not $script:alive) { $script:missing++ }
            elseif ($script:alive) { $script:missing=0 }
            $script:probe.Dispose();$script:probe=$null;$script:lastProbe=$now
        } elseif (($now-$script:probeStarted).TotalSeconds -gt 10) {
            $script:probe.Kill();$script:probe.Dispose();$script:probe=$null
            $script:alive=$null;$script:lastProbe=$now
        }
        return
    }
    if (($now-$script:lastProbe).TotalSeconds -lt 15) { return }
    $lockPath=Join-Path $OutputDirectory "logs\$RunId.lock"
    $candidateId=0
    if (Test-Path -LiteralPath $lockPath) {
        [void][int]::TryParse([IO.File]::ReadAllText($lockPath).Trim(),[ref]$candidateId)
    }
    if ($candidateId -le 0 -and $receipt) { $candidateId=[int]$receipt.main_pid }
    if ($candidateId -le 0) { $script:alive=$null;return }
    if ($candidateId -ne $script:mainProcessId) {
        $script:mainProcessId=$candidateId;$script:processIdentity='';$script:missing=0
    }
    $info=[Diagnostics.ProcessStartInfo]::new()
    $info.FileName=Join-Path $env:WINDIR 'System32\wsl.exe'
    $info.Arguments="-d Ubuntu-24.04 --exec /bin/ps -p $candidateId -o lstart=,args="
    $info.UseShellExecute=$false;$info.CreateNoWindow=$true
    $info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
    $script:probe=[Diagnostics.Process]::Start($info)
    $script:probeOutput=$script:probe.StandardOutput.ReadToEndAsync()
    # Drain stderr asynchronously so a failed probe cannot block the UI.
    $script:probeError=$script:probe.StandardError.ReadToEndAsync()
    $script:probeStarted=$now
}

function Update-ProgressWindow {
    try {
        $current=Read-ProgressJson (Join-Path $script:runDirectory 'status.json')
        if ($current) { $script:lastStatus=$current }
        $current=$script:lastStatus
        $view=Get-ProgressView $script:runDirectory $script:protocol $current
        $age=0.0
        if ($current -and $current.at) {
            $age=([DateTimeOffset]::UtcNow-[DateTimeOffset]::Parse($current.at)).TotalSeconds
        }
        if (-not $view.Complete -and $current.state -ne 'failed' -and -not $TestMode) { Update-ProcessProbe }
        $script:health=Get-ExperimentHealth -State ([string]$current.state) -Complete $view.Complete -Alive $script:alive -MissingCount $script:missing -AgeSeconds $age -StaleMinutes $StaleMinutes
        $script:OverallBar.Value=$view.Overall
        $script:OverallText.Text=('{0}%' -f [Math]::Floor($view.Overall))
        $script:StageText.Text=$view.Caption
        $script:StageBar.IsIndeterminate=$view.Indeterminate -and $script:health -notin @('interrupted','complete')
        $script:StageBar.Value=$view.Current
        $script:StagePercent.Text=$(if ($view.Indeterminate) {''} else {'{0}%' -f [Math]::Floor($view.Current)})
        $script:CompletedText.Text=('{0} / {1} 组' -f $view.Completed,$view.Units)
        $color='#367BF5';$dot='#22A06B';$label='正常运行'
        switch ($script:health) {
            checking { $dot='#D69B26';$label='正在确认运行状态' }
            interrupted { $color='#DC4B55';$dot='#DC4B55';$label='实验已中断';$script:StageText.Text='已完成的进度仍保留' }
            stalled { $color='#D69B26';$dot='#D69B26';$label='进程仍在 · 进度暂未更新' }
            complete { $color='#22A06B';$dot='#22A06B';$label='全部完成' }
        }
        $script:OverallBar.Foreground=[Windows.Media.BrushConverter]::new().ConvertFromString($color)
        $script:StateDot.Fill=[Windows.Media.BrushConverter]::new().ConvertFromString($dot)
        $script:StateText.Text=$label
        $script:StateText.ToolTip=$(if ($current.error) {[string]$current.error} else {"上次更新：$($current.at)"})
        Show-ExperimentNotice $script:health
        if (([DateTime]::UtcNow-$script:lastStateWrite).TotalSeconds -gt 5) {
            $state=[ordered]@{pid=$PID;run=$RunId;health=$script:health;overall=$view.Overall;current=$view.Current;caption=$view.Caption;main_pid=$script:mainProcessId;checked_at=[DateTimeOffset]::Now.ToString('o')}
            [IO.File]::WriteAllText($script:viewFile,($state | ConvertTo-Json),[Text.UTF8Encoding]::new($false))
            $script:lastStateWrite=[DateTime]::UtcNow
        }
    } catch {
        $script:StateText.Text='暂时无法读取 · 自动重试'
        $script:StateText.ToolTip=$_.Exception.Message
        [IO.File]::AppendAllText((Join-Path $monitorDirectory 'window_errors.log'),([DateTimeOffset]::Now.ToString('o')+' '+$_.Exception.ToString()+[Environment]::NewLine))
    }
}

$script:timer=[Windows.Threading.DispatcherTimer]::new()
$script:timer.Interval=[TimeSpan]::FromSeconds(2)
$script:timer.Add_Tick({ Update-ProgressWindow })
$script:window.Add_Loaded({ Update-ProgressWindow; $script:timer.Start() })
$script:window.Add_ContentRendered({
    # Bring this newly requested window into view once, without staying on top.
    $script:window.Topmost=$true
    [void]$script:window.Activate()
    $script:window.Topmost=$false
})
$script:window.Add_Closed({
    $script:timer.Stop()
    if ($script:probe -and -not $script:probe.HasExited) { $script:probe.Kill() }
    if ($script:probe) { $script:probe.Dispose() }
    $script:tray.Visible=$false;$script:tray.Dispose()
    $script:mutex.ReleaseMutex();$script:mutex.Dispose()
})
[void]$script:window.ShowDialog()

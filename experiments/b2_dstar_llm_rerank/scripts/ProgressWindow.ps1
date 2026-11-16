Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$Root = Split-Path -Parent $PSScriptRoot
$Form = New-Object System.Windows.Forms.Form
$Form.Text = 'B2+D* 图谱重排实验进度'
$Form.Size = New-Object System.Drawing.Size(640, 280)
$Form.StartPosition = 'CenterScreen'
$Label = New-Object System.Windows.Forms.Label
$Label.Location = New-Object System.Drawing.Point(20, 20)
$Label.Size = New-Object System.Drawing.Size(580, 150)
$Label.Font = New-Object System.Drawing.Font('Microsoft YaHei UI', 10)
$Bar = New-Object System.Windows.Forms.ProgressBar
$Bar.Location = New-Object System.Drawing.Point(20, 180)
$Bar.Size = New-Object System.Drawing.Size(580, 25)
$Form.Controls.Add($Label)
$Form.Controls.Add($Bar)
$Timer = New-Object System.Windows.Forms.Timer
$Timer.Interval = 2000
$Timer.Add_Tick({
    try {
        $S = Get-Content -LiteralPath "$Root/status.json" -Raw | ConvertFrom-Json
        $Percent = if ($S.total -gt 0) { [int][math]::Min(100, 100 * $S.done / $S.total) } else { 0 }
        $Bar.Value = $Percent
        $Label.Text = "状态：$($S.state) / $($S.stage)`n图谱：$($S.graph)；种子：$($S.seed)；阶段：$($S.split)`n当前进度：$($S.done) / $($S.total) ($Percent%)；已完成图谱：$($S.completed_graphs.Count) / 11`n本阶段剩余约：$([int]$S.eta_seconds) 秒；累计 API：$($S.api_calls)`nToken：$($S.prompt_tokens) 输入 + $($S.completion_tokens) 输出；费用上界约：USD $($S.estimated_cost_usd_upper)`n更新时间：$($S.updated_at)"
        if ($S.state -eq 'failed') { $Label.Text += "`n错误：$($S.error)" }
    } catch { $Label.Text = '正在等待实验状态文件…' }
})
$Form.Add_FormClosed({ $Timer.Stop(); $Timer.Dispose() })
$Timer.Start()
[void]$Form.ShowDialog()

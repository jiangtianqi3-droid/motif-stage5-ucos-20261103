param([string]$OutputDirectory='D:\motif-phase4-work\route-a-rerank-20261109')
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'ProgressModel.ps1')
function Assert-Check { param([bool]$Condition,[string]$Message); if(-not $Condition){throw $Message} }
Assert-Check ((Get-ExperimentHealth 'running' $false $true 0 10) -eq 'running') 'Healthy process was not recognized'
Assert-Check ((Get-ExperimentHealth 'failed' $false $true 0 10) -eq 'interrupted') 'Failure flag must alert'
Assert-Check ((Get-ExperimentHealth 'running' $false $false 1 10) -eq 'checking') 'One missing probe must not falsely alert'
Assert-Check ((Get-ExperimentHealth 'running' $false $false 2 10) -eq 'interrupted') 'Confirmed exit must alert'
Assert-Check ((Get-ExperimentHealth 'running' $false $true 0 901) -eq 'stalled') 'Stale live process needs a warning'
Assert-Check ((Get-ExperimentHealth 'running' $false $null 0 5000) -eq 'checking') 'A WSL connection failure is not an experiment exit'
Assert-Check ((Get-ExperimentHealth 'failed' $true $false 2 9999) -eq 'complete') 'Final completion must take precedence'

$fixture=Join-Path $OutputDirectory 'monitor\test_fixture'
$run=Join-Path $fixture 'results\runs\monitor_test'
[void][IO.Directory]::CreateDirectory($run)
function Write-Fixture { param([string]$Path,$Object); [void][IO.Directory]::CreateDirectory((Split-Path -Parent $Path)); [IO.File]::WriteAllText($Path,($Object|ConvertTo-Json -Depth 7),[Text.UTF8Encoding]::new($false)) }
$protocol=[pscustomobject]@{datasets=@('small','large');families=@('B0');validation_max_facts=256}
Write-Fixture (Join-Path $run 'protocol.json') $protocol
Write-Fixture (Join-Path $run 'completion.json') @{state='pending'}
Write-Fixture (Join-Path $run 'small\B0\split_leakage_audit.json') @{valid_facts=0;test_facts=10}
Write-Fixture (Join-Path $run 'large\B0\split_leakage_audit.json') @{valid_facts=0;test_facts=90}
Write-Fixture (Join-Path $run 'small\B0\completion.json') @{state='complete'}
$status=[pscustomobject]@{state='running';stage='structural_evidence';dataset='large';family='B0';done=50;total=100;at=[DateTimeOffset]::Now.ToString('o')}
$view=Get-ProgressView $run $protocol $status
Assert-Check ([Math]::Abs($view.Overall-53.2) -lt .00001) 'Overall progress must be weighted by query count'
Assert-Check ($view.Current -eq 50 -and $view.Completed -eq 1 -and -not $view.Complete) 'Stage progress calculation failed'
$status.stage='validation_selection';$status|Add-Member -NotePropertyName method -NotePropertyValue 'R2'
$view=Get-ProgressView $run $protocol $status
Assert-Check ($view.Overall -gt 95 -and $view.Overall -lt 100) 'Validation stage should reserve report completion'
Write-Fixture (Join-Path $run 'completion.json') @{state='complete'}
$view=Get-ProgressView $run $protocol $status
Assert-Check ($view.Overall -eq 100 -and $view.Complete) 'Terminal completion should show 100 percent'

# A separate fixture drives the GUI failure state without stopping a real process.
$gui=Join-Path $OutputDirectory 'monitor\gui_fixture'
$guiRun=Join-Path $gui 'results\runs\monitor_test'
Write-Fixture (Join-Path $guiRun 'protocol.json') @{datasets=@('FBIngram_25');families=@('B0');validation_max_facts=256}
Write-Fixture (Join-Path $guiRun 'completion.json') @{state='pending'}
Write-Fixture (Join-Path $guiRun 'FBIngram_25\B0\split_leakage_audit.json') @{valid_facts=256;test_facts=5716}
Write-Fixture (Join-Path $guiRun 'status.json') @{state='failed';stage='exception';dataset='FBIngram_25';family='B0';at=[DateTimeOffset]::Now.ToString('o');error='Simulated failure in isolated monitor fixture'}
Write-Output '11 monitor checks passed. Isolated GUI failure fixture is ready.'

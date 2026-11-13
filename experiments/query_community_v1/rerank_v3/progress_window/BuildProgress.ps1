$ErrorActionPreference='Stop'
$target=Join-Path $PSScriptRoot 'ExperimentProgress.exe'
if (Test-Path -LiteralPath $target) { throw 'Executable already exists; preserve the current build before rebuilding.' }
Add-Type -TypeDefinition ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'ProgressHost.cs'))) -Language CSharp -ReferencedAssemblies 'System.Management.Automation' -OutputAssembly $target -OutputType WindowsApplication
Write-Output $target

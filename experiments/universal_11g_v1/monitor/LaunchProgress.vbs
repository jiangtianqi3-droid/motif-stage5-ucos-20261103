Option Explicit
Dim shell, files, scriptPath
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
scriptPath = files.BuildPath(files.GetParentFolderName(WScript.ScriptFullName), "ProgressWindow.ps1")
shell.Run "powershell.exe -NoProfile -STA -WindowStyle Hidden -File """ & scriptPath & """", 0, False

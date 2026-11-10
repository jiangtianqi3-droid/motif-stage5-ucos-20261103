Option Explicit
Dim shell, files, folder, programPath, command
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
folder = files.GetParentFolderName(WScript.ScriptFullName)
programPath = files.BuildPath(folder, "ExperimentProgress.exe")
command = """" & programPath & """"
shell.Run command, 1, False

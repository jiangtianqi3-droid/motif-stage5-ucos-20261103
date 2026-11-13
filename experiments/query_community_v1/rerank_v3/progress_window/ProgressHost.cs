using System;
using System.IO;
using System.Management.Automation;
using System.Management.Automation.Runspaces;
using System.Threading;

internal static class ProgressHost
{
    [STAThread]
    private static int Main(string[] args)
    {
        string entry = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ProgressWindow.ps1");
        try
        {
            var initial = InitialSessionState.CreateDefault();
            // Trust only this local application session; no machine policy is changed.
            initial.AuthorizationManager = new AuthorizationManager("ExperimentProgress");
            using (var runspace = RunspaceFactory.CreateRunspace(initial))
            {
                runspace.ApartmentState = ApartmentState.STA;
                runspace.ThreadOptions = PSThreadOptions.UseCurrentThread;
                runspace.Open();
                using (var shell = PowerShell.Create())
                {
                    shell.Runspace = runspace;
                    shell.AddCommand(entry);
                    shell.Invoke();
                    if (shell.HadErrors)
                    {
                        foreach (var error in shell.Streams.Error) SaveError(error.ToString());
                        return 1;
                    }
                }
            }
            return 0;
        }
        catch (Exception error) { SaveError(error.ToString()); return 1; }
    }

    private static void SaveError(string text)
    {
        string folder = Path.Combine(Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..")), "monitor");
        Directory.CreateDirectory(folder);
        File.AppendAllText(Path.Combine(folder, "native_startup_error.log"), DateTimeOffset.Now + " " + text + Environment.NewLine);
    }
}

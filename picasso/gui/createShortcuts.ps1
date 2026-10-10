# Creates Windows shortcuts for the Picasso modules in the repository root.
# Run it from the activated conda environment that has Picasso installed:
#   powershell -ExecutionPolicy Bypass -File picasso\gui\createShortcuts.ps1

if (-not $env:CONDA_PREFIX) {
    Write-Error "Activate the conda environment with Picasso installed first."
    exit 1
}
$pythonw = Join-Path $env:CONDA_PREFIX "pythonw.exe"
$python = Join-Path $env:CONDA_PREFIX "python.exe"
& $python -c "import picasso" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Picasso is not installed in $env:CONDA_PREFIX."
    exit 1
}

$root = (Resolve-Path "$PSScriptRoot/../..").Path
$modules = [ordered]@{
    Design   = "design"
    Simulate = "simulate"
    Localize = "localize"
    Filter   = "filter"
    Render   = "render"
    Average  = "average"
    SPINNA   = "spinna"
    Server   = "server"
    Nanotron = "nanotron"
}
$shell = New-Object -COM WScript.Shell
foreach ($name in $modules.Keys) {
    $command = $modules[$name]
    $s = $shell.CreateShortcut("$root/$name.lnk")
    # Server is a terminal app, so it needs a console window
    $s.TargetPath = if ($command -eq "server") { $python } else { $pythonw }
    $s.Arguments = "-m picasso $command"
    $s.WorkingDirectory = $root
    $s.IconLocation = "$PSScriptRoot/icons/$command.ico"
    $s.Save()
}
Write-Output "Created shortcuts in $root"

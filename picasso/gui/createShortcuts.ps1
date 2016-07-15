$s=(New-Object -COM WScript.Shell).CreateShortcut("$PSScriptRoot/../../ToRaw.lnk")
$s.TargetPath="pythonw"
$s.Arguments="-m picasso toraw"
$s.IconLocation="$PSScriptRoot/icons/toraw.ico"
$s.Save()

$s=(New-Object -COM WScript.Shell).CreateShortcut("$PSScriptRoot/../../Localize.lnk")
$s.TargetPath="pythonw"
$s.Arguments="-m picasso localize"
$s.IconLocation="$PSScriptRoot/icons/localize.ico"
$s.Save()

$s=(New-Object -COM WScript.Shell).CreateShortcut("$PSScriptRoot/../../Filter.lnk")
$s.TargetPath="pythonw"
$s.Arguments="-m picasso filter"
$s.IconLocation="$PSScriptRoot/icons/filter.ico"
$s.Save()

$s=(New-Object -COM WScript.Shell).CreateShortcut("$PSScriptRoot/../../Render.lnk")
$s.TargetPath="pythonw"
$s.Arguments="-m picasso render"
$s.IconLocation="$PSScriptRoot/icons/render.ico"
$s.Save()

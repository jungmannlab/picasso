$s=(New-Object -COM WScript.Shell).CreateShortcut("$PSScriptRoot/../../ToRaw.lnk")
$s.TargetPath="picasso"
$s.Arguments="toraw"
$s.IconLocation="$PSScriptRoot/icons/toraw.ico"
$s.Save()

$s=(New-Object -COM WScript.Shell).CreateShortcut("$PSScriptRoot/../../Localize.lnk")
$s.TargetPath="picasso"
$s.Arguments="localize"
$s.IconLocation="$PSScriptRoot/icons/localize.ico"
$s.Save()

$s=(New-Object -COM WScript.Shell).CreateShortcut("$PSScriptRoot/../../Filter.lnk")
$s.TargetPath="picasso"
$s.Arguments="filter"
$s.IconLocation="$PSScriptRoot/icons/filter.ico"
$s.Save()

$s=(New-Object -COM WScript.Shell).CreateShortcut("$PSScriptRoot/../../Render.lnk")
$s.TargetPath="picasso"
$s.Arguments="render"
$s.IconLocation="$PSScriptRoot/icons/render.ico"
$s.Save()

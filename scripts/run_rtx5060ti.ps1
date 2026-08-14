#requires -Version 5.1
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $ProjectRoot "src\pipeline_rtx5060ti.py"

if (-not (Test-Path $Launcher)) {
    throw "Missing launcher: $Launcher"
}

Set-Location $ProjectRoot

& py -3.12 $Launcher @args
exit $LASTEXITCODE

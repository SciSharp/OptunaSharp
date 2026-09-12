[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z][a-z0-9-]{0,63}$')]
    [string]$Name
)

$ErrorActionPreference = 'Stop'
$webRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'wwwroot'))
$destination = [IO.Path]::GetFullPath((Join-Path $webRoot $Name))
if (-not $destination.StartsWith($webRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The presentation path must stay inside wwwroot.'
}
if (Test-Path -LiteralPath $destination) {
    throw "A presentation named '$Name' already exists. Choose a new name."
}
$null = New-Item -ItemType Directory -Path $destination
Copy-Item -LiteralPath (Join-Path $webRoot 'template/index.html') -Destination (Join-Path $destination 'index.html')
Write-Host "Created $destination"
Write-Host "After starting run.ps1, open http://127.0.0.1:5088/$Name/"

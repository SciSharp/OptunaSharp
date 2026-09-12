[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 5088,
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
$projectPath = Join-Path $PSScriptRoot 'OptunaSharp.PresentationHost/OptunaSharp.PresentationHost.csproj'
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    throw 'Install the .NET 10 SDK, then run this script again.'
}
$sdkVersion = & dotnet --version
if ($LASTEXITCODE -ne 0 -or [int]($sdkVersion.Split('.')[0]) -lt 10) {
    throw 'The presentation host requires the .NET 10 SDK.'
}

if ($SelfTest) {
    & dotnet run --project $projectPath --configuration $Configuration -- --self-test
} else {
    Write-Host "Presentation: http://127.0.0.1:$Port/nsga3/"
    Write-Host "Three.js atlas: http://127.0.0.1:$Port/nsga3-atlas/"
    Write-Host "Complete sourcebook: http://127.0.0.1:$Port/nsga3-atlas/sourcebook.html"
    Write-Host 'Press Ctrl+C to stop the host.'
    & dotnet run --project $projectPath --configuration $Configuration -- --urls "http://127.0.0.1:$Port"
}
exit $LASTEXITCODE

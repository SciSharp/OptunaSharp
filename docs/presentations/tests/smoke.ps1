# Run against an already started presentation host. Uses only PowerShell and .NET.
[CmdletBinding()]
param(
    [string]$BaseUrl = 'http://127.0.0.1:5088'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Net.Http
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds(60)
$baseAddress = $BaseUrl.TrimEnd('/')
$runIds = [System.Collections.Generic.List[string]]::new()
$script:checks = 0

function Assert-That {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw "Assertion failed: $Message" }
    $script:checks++
}

function Invoke-TestRequest {
    param(
        [string]$Method = 'GET',
        [Parameter(Mandatory = $true)][string]$Path,
        [object]$Body = $null,
        [int]$ExpectedStatus = 200
    )
    $request = [System.Net.Http.HttpRequestMessage]::new(
        [System.Net.Http.HttpMethod]::new($Method), "$baseAddress$Path")
    $response = $null
    try {
        if ($null -ne $Body) {
            $jsonBody = ConvertTo-Json -InputObject $Body -Depth 30 -Compress
            $request.Content = [System.Net.Http.StringContent]::new(
                $jsonBody, [System.Text.Encoding]::UTF8, 'application/json')
        }
        $response = $client.SendAsync($request).GetAwaiter().GetResult()
        $content = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        $status = [int]$response.StatusCode
        Assert-That ($status -eq $ExpectedStatus) "$Method $Path returned $status; expected $ExpectedStatus. $content"
        $json = $null
        if ($content.TrimStart().StartsWith('{') -or $content.TrimStart().StartsWith('[')) {
            $json = ConvertFrom-Json -InputObject $content
        }
        return [pscustomobject]@{ StatusCode = $status; Content = $content; Json = $json }
    }
    finally {
        if ($null -ne $response) { $response.Dispose() }
        $request.Dispose()
    }
}

function Assert-Finite {
    param([double]$Value, [string]$Message)
    Assert-That (-not [double]::IsNaN($Value) -and -not [double]::IsInfinity($Value)) $Message
}

function Get-SnapshotSignature {
    param([object]$Snapshot)
    # Exclude only the unique session identifier; compare observable algorithm state.
    return ConvertTo-Json -Depth 30 -Compress -InputObject ([ordered]@{
        generation = $Snapshot.generation
        evaluations = $Snapshot.evaluations
        config = $Snapshot.config
        referenceDirections = $Snapshot.referenceDirections
        parents = $Snapshot.parents
        offspring = $Snapshot.offspring
        candidates = $Snapshot.candidates
        survivors = $Snapshot.survivors
        paretoFront = $Snapshot.paretoFront
        metrics = $Snapshot.metrics
        selection = $Snapshot.selection
    })
}

function Assert-Snapshot {
    param([object]$Snapshot, [int]$Generation)
    $populationSize = 12
    $candidateCount = if ($Generation -eq 0) { 12 } else { 24 }
    Assert-That ($Snapshot.generation -eq $Generation) "Generation is $Generation"
    Assert-That ($Snapshot.evaluations -eq (($Generation + 1) * $populationSize)) 'Evaluation count tracks completed generations'
    Assert-That (@($Snapshot.offspring).Count -eq $populationSize) 'One complete offspring generation is exposed'
    Assert-That (@($Snapshot.parents).Count -eq ($candidateCount - $populationSize)) 'Parent count matches generation phase'
    Assert-That (@($Snapshot.candidates).Count -eq $candidateCount) 'Candidate pool combines offspring and previous survivors'
    Assert-That (@($Snapshot.survivors).Count -eq $populationSize) 'Exactly one population survives'
    Assert-That (@($Snapshot.referenceDirections).Count -eq 10) 'Three objectives and three divisions provide ten reference directions'

    $candidateIds = @($Snapshot.candidates | ForEach-Object { [int]$_.id })
    $survivorIds = @($Snapshot.survivors | ForEach-Object { [int]$_.id })
    $acceptedIds = @($Snapshot.selection.acceptedIds)
    $eligibleIds = @($acceptedIds) + @($Snapshot.selection.splittingFrontIds)
    Assert-That (@($candidateIds | Select-Object -Unique).Count -eq $candidateCount) 'Candidate identifiers are unique'
    Assert-That (@($survivorIds | Select-Object -Unique).Count -eq $populationSize) 'Survivor identifiers are unique'
    Assert-That (@($Snapshot.selection.selectedIds).Count -eq $populationSize) 'Selection trace records every survivor'
    Assert-That ((@($Snapshot.selection.selectedIds) -join ',') -eq ($survivorIds -join ',')) 'Selection order matches the returned survivors'
    foreach ($id in $survivorIds) {
        Assert-That ($candidateIds -contains $id) 'Every survivor was a candidate'
        Assert-That ($eligibleIds -contains $id) 'Survivors come from accepted fronts or the splitting front'
    }
    foreach ($id in $acceptedIds) {
        Assert-That ($survivorIds -contains $id) 'Every member of a fully accepted front survives'
    }

    foreach ($point in $Snapshot.candidates) {
        Assert-That (@($point.values).Count -eq 3) 'Objective vectors have three coordinates'
        Assert-That (@($point.normalized).Count -eq 3) 'Normalized vectors have three coordinates'
        $radiusSquared = 0.0
        foreach ($value in $point.values) {
            Assert-Finite $value 'Objective coordinates are finite'
            Assert-That ($value -ge 0) 'DTLZ2 objectives lie in the positive octant'
            $radiusSquared += $value * $value
        }
        Assert-That ($radiusSquared -ge (1.0 - 1e-10)) 'DTLZ2 samples lie on or outside the unit sphere'
        foreach ($value in $point.normalized) { Assert-Finite $value 'Normalized coordinates are finite' }
        Assert-Finite $point.distance 'Reference distances are finite'
        Assert-That ($point.distance -ge 0) 'Reference distance is nonnegative'
        Assert-That ($point.referenceIndex -ge 0 -and $point.referenceIndex -lt 10) 'Reference association indexes an existing direction'
        Assert-That ($point.selected -eq ($survivorIds -contains $point.id)) 'Candidate selection flags match survivor membership'
    }

    Assert-That (@($Snapshot.selection.nicheSteps).Count -eq ($populationSize - $acceptedIds.Count)) 'Niching trace accounts for remaining survivor slots'
    foreach ($step in $Snapshot.selection.nicheSteps) {
        Assert-That (($survivorIds -contains $step.trialId) -and ($acceptedIds -notcontains $step.trialId)) 'Each niching step identifies a selected splitting-front candidate'
        Assert-That ($step.occupancyBefore -ge 0) 'Niche occupancy cannot be negative'
    }
    Assert-Finite $Snapshot.metrics.meanDistanceToFront 'Reported convergence metric is finite'
    Assert-That ($Snapshot.metrics.meanDistanceToFront -ge 0) 'Reported convergence metric is nonnegative'
    Assert-That (@($Snapshot.paretoFront).Count -gt 0) 'Survivor population exposes a nonempty nondominated subset'
    foreach ($point in $Snapshot.paretoFront) {
        Assert-That ($survivorIds -contains $point.id) 'Displayed Pareto approximation belongs to current survivors'
    }
}

try {
    Write-Host "Checking presentation host at $baseAddress"
    $health = (Invoke-TestRequest -Path '/api/health').Json
    Assert-That ($health.status -eq 'ok' -and $health.engine -eq 'OptunaSharp') 'Health reports the OptunaSharp backend'
    foreach ($path in @('/', '/nsga3/', '/nsga3/nsga3.js', '/nsga3/nsga3.css', '/nsga3/readability.css', '/shared/presentation.js', '/shared/presentation.css', '/template/')) {
        $static = Invoke-TestRequest -Path $path
        Assert-That (-not [string]::IsNullOrWhiteSpace($static.Content)) "$path serves content"
    }

    foreach ($divisions in @(1, 3, 6, 16)) {
        $lab = (Invoke-TestRequest -Method 'POST' -Path '/api/nsga3/reference-lab' -Body @{
            divisions = $divisions; point = @(2.0, 0.0, 0.0)
        }).Json
        $expectedDirections = ($divisions + 1) * ($divisions + 2) / 2
        Assert-That (@($lab.referenceDirections).Count -eq $expectedDirections) 'Reference density matches the three-objective simplex combination count'
        $direction = $lab.referenceDirections[$lab.referenceIndex]
        Assert-That ($direction[0] -eq 1 -and $direction[1] -eq 0 -and $direction[2] -eq 0) 'An axis-aligned point associates with its axis direction'
        Assert-That ([Math]::Abs($lab.distance) -lt 1e-12) 'A point on a reference ray has zero perpendicular distance'
        Assert-That ([Math]::Abs($lab.projection[0] - 2.0) -lt 1e-12) 'Projection follows the ray beyond its simplex endpoint'
    }
    foreach ($body in @(@{ divisions = 0 }, @{ divisions = 17 }, @{ point = @(0.2, 0.8) }, @{ point = @(-0.1, 0.2, 0.3) }, @{ point = @(6, 0, 0) })) {
        $null = Invoke-TestRequest -Method 'POST' -Path '/api/nsga3/reference-lab' -Body $body -ExpectedStatus 400
    }

    $config = @{ populationSize = 12; divisions = 3; seed = 42; mutationProbability = 0.15; problem = 'dtlz2' }
    $first = (Invoke-TestRequest -Method 'POST' -Path '/api/nsga3/runs' -Body $config).Json
    $runIds.Add($first.runId)
    Assert-Snapshot $first 0
    $initialSignature = Get-SnapshotSignature $first
    $read = (Invoke-TestRequest -Path "/api/nsga3/runs/$($first.runId)").Json
    Assert-That ((Get-SnapshotSignature $read) -eq $initialSignature) 'Reading a snapshot neither evaluates nor alters algorithm state'

    $firstStep = (Invoke-TestRequest -Method 'POST' -Path "/api/nsga3/runs/$($first.runId)/step").Json
    Assert-Snapshot $firstStep 1
    $firstStepSignature = Get-SnapshotSignature $firstStep
    $read = (Invoke-TestRequest -Path "/api/nsga3/runs/$($first.runId)").Json
    Assert-That ((Get-SnapshotSignature $read) -eq $firstStepSignature) 'Reading generation one preserves the completed snapshot'

    $repeat = (Invoke-TestRequest -Method 'POST' -Path '/api/nsga3/runs' -Body $config).Json
    $runIds.Add($repeat.runId)
    Assert-That ($repeat.runId -ne $first.runId) 'Independent sessions have unique identifiers'
    Assert-That ((Get-SnapshotSignature $repeat) -eq $initialSignature) 'The same seed and configuration reproduce initial values and selection'
    $repeatStep = (Invoke-TestRequest -Method 'POST' -Path "/api/nsga3/runs/$($repeat.runId)/step").Json
    Assert-That ((Get-SnapshotSignature $repeatStep) -eq $firstStepSignature) 'The same seed reproduces the next generation even when only one session had extra reads'

    foreach ($body in @(@{ populationSize = 11 }, @{ populationSize = 97 }, @{ divisions = 0 }, @{ seed = -1 }, @{ mutationProbability = 1.1 }, @{ problem = 'unknown' })) {
        $null = Invoke-TestRequest -Method 'POST' -Path '/api/nsga3/runs' -Body $body -ExpectedStatus 400
    }
    $missingId = [Guid]::NewGuid().ToString('N')
    $null = Invoke-TestRequest -Path "/api/nsga3/runs/$missingId" -ExpectedStatus 404
    $null = Invoke-TestRequest -Method 'POST' -Path "/api/nsga3/runs/$missingId/step" -ExpectedStatus 404
    foreach ($runId in $runIds.ToArray()) {
        $null = Invoke-TestRequest -Method 'DELETE' -Path "/api/nsga3/runs/$runId" -ExpectedStatus 204
        $null = Invoke-TestRequest -Path "/api/nsga3/runs/$runId" -ExpectedStatus 404
        $null = $runIds.Remove($runId)
    }
    Write-Host "PASS: $script:checks assertions covering assets, reference geometry, real optimization, reproducibility, read-only snapshots, validation, and session cleanup."
}
finally {
    # Ensure failures do not consume the host's limited active-session slots.
    foreach ($runId in $runIds) {
        try { $null = Invoke-TestRequest -Method 'DELETE' -Path "/api/nsga3/runs/$runId" -ExpectedStatus 204 }
        catch { Write-Warning "Could not clean up run ${runId}: $($_.Exception.Message)" }
    }
    $client.Dispose()
}

[CmdletBinding()]
param(
    [string[]]$Levels = @("fp64", "path1", "path2", "path3", "path4", "path5", "path6", "path7", "path8"),
    [string]$Config = "RelWithDebInfo",
    [string]$Generator = "Ninja",
    [int]$Parallel = 0,
    [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$levelOrder = @("fp64", "path1", "path2", "path3", "path4", "path5", "path6", "path7", "path8")

function Get-DefaultParallelism {
    param(
        [string]$SelectedGenerator
    )

    $cpuCount = 1
    if ($env:NUMBER_OF_PROCESSORS -and ($env:NUMBER_OF_PROCESSORS -as [int]) -gt 0) {
        $cpuCount = [int]$env:NUMBER_OF_PROCESSORS
    }

    return [Math]::Min($cpuCount, 8)
}

function Test-CommandAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName
    )

    return $null -ne (Get-Command $CommandName -ErrorAction SilentlyContinue)
}

function Normalize-Levels {
    param(
        [string[]]$RawLevels,
        [string[]]$AllowedLevels
    )

    $requested = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in $RawLevels) {
        if (-not $entry) {
            continue
        }

        foreach ($part in ($entry -split ',')) {
            $level = $part.Trim()
            if (-not $level) {
                continue
            }
            if (-not $requested.Add($level)) {
                continue
            }
        }
    }

    if ($requested.Count -eq 0) {
        throw "No levels were provided."
    }

    $unknown = @()
    foreach ($level in $requested) {
        if ($AllowedLevels -notcontains $level) {
            $unknown += $level
        }
    }
    if ($unknown.Count -gt 0) {
        throw "Unsupported level(s): $($unknown -join ', '). Valid values: $($AllowedLevels -join ', ')"
    }

    $normalized = @()
    foreach ($level in $AllowedLevels) {
        if ($requested.Contains($level)) {
            $normalized += $level
        }
    }
    return $normalized
}

function Get-CachedGenerator {
    param(
        [string]$CMakeCachePath
    )

    if (-not (Test-Path -LiteralPath $CMakeCachePath)) {
        return $null
    }

    $match = Select-String -Path $CMakeCachePath -Pattern '^CMAKE_GENERATOR:INTERNAL=(.*)$'
    if ($match -and $match.Matches.Count -gt 0) {
        return $match.Matches[0].Groups[1].Value.Trim()
    }

    return $null
}

function Format-Command {
    param(
        [string]$Exe,
        [string[]]$CommandArgs
    )

    $parts = @($Exe)
    foreach ($arg in $CommandArgs) {
        if ($arg -match '\s') {
            $parts += ('"{0}"' -f $arg.Replace('"', '\"'))
        }
        else {
            $parts += $arg
        }
    }
    return ($parts -join ' ')
}

function Invoke-Step {
    param(
        [string]$Level,
        [string]$Phase,
        [string]$WorkingDirectory,
        [string]$Exe,
        [string[]]$CommandArgs
    )

    $display = Format-Command -Exe $Exe -CommandArgs $CommandArgs
    Write-Host "[$Level][$Phase] $display"
    Push-Location $WorkingDirectory
    try {
        & $Exe @CommandArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE."
        }
    }
    catch {
        throw "Level '$Level' failed during $Phase in '$WorkingDirectory'. Command: $display`n$($_.Exception.Message)"
    }
    finally {
        Pop-Location
    }
}

$parallelWasImplicit = $Parallel -le 0
if ($Parallel -le 0) {
    $Parallel = Get-DefaultParallelism -SelectedGenerator $Generator
}
if ($Parallel -le 0) {
    throw "Parallel must be greater than zero."
}
if (-not (Test-CommandAvailable -CommandName "cmake")) {
    throw "cmake is not available on PATH."
}
if ($Generator -eq "Ninja" -and -not (Test-CommandAvailable -CommandName "ninja")) {
    throw "Generator 'Ninja' requires 'ninja' on PATH. Install Ninja or pass -Generator `"Visual Studio 17 2022`"."
}
if ($Generator -eq "Ninja" -and -not (Test-CommandAvailable -CommandName "cl.exe")) {
    throw "Generator 'Ninja' requires cl.exe on PATH. Open a Visual Studio x64 Native Tools Developer Command Prompt, then rerun."
}

$resolvedLevels = Normalize-Levels -RawLevels $Levels -AllowedLevels $levelOrder

$scriptDir = Split-Path -Parent $PSCommandPath
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$buildRoot = Join-Path $repoRoot "build"

if (-not (Test-Path -LiteralPath $buildRoot)) {
    New-Item -ItemType Directory -Path $buildRoot | Out-Null
}

Write-Host "Repo root: $repoRoot"
Write-Host "Levels: $($resolvedLevels -join ', ')"
Write-Host "Config: $Config"
Write-Host "Generator: $Generator"
Write-Host "Parallel: $Parallel"
if ($parallelWasImplicit) {
    Write-Host "Parallel policy: auto-capped default"
}
if ($Parallel -gt 8) {
    Write-Warning "Worker count is capped by default to 8. Use -Parallel lower if you still see memory pressure."
}
Write-Host "SkipBuild: $SkipBuild"

foreach ($level in $resolvedLevels) {
    $buildDir = Join-Path $buildRoot "build_impl_$level"
    if (-not (Test-Path -LiteralPath $buildDir)) {
        New-Item -ItemType Directory -Path $buildDir | Out-Null
    }
    else {
        $cacheFile = Join-Path $buildDir "CMakeCache.txt"
        $cachedGenerator = Get-CachedGenerator -CMakeCachePath $cacheFile
        if ($cachedGenerator -and ($cachedGenerator -ne $Generator)) {
            Write-Host "[$level][configure] Cached generator is '$cachedGenerator' but requested '$Generator'. Clearing build directory for reconfigure."
            Remove-Item -LiteralPath $buildDir -Recurse -Force
            New-Item -ItemType Directory -Path $buildDir | Out-Null
        }
    }

    $configureArgs = @(
        "-S", $repoRoot,
        "-B", $buildDir,
        "-G", $Generator,
        "-DCMAKE_BUILD_TYPE=$Config",
        "-DUIPC_BUILD_BENCHMARKS=ON",
        "-DUIPC_BUILD_TESTS=OFF",
        "-DUIPC_BUILD_EXAMPLES=OFF",
        "-DUIPC_BUILD_GUI=OFF",
        "-DUIPC_BUILD_PYBIND=ON",
        "-DUIPC_WITH_CUDA_BACKEND=OFF",
        "-DUIPC_WITH_CUDA_MIXED_BACKEND=ON",
        "-DUIPC_CUDA_MIXED_PRECISION_LEVEL=$level"
    )
    Invoke-Step -Level $level -Phase "configure" -WorkingDirectory $repoRoot -Exe "cmake" -CommandArgs $configureArgs
    $cacheFile = Join-Path $buildDir "CMakeCache.txt"
    if (-not (Test-Path -LiteralPath $cacheFile)) {
        throw "Level '$level' configure completed without generating '$cacheFile'."
    }

    if (-not $SkipBuild) {
        $buildArgs = @(
            "--build", $buildDir,
            "--config", $Config,
            "--parallel", "$Parallel"
        )
        Invoke-Step -Level $level -Phase "build" -WorkingDirectory $repoRoot -Exe "cmake" -CommandArgs $buildArgs
    }
}

Write-Host "Prepared mixed build directories: $($resolvedLevels -join ', ')"

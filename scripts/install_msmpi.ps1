$ErrorActionPreference = 'Stop'
$msmpiRoot = Join-Path $env:RUNNER_TEMP 'msmpi'
New-Item -ItemType Directory -Force -Path $msmpiRoot | Out-Null

$sdkPath = Join-Path $msmpiRoot 'msmpisdk.msi'
$runtimePath = Join-Path $msmpiRoot 'msmpisetup.exe'
$baseUrl = 'https://download.microsoft.com/download/7/2/7/72731ebb-b63c-4170-ade7-836966263a8f'
Invoke-WebRequest -Uri "$baseUrl/msmpisdk.msi" -OutFile $sdkPath
Invoke-WebRequest -Uri "$baseUrl/msmpisetup.exe" -OutFile $runtimePath

$expectedSdkSha256 = '8ccbc77a0cfe1e30f7495a759886b4d99b2c494b9b9ad414dcda1da84c00d3fa'
$expectedRuntimeSha256 = '47443829114d8d8670f77af98939fe876d33eceb35d0ce4e0e85efeec4d87213'
if ((Get-FileHash -Algorithm SHA256 $sdkPath).Hash.ToLowerInvariant() -ne $expectedSdkSha256) {
  throw 'MS-MPI SDK checksum mismatch'
}
if ((Get-FileHash -Algorithm SHA256 $runtimePath).Hash.ToLowerInvariant() -ne $expectedRuntimeSha256) {
  throw 'MS-MPI runtime checksum mismatch'
}

$sdkInstall = Start-Process msiexec.exe -ArgumentList @('/quiet', '/passive', '/qn', '/i', $sdkPath) -Wait -PassThru
if ($sdkInstall.ExitCode -ne 0) {
  throw "MS-MPI SDK installation failed with exit code $($sdkInstall.ExitCode)"
}
$runtimeInstall = Start-Process $runtimePath -ArgumentList @('-unattend') -Wait -PassThru
if ($runtimeInstall.ExitCode -ne 0) {
  throw "MS-MPI runtime installation failed with exit code $($runtimeInstall.ExitCode)"
}

foreach ($name in @('MSMPI_BIN', 'MSMPI_INC', 'MSMPI_LIB64')) {
  $value = [Environment]::GetEnvironmentVariable($name, 'Machine')
  if ([string]::IsNullOrWhiteSpace($value)) {
    throw "MS-MPI did not define $name"
  }
  Add-Content -Path $env:GITHUB_ENV -Value "$name=$value"
}
Add-Content -Path $env:GITHUB_PATH -Value ([Environment]::GetEnvironmentVariable('MSMPI_BIN', 'Machine'))

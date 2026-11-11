# PowerShell version of ghd.sh:  .\tools\gh-docker\ghd.ps1 'git push origin develop'
param([Parameter(Mandatory = $true)][string]$Command)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path "$here\..\..").Path
docker image inspect lme4-gh *> $null
if ($LASTEXITCODE -ne 0) { docker build -q -t lme4-gh $here | Out-Null }
docker run --rm -i -v lme4-gh-ljofreflor:/root/.config/gh -v "${repo}:/work" lme4-gh $Command

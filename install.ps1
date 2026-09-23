# ditch installer for Windows.
#
#   irm https://ditchcensorship.vercel.app/install.ps1 | iex
#
# Picks the release archive for this machine (the AVX2 build on CPUs that have
# it), checks it against the release's SHA256SUMS, puts ditch.exe in
# %LOCALAPPDATA%\Programs\ditch and adds that directory to the user PATH.
# Settings, all optional, as environment variables:
#
#   $env:DITCH_VERSION = "v0.5.0"     a release tag (default: the latest release)
#   $env:DITCH_INSTALL_DIR = "D:\bin"
#   $env:DITCH_BASELINE = "1"         the portable x86-64 build even when AVX2 is there
#   $env:DITCH_UNINSTALL = "1"        remove ditch instead
#
# Everything runs from Install-Ditch at the bottom, so a download cut short
# runs nothing.

function Install-Ditch {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    $repo = 'plyght/ditch'

    function Step($msg) { Write-Host '==> ' -ForegroundColor Cyan -NoNewline; Write-Host $msg }
    function Info($msg) { Write-Host "    $msg" -ForegroundColor DarkGray }
    function Fail($msg) { Write-Host 'error: ' -ForegroundColor Red -NoNewline; Write-Host $msg; throw $msg }

    # Windows PowerShell 5.1 does not offer TLS 1.2 by default.
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}

    $dir = if ($env:DITCH_INSTALL_DIR) { $env:DITCH_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Programs\ditch' }

    if ($env:DITCH_UNINSTALL) {
        Step 'Removing ditch'
        $exe = Join-Path $dir 'ditch.exe'
        if (Test-Path $exe) { Remove-Item -Force $exe; Info "removed $exe" } else { Info "no $exe" }
        $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        if ($userPath) {
            $kept = ($userPath -split ';' | Where-Object { $_ -and ($_.TrimEnd('\') -ne $dir.TrimEnd('\')) }) -join ';'
            if ($kept -ne $userPath) { [Environment]::SetEnvironmentVariable('Path', $kept, 'User'); Info "removed $dir from the user PATH" }
        }
        Write-Host 'done' -ForegroundColor Green
        return
    }

    # The wordmark (src/logo.txt): {W} starts the letters, {R} the thread, {0} resets.
    $logo = @'
           {W}....   ..                  ....{0}
          {W}.+@@+  +@@:                 :@@@{0}
           {W}-@@+  .--   .+#.            %@@       {R}/`''`\       __-`{0}
      {W}:+#+=*@@+ :+**: =#@@++. :+#++*+  %@@-+##*: {R}\\  //  ___-''{0}
     {W}-@@+  -@@+  *@@-  #@@.  =@@=  .*  %@@  :@@#  {R}\-----`'{0}
{R}_-------____{W}@@+{R}`-{W}*{R}_____--`''`{W}%@@{R}_____--```---___--`'{0}
     {W}+@@-  -@@+  *@@-  #@@.  *@@:      %@@  .@@#{0}
     {W}.*@%==+@@#::#@@+. +@@*=:.+%%+=== -@@@- =@@%:{0}
        {W}... .........   ....    ....  ..... .....{0}
'@
    # A console narrower than the art gets a one-line form instead of wrapped rows.
    $cols = 0
    try { $cols = [Console]::WindowWidth } catch { try { $cols = $Host.UI.RawUI.WindowSize.Width } catch {} }
    if ($cols -gt 0 -and $cols -le 66) { $logo = "  {W}ditch {R}--.__.-'``{0}" }
    # A console narrower than the art gets the one-line form instead of wrapped rows.
    $cols = 0
    try { $cols = [Console]::WindowWidth } catch { try { $cols = $Host.UI.RawUI.WindowSize.Width } catch {} }
    if ($cols -gt 0 -and $cols -le 77) { $logo = "  {W}ditch {R}--.__.-'``{0}" }
    if ($Host.UI.SupportsVirtualTerminal) {
        $e = [char]27
        $logo = $logo -replace '\{W\}', "$e[38;5;252m" -replace '\{R\}', "$e[38;5;174m" -replace '\{0\}', "$e[0m"
    } else {
        $logo = $logo -replace '\{[WR0]\}', ''
    }
    Write-Host $logo
    Write-Host "`n  ditch censorship.`n" -ForegroundColor DarkGray

    $arch = $env:PROCESSOR_ARCHITEW6432
    if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
    if ($arch -ne 'AMD64') {
        # An ARM64 Windows runs the x86-64 build under emulation.
        if ($arch -eq 'ARM64') { Info 'ARM64 Windows: installing the x86-64 build, which runs under emulation' }
        else { Fail "no ditch build for $arch; build from source: https://github.com/$repo#install" }
    }

    # The -v3 build needs x86-64-v3: AVX2, FMA, BMI1/2, LZCNT (and F16C, MOVBE,
    # which every such CPU has). .NET Core exposes the flags; Windows
    # PowerShell 5.1 runs on .NET Framework, so ask the kernel for AVX2 there.
    $v3 = $false
    if (-not $env:DITCH_BASELINE -and $arch -eq 'AMD64') {
        try {
            $x86 = 'System.Runtime.Intrinsics.X86'
            $v3 = ([type]"$x86.Avx2")::IsSupported -and ([type]"$x86.Fma")::IsSupported -and
                  ([type]"$x86.Bmi1")::IsSupported -and ([type]"$x86.Bmi2")::IsSupported -and
                  ([type]"$x86.Lzcnt")::IsSupported
        } catch {
            try {
                Add-Type -Namespace DitchInstall -Name Cpu -MemberDefinition '[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(uint feature);'
                $v3 = [DitchInstall.Cpu]::IsProcessorFeaturePresent(40)  # PF_AVX2_INSTRUCTIONS_AVAILABLE
            } catch { $v3 = $false }
        }
    }
    $variant = if ($v3) { '-v3' } else { '' }
    $note = if ($v3) { 'AVX2 + FMA build (x86-64-v3)' } elseif ($env:DITCH_BASELINE) { 'portable x86-64 build (DITCH_BASELINE set)' } else { 'portable x86-64 build (this CPU has no AVX2)' }

    $tag = $env:DITCH_VERSION
    if (-not $tag) {
        Step 'Finding the latest release'
        $tag = (Invoke-RestMethod -UseBasicParsing "https://api.github.com/repos/$repo/releases/latest").tag_name
        if (-not $tag) { Fail "could not find the latest release of $repo" }
    }
    if (-not $tag.StartsWith('v')) { $tag = "v$tag" }
    $name = "ditch-$tag-x86_64-windows-gnu$variant"
    $base = "https://github.com/$repo/releases/download/$tag"
    Info "$tag for x86_64-windows, $note"

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("ditch-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $zip = Join-Path $tmp "$name.zip"
        Step "Downloading $name.zip"
        try { Invoke-WebRequest -UseBasicParsing "$base/$name.zip" -OutFile $zip } catch { Fail "download failed: $base/$name.zip" }

        Step 'Verifying the checksum'
        $sums = $null
        try { $sums = (Invoke-WebRequest -UseBasicParsing "$base/SHA256SUMS").Content } catch {}
        if ($sums) {
            if ($sums -is [byte[]]) { $sums = [Text.Encoding]::UTF8.GetString($sums) }
            $line = $sums -split "`n" | Where-Object { $_ -match " $([regex]::Escape("$name.zip"))\s*$" } | Select-Object -First 1
            if (-not $line) { Fail "$name.zip is not listed in the release's SHA256SUMS" }
            $want = ($line -split '\s+')[0].ToLower()
            $got = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
            if ($want -ne $got) { Fail "checksum mismatch for $name.zip (expected $want, got $got)" }
            Info "sha256 $got"
        } else {
            Write-Host 'warning: ' -ForegroundColor Yellow -NoNewline; Write-Host "this release has no SHA256SUMS; the download is not verified"
        }

        Step 'Installing'
        Expand-Archive -Path $zip -DestinationPath $tmp -Force
        $exe = Join-Path $tmp "$name\ditch.exe"
        if (-not (Test-Path $exe)) { Fail 'the archive has no ditch.exe' }
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Copy-Item -Force $exe (Join-Path $dir 'ditch.exe')
        Info (Join-Path $dir 'ditch.exe')

        # The documented example of every setting, next to where a user config.lua goes.
        $conf = Join-Path $env:USERPROFILE '.config\ditch'
        $example = Join-Path $tmp "$name\config.default.lua"
        if (Test-Path $example) {
            New-Item -ItemType Directory -Force -Path $conf | Out-Null
            Copy-Item -Force $example (Join-Path $conf 'config.default.lua')
            Info "$(Join-Path $conf 'config.default.lua') (every setting, documented)"
        }
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }

    $version = $null
    try { $version = & (Join-Path $dir 'ditch.exe') --version 2>$null } catch {}
    if (-not $version) { Fail 'the installed binary does not run on this machine' }
    Write-Host "`n installed " -ForegroundColor Green -NoNewline; Write-Host " $version" -ForegroundColor DarkGray

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $onPath = ($userPath -split ';' | Where-Object { $_.TrimEnd('\') -eq $dir.TrimEnd('\') })
    if (-not $onPath) {
        $newPath = if ($userPath) { "$userPath;$dir" } else { $dir }
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        $env:Path = "$env:Path;$dir"
        Write-Host "`nAdded $dir to your user PATH; open a new terminal to use it everywhere."
    }

    Write-Host "`nGet started:"
    Write-Host '    ditch --help'
    Write-Host '    ditch Qwen/Qwen2.5-0.5B-Instruct   ' -NoNewline; Write-Host '# abliterate a model from the Hub' -ForegroundColor DarkGray
    Write-Host ''
}

Install-Ditch

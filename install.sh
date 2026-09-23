#!/bin/sh
# ditch installer for Linux and macOS.
#
#   curl -fsSL https://ditchcensorship.vercel.app/install | sh
#
# Picks the release archive for this machine (the AVX2 build on x86-64 CPUs
# that have it), checks it against the release's SHA256SUMS and puts `ditch`
# in ~/.local/bin. Settings, all optional:
#
#   DITCH_VERSION=v0.5.0         a release tag (default: the latest release)
#   DITCH_INSTALL_DIR=/usr/local/bin
#   DITCH_BASELINE=1             the portable x86-64 build even when AVX2 is there
#   NO_COLOR=1                   plain output
#
#   curl -fsSL https://ditchcensorship.vercel.app/install | sh -s -- --uninstall
#
# Everything runs from main at the bottom, so a download cut short runs nothing.

set -eu

REPO="plyght/ditch"

if [ -t 2 ] && [ -z "${NO_COLOR:-}" ]; then
	BOLD=$(printf '\033[1m') DIM=$(printf '\033[2m') RED=$(printf '\033[31m')
	GREEN=$(printf '\033[32m') YELLOW=$(printf '\033[33m') CYAN=$(printf '\033[36m')
	RESET=$(printf '\033[0m')
else
	BOLD='' DIM='' RED='' GREEN='' YELLOW='' CYAN='' RESET=''
fi

step() { printf '%s==>%s %s\n' "$CYAN" "$RESET" "$*" >&2; }
info() { printf '    %s%s%s\n' "$DIM" "$*" "$RESET" >&2; }
warn() { printf '%swarning:%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die() {
	printf '%serror:%s %s\n' "$RED" "$RESET" "$*" >&2
	exit 1
}

# The wordmark (src/logo.txt): {W} starts the letters, {R} the thread, {0} resets.
banner() {
	if [ -n "$RESET" ]; then
		W=$(printf '\033[38;5;252m') R=$(printf '\033[38;5;174m') Z=$(printf '\033[0m')
	else
		W='' R='' Z=''
	fi
	# A terminal narrower than the art gets a one-line form instead of wrapped rows.
	cols=$( (stty size </dev/tty) 2>/dev/null | cut -d ' ' -f 2)
	case "$cols" in '' | *[!0-9]*) cols=${COLUMNS:-0} ;; esac
	case "$cols" in '' | *[!0-9]*) cols=0 ;; esac
	if [ "$cols" -gt 0 ] && [ "$cols" -le 66 ]; then
		printf '  %sditch %s--.__.-'"'"'`%s\n' "$W" "$R" "$Z" >&2
	else
		sed -e "s/{W}/$W/g" -e "s/{R}/$R/g" -e "s/{0}/$Z/g" >&2 <<'LOGO'
           {W}....   ..                  ....{0}
          {W}.+@@+  +@@:                 :@@@{0}
           {W}-@@+  .--   .+#.            %@@       {R}/`''`\       __-`{0}
      {W}:+#+=*@@+ :+**: =#@@++. :+#++*+  %@@-+##*: {R}\\  //  ___-''{0}
     {W}-@@+  -@@+  *@@-  #@@.  =@@=  .*  %@@  :@@#  {R}\-----`'{0}
{R}_-------____{W}@@+{R}`-{W}*{R}_____--`''`{W}%@@{R}_____--```---___--`'{0}
     {W}+@@-  -@@+  *@@-  #@@.  *@@:      %@@  .@@#{0}
     {W}.*@%==+@@#::#@@+. +@@*=:.+%%+=== -@@@- =@@%:{0}
        {W}... .........   ....    ....  ..... .....{0}
LOGO
	fi
	printf '\n  %sditch censorship.%s\n\n' "$DIM" "$RESET" >&2
}

have() { command -v "$1" >/dev/null 2>&1; }

# fetch URL FILE: download to a file, failing on HTTP errors.
fetch() {
	if have curl; then
		curl -fsSL --retry 3 --proto '=https' --tlsv1.2 -o "$2" "$1"
	elif have wget; then
		wget -q --https-only -O "$2" "$1"
	else
		die "curl or wget is needed to download ditch"
	fi
}

# The latest release's tag, read from the redirect of /releases/latest (no API
# token or rate limit involved).
latest_tag() {
	url="https://github.com/$REPO/releases/latest"
	if have curl; then
		final=$(curl -fsSLI --retry 3 -o /dev/null -w '%{url_effective}' "$url")
	else
		final=$(wget -q --max-redirect=5 -S --spider "$url" 2>&1 | sed -n 's/^ *[Ll]ocation: *//p' | tail -n 1 | tr -d '\r')
	fi
	tag=${final##*/}
	case "$tag" in
	v*) printf '%s\n' "$tag" ;;
	*) die "could not find the latest release of $REPO (got '$final')" ;;
	esac
}

# Whether this x86-64 CPU runs the -v3 build: AVX2, FMA, BMI1/2, F16C, MOVBE
# and LZCNT, the x86-64-v3 level the build targets.
has_x86_64_v3() {
	case "$1" in
	linux)
		[ -r /proc/cpuinfo ] || return 1
		flags=$(grep -m 1 '^flags' /proc/cpuinfo) || return 1
		for f in avx2 fma bmi1 bmi2 f16c movbe abm; do
			case " $flags " in *" $f "*) ;; *) return 1 ;; esac
		done
		;;
	macos)
		feats="$(sysctl -n machdep.cpu.features machdep.cpu.leaf7_features machdep.cpu.extfeatures 2>/dev/null | tr '\n' ' ')"
		for f in AVX2 FMA BMI1 BMI2 F16C MOVBE LZCNT; do
			case " $feats " in *" $f "*) ;; *) return 1 ;; esac
		done
		;;
	*) return 1 ;;
	esac
}

detect_target() {
	case "$(uname -s)" in
	Linux) os=linux ;;
	Darwin) os=macos ;;
	MINGW* | MSYS* | CYGWIN*) die "on Windows run: irm https://ditchcensorship.vercel.app/install.ps1 | iex" ;;
	*) die "no ditch build for $(uname -s); build from source: https://github.com/$REPO#install" ;;
	esac
	case "$(uname -m)" in
	x86_64 | amd64) arch=x86_64 ;;
	aarch64 | arm64) arch=aarch64 ;;
	*) die "no ditch build for $(uname -m); build from source: https://github.com/$REPO#install" ;;
	esac
	# An x86-64 shell under Rosetta on Apple Silicon: the native build is faster.
	if [ "$os" = macos ] && [ "$arch" = x86_64 ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
		arch=aarch64
	fi

	case "$os" in
	linux) TARGET="$arch-linux-musl" ;;
	macos) TARGET="$arch-macos" ;;
	esac
	VARIANT=""
	CPU_NOTE="baseline build"
	if [ "$arch" = x86_64 ]; then
		if [ -n "${DITCH_BASELINE:-}" ]; then
			CPU_NOTE="portable x86-64 build (DITCH_BASELINE set)"
		elif has_x86_64_v3 "$os"; then
			VARIANT="-v3"
			CPU_NOTE="AVX2 + FMA build (x86-64-v3)"
		else
			CPU_NOTE="portable x86-64 build (this CPU has no AVX2)"
		fi
	else
		CPU_NOTE="NEON build"
	fi
}

sha256_of() {
	if have sha256sum; then
		sha256sum "$1" | cut -d ' ' -f 1
	elif have shasum; then
		shasum -a 256 "$1" | cut -d ' ' -f 1
	elif have openssl; then
		openssl dgst -sha256 -r "$1" | cut -d ' ' -f 1
	else
		return 1
	fi
}

uninstall() {
	dir="${DITCH_INSTALL_DIR:-$HOME/.local/bin}"
	step "Removing ditch"
	if [ -e "$dir/ditch" ]; then
		rm -f "$dir/ditch"
		info "removed $dir/ditch"
	else
		info "no $dir/ditch"
	fi
	cache="${DITCH_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/ditch}"
	if [ -d "$cache" ]; then
		info "the model cache is left in $cache ($(du -sh "$cache" 2>/dev/null | cut -f 1)); rm -rf it to free the space"
	fi
	printf '%sdone%s\n' "$GREEN" "$RESET" >&2
}

main() {
	for arg in "$@"; do
		case "$arg" in
		--uninstall) uninstall; return ;;
		-h | --help)
			cat >&2 <<'EOF2'
ditch installer

  curl -fsSL https://ditchcensorship.vercel.app/install | sh
  curl -fsSL https://ditchcensorship.vercel.app/install | sh -s -- --uninstall

  DITCH_VERSION=v0.5.0         a release tag (default: the latest release)
  DITCH_INSTALL_DIR=DIR        where to put ditch (default: ~/.local/bin)
  DITCH_BASELINE=1             the portable x86-64 build even when AVX2 is there
  NO_COLOR=1                   plain output
EOF2
			return
			;;
		*) die "unknown option: $arg" ;;
		esac
	done

	banner
	detect_target
	tag="${DITCH_VERSION:-}"
	if [ -z "$tag" ]; then
		step "Finding the latest release"
		tag=$(latest_tag)
	fi
	case "$tag" in v*) ;; *) tag="v$tag" ;; esac
	name="ditch-$tag-$TARGET$VARIANT"
	base="https://github.com/$REPO/releases/download/$tag"
	info "$tag for $TARGET, $CPU_NOTE"

	tmp=$(mktemp -d 2>/dev/null || mktemp -d -t ditch)
	trap 'rm -rf "$tmp"' EXIT INT TERM

	step "Downloading $name.tar.gz"
	fetch "$base/$name.tar.gz" "$tmp/$name.tar.gz" || die "download failed: $base/$name.tar.gz"

	step "Verifying the checksum"
	if fetch "$base/SHA256SUMS" "$tmp/SHA256SUMS" 2>/dev/null; then
		want=$(grep " $name.tar.gz\$" "$tmp/SHA256SUMS" | cut -d ' ' -f 1)
		[ -n "$want" ] || die "$name.tar.gz is not listed in the release's SHA256SUMS"
		got=$(sha256_of "$tmp/$name.tar.gz") || die "no sha256sum, shasum or openssl to check the download with"
		[ "$want" = "$got" ] || die "checksum mismatch for $name.tar.gz (expected $want, got $got)"
		info "sha256 $got"
	else
		warn "this release has no SHA256SUMS; the download is not verified"
	fi

	step "Installing"
	tar -xzf "$tmp/$name.tar.gz" -C "$tmp"
	[ -f "$tmp/$name/ditch" ] || die "the archive has no ditch binary"
	dir="${DITCH_INSTALL_DIR:-$HOME/.local/bin}"
	mkdir -p "$dir" 2>/dev/null || die "cannot create $dir (set DITCH_INSTALL_DIR to a directory you can write)"
	[ -w "$dir" ] || die "cannot write to $dir (set DITCH_INSTALL_DIR, or run with sudo for a system directory)"
	# Move into place from the same directory so a running ditch is never half-written.
	cp "$tmp/$name/ditch" "$dir/.ditch.new"
	chmod 755 "$dir/.ditch.new"
	mv -f "$dir/.ditch.new" "$dir/ditch"
	# macOS marks downloads as quarantined; the binary is not notarised.
	if [ "$(uname -s)" = Darwin ] && have xattr; then
		xattr -d com.apple.quarantine "$dir/ditch" 2>/dev/null || true
	fi
	info "$dir/ditch"

	# The documented example of every setting, next to where a user config.lua goes.
	conf="${XDG_CONFIG_HOME:-$HOME/.config}/ditch"
	if mkdir -p "$conf" 2>/dev/null && [ -f "$tmp/$name/config.default.lua" ]; then
		cp "$tmp/$name/config.default.lua" "$conf/config.default.lua"
		info "$conf/config.default.lua (every setting, documented)"
	fi

	version=$("$dir/ditch" --version 2>/dev/null || true)
	[ -n "$version" ] || die "the installed binary does not run on this machine"
	printf '\n%s%s installed%s  %s%s%s\n' "$GREEN" "$BOLD" "$RESET" "$DIM" "$version" "$RESET" >&2

	case ":$PATH:" in
	*":$dir:"*) ;;
	*)
		shell_rc="your shell's startup file"
		case "${SHELL:-}" in
		*/zsh) shell_rc="~/.zshrc" ;;
		*/bash) shell_rc="~/.bashrc" ;;
		*/fish) shell_rc="~/.config/fish/config.fish" ;;
		esac
		printf '\n%s is not on your PATH. Add it with this line in %s:\n' "$dir" "$shell_rc" >&2
		case "${SHELL:-}" in
		*/fish) printf '    %sfish_add_path %s%s\n' "$BOLD" "$dir" "$RESET" >&2 ;;
		*) printf '    %sexport PATH="%s:$PATH"%s\n' "$BOLD" "$dir" "$RESET" >&2 ;;
		esac
		;;
	esac

	printf '\nGet started:\n' >&2
	printf '    %sditch --help%s\n' "$BOLD" "$RESET" >&2
	printf '    %sditch Qwen/Qwen2.5-0.5B-Instruct%s   %s# abliterate a model from the Hub%s\n\n' "$BOLD" "$RESET" "$DIM" "$RESET" >&2
}

main "$@"

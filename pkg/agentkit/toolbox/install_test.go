package toolbox

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/klauspost/compress/zstd"
	"github.com/orka-agents/agentkit/pkg/utils"
)

const testMember = "linux-amd64/helm"

// elfHeader returns the first bytes of an ELF executable for machine.
func elfHeader(class, data byte, machine uint16) []byte {
	h := make([]byte, 64)
	copy(h, "\x7fELF")
	h[4], h[5], h[6] = class, data, 1
	h[16] = 2 // ET_EXEC
	if data == 2 {
		binary.BigEndian.PutUint16(h[18:], machine)
	} else {
		binary.LittleEndian.PutUint16(h[18:], machine)
	}
	return append(h, "payload"...)
}

var (
	amd64ELF = elfHeader(2, 1, 62)
	arm64ELF = elfHeader(2, 1, 183)
	s390xELF = elfHeader(2, 2, 22)
)

type tarEntry struct {
	hdr  tar.Header
	body []byte
}

func file(name string, body []byte) tarEntry {
	return tarEntry{hdr: tar.Header{Name: name, Typeflag: tar.TypeReg, Mode: 0o755, Size: int64(len(body))}, body: body}
}

func tarBytes(t *testing.T, entries ...tarEntry) []byte {
	t.Helper()
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for _, e := range entries {
		hdr := e.hdr
		if err := tw.WriteHeader(&hdr); err != nil {
			t.Fatalf("WriteHeader(%s): %v", hdr.Name, err)
		}
		if _, err := tw.Write(e.body); err != nil {
			t.Fatalf("Write(%s): %v", hdr.Name, err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func gzipBytes(t *testing.T, in []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	if _, err := zw.Write(in); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func zstdBytes(t *testing.T, in []byte) []byte {
	t.Helper()
	zw, err := zstd.NewWriter(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer zw.Close()
	return zw.EncodeAll(in, nil)
}

func install(t *testing.T, download []byte, member, arch string, limits Limits) (string, Result, error) {
	t.Helper()
	dir := t.TempDir()
	src := filepath.Join(dir, "download")
	if err := os.WriteFile(src, download, 0o600); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(dir, "out")
	if err := os.Mkdir(dest, 0o755); err != nil {
		t.Fatal(err)
	}
	result, err := Install(Request{Source: src, Member: member, Arch: arch, Dest: dest}, limits)
	return dest, result, err
}

func assertInstalled(t *testing.T, dest string, result Result, want []byte) {
	t.Helper()
	got, err := os.ReadFile(filepath.Join(dest, ToolFile))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("installed %q, want %q", got, want)
	}
	sum := sha256.Sum256(want)
	if result.SHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("SHA256 = %s, want %x", result.SHA256, sum)
	}
	var written Result
	dt, err := os.ReadFile(filepath.Join(dest, ResultFile))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(dt, &written); err != nil || written != result {
		t.Fatalf("%s = %s (%v), want %+v", ResultFile, dt, err, result)
	}
	info, err := os.Stat(filepath.Join(dest, ToolFile))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o555 {
		t.Fatalf("mode = %v, want 0555", info.Mode().Perm())
	}
}

func assertInstallError(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("Install error = %v, want it to contain %q", err, want)
	}
}

func TestInstallPlainDownload(t *testing.T) {
	for name, tc := range map[string]struct {
		body []byte
		arch string
	}{
		"amd64 ELF": {amd64ELF, utils.PlatformAMD64},
		"arm64 ELF": {arm64ELF, utils.PlatformARM64},
		"script":    {[]byte("#!/bin/sh\necho hi\n"), utils.PlatformARM64},
	} {
		t.Run(name, func(t *testing.T) {
			dest, result, err := install(t, tc.body, "", tc.arch, DefaultLimits)
			if err != nil {
				t.Fatalf("Install: %v", err)
			}
			assertInstalled(t, dest, result, tc.body)
		})
	}
}

func TestInstallRejectsWrongArchitecture(t *testing.T) {
	_, _, err := install(t, amd64ELF, "", utils.PlatformARM64, DefaultLimits)
	assertInstallError(t, err, "the tool is an ELF binary for amd64, not arm64")

	_, _, err = install(t, gzipBytes(t, tarBytes(t, file(testMember, arm64ELF))), testMember, utils.PlatformAMD64, DefaultLimits)
	assertInstallError(t, err, "the tool is an ELF binary for arm64, not amd64")

	_, _, err = install(t, s390xELF, "", utils.PlatformAMD64, DefaultLimits)
	assertInstallError(t, err, "the tool is an ELF binary for EM_S390 (ELFCLASS64, ELFDATA2MSB), not amd64")
}

func TestInstallRejectsUnsupportedArchitecture(t *testing.T) {
	for _, arch := range []string{"mips64le", "s390x", "riscv64", ""} {
		_, _, err := install(t, []byte("#!/bin/sh\n"), "", arch, DefaultLimits)
		assertInstallError(t, err, fmt.Sprintf("architecture %q is not supported; toolboxes are built for amd64 and arm64", arch))
	}
}

func TestInstallRejectsNonExecutable(t *testing.T) {
	archive := gzipBytes(t, tarBytes(t, file(testMember, amd64ELF)))
	_, _, err := install(t, archive, "", utils.PlatformAMD64, DefaultLimits)
	assertInstallError(t, err, "neither an ELF binary nor a #! script; set extract")
}

func TestInstallPlainDownloadLimit(t *testing.T) {
	limits := DefaultLimits
	limits.MaxFileBytes = int64(len(amd64ELF) - 1)
	_, _, err := install(t, amd64ELF, "", utils.PlatformAMD64, limits)
	assertInstallError(t, err, "the tool is larger than")
}

func TestInstallExtractsFromArchiveFormats(t *testing.T) {
	plain := tarBytes(t,
		tarEntry{hdr: tar.Header{Name: "./", Typeflag: tar.TypeDir, Mode: 0o755}},
		tarEntry{hdr: tar.Header{Name: "./linux-amd64/", Typeflag: tar.TypeDir, Mode: 0o755}},
		file("./linux-amd64/LICENSE", []byte("license")),
		// Links and other members elsewhere in the archive are never written.
		tarEntry{hdr: tar.Header{Name: "./linux-amd64/latest", Typeflag: tar.TypeSymlink, Linkname: "/etc/passwd"}},
		file("./linux-amd64/helm", amd64ELF),
	)
	for name, archive := range map[string][]byte{
		"tar":     plain,
		"tar.gz":  gzipBytes(t, plain),
		"tar.zst": zstdBytes(t, plain),
	} {
		t.Run(name, func(t *testing.T) {
			dest, result, err := install(t, archive, testMember, utils.PlatformAMD64, DefaultLimits)
			if err != nil {
				t.Fatalf("Install: %v", err)
			}
			assertInstalled(t, dest, result, amd64ELF)
		})
	}
}

func TestInstallRejectsCraftedTarArchives(t *testing.T) {
	tests := []struct {
		name    string
		entries []tarEntry
		limits  func(*Limits)
		want    string
	}{
		{
			name:    "dotdot path",
			entries: []tarEntry{file("../../etc/cron.d/x", amd64ELF), file(testMember, amd64ELF)},
			want:    `a path with '..': "../../etc/cron.d/x"`,
		},
		{
			name:    "dotdot inside path",
			entries: []tarEntry{file("linux-amd64/../../helm", amd64ELF)},
			want:    "a path with '..'",
		},
		{
			name:    "absolute path",
			entries: []tarEntry{file("/usr/bin/helm", amd64ELF), file(testMember, amd64ELF)},
			want:    `an absolute path "/usr/bin/helm"`,
		},
		{
			name:    "symlink member",
			entries: []tarEntry{{hdr: tar.Header{Name: testMember, Typeflag: tar.TypeSymlink, Linkname: "/etc/shadow"}}},
			want:    `archive member "linux-amd64/helm" is a symlink, not a regular file`,
		},
		{
			name: "hard link member",
			entries: []tarEntry{
				file("linux-amd64/real", amd64ELF),
				{hdr: tar.Header{Name: testMember, Typeflag: tar.TypeLink, Linkname: "linux-amd64/real"}},
			},
			want: `archive member "linux-amd64/helm" is a hard link, not a regular file`,
		},
		{
			name:    "device member",
			entries: []tarEntry{{hdr: tar.Header{Name: testMember, Typeflag: tar.TypeChar, Devmajor: 1, Devminor: 1, Mode: 0o666}}},
			want:    "is a device, not a regular file",
		},
		{
			name:    "fifo member",
			entries: []tarEntry{{hdr: tar.Header{Name: testMember, Typeflag: tar.TypeFifo, Mode: 0o666}}},
			want:    "is a named pipe, not a regular file",
		},
		{
			name:    "directory member",
			entries: []tarEntry{{hdr: tar.Header{Name: testMember + "/", Typeflag: tar.TypeDir, Mode: 0o755}}},
			want:    "is a directory, not a regular file",
		},
		{
			name:    "duplicate member",
			entries: []tarEntry{file(testMember, amd64ELF), file("./"+testMember, []byte("#!/bin/sh\n"))},
			want:    `the archive contains "linux-amd64/helm" more than once`,
		},
		{
			name:    "missing member",
			entries: []tarEntry{file("helm-v3/linux-amd64/helm", amd64ELF)},
			want:    `the archive has no "linux-amd64/helm"; entries with the same name: helm-v3/linux-amd64/helm`,
		},
		{
			name:    "oversized member",
			entries: []tarEntry{file(testMember, amd64ELF)},
			limits:  func(l *Limits) { l.MaxFileBytes = 10 },
			want:    `archive member "linux-amd64/helm" is larger than 10 bytes`,
		},
		{
			name:    "too many entries",
			entries: []tarEntry{file("a", nil), file("b", nil), file(testMember, amd64ELF)},
			limits:  func(l *Limits) { l.MaxEntries = 2 },
			want:    "the archive has more than 2 entries",
		},
		{
			name:    "expands too far",
			entries: []tarEntry{file("padding", make([]byte, 4096)), file(testMember, amd64ELF)},
			limits:  func(l *Limits) { l.MaxUncompressedBytes = 2048 },
			want:    "the archive expands to more than 2048 bytes",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			limits := DefaultLimits
			if tt.limits != nil {
				tt.limits(&limits)
			}
			_, _, err := install(t, gzipBytes(t, tarBytes(t, tt.entries...)), testMember, utils.PlatformAMD64, limits)
			assertInstallError(t, err, tt.want)
		})
	}
}

func TestInstallRejectsArchiveLimitsAndFormats(t *testing.T) {
	archive := gzipBytes(t, tarBytes(t, file(testMember, amd64ELF)))
	limits := DefaultLimits
	limits.MaxArchiveBytes = int64(len(archive) - 1)
	_, _, err := install(t, archive, testMember, utils.PlatformAMD64, limits)
	assertInstallError(t, err, "the archive is larger than")

	_, _, err = install(t, []byte{0xfd, '7', 'z', 'X', 'Z', 0x00, 1, 2, 3}, testMember, utils.PlatformAMD64, DefaultLimits)
	assertInstallError(t, err, "xz archives are not supported")

	_, _, err = install(t, []byte("PK\x03\x04 a zip local header"), testMember, utils.PlatformAMD64, DefaultLimits)
	assertInstallError(t, err, "zip archives are not supported")

	_, _, err = install(t, bytes.Repeat(amd64ELF, 20), testMember, utils.PlatformAMD64, DefaultLimits)
	assertInstallError(t, err, "not a supported archive")
}

func TestInstallRejectsNonRegularSource(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target")
	if err := os.WriteFile(target, amd64ELF, 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	for src, want := range map[string]string{
		link: "the tool is a symlink, not a regular file",
		dir:  "the tool is a directory, not a regular file",
	} {
		dest := t.TempDir()
		_, err := Install(Request{Source: src, Arch: utils.PlatformAMD64, Dest: dest}, DefaultLimits)
		assertInstallError(t, err, want)
	}
}

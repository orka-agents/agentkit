// Package toolbox holds the helpers the AgentKit frontend runs inside its own
// image during a kind: Toolbox build. Install takes exactly one regular file
// from a download, an archive, or an image copy and checks it; Verify runs the
// smoke commands of the verify stage. Both read untrusted bytes, so every
// limit fails closed.
package toolbox

import (
	"archive/tar"
	"archive/zip"
	"bufio"
	"bytes"
	"compress/bzip2"
	"compress/gzip"
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	pathpkg "path"
	"path/filepath"
	"strings"

	"github.com/klauspost/compress/zstd"
	"github.com/orka-agents/agentkit/pkg/utils"
)

const (
	// ToolFile and ResultFile are the names Install writes into its destination.
	ToolFile   = "tool"
	ResultFile = "result.json"

	// maxZstdWindow bounds decoder memory for a hostile zstd frame.
	maxZstdWindow = 128 << 20
	// maxSuggestions caps the "did you mean" member names in an error.
	maxSuggestions = 5
)

// Limits bound the work an untrusted download can cause.
type Limits struct {
	// MaxArchiveBytes caps the size of a downloaded archive.
	MaxArchiveBytes int64
	// MaxUncompressedBytes caps the bytes an archive expands to.
	MaxUncompressedBytes int64
	// MaxFileBytes caps the installed tool. It matches Orka's per-file limit.
	MaxFileBytes int64
	// MaxEntries caps the number of archive entries.
	MaxEntries int
}

// DefaultLimits are the limits every toolbox build uses.
var DefaultLimits = Limits{
	MaxArchiveBytes:      1 << 30,
	MaxUncompressedBytes: 2 << 30,
	MaxFileBytes:         512 << 20,
	MaxEntries:           100_000,
}

// Request describes one tool to install.
type Request struct {
	// Source is the downloaded file or the file copied out of an image.
	Source string
	// Member is the archive path to extract. Empty installs Source itself.
	Member string
	// Arch is the GOARCH the tool must be built for.
	Arch string
	// Dest is the directory that receives ToolFile and ResultFile.
	Dest string
}

// Result records the installed tool for the toolbox inventory.
type Result struct {
	SHA256 string `json:"sha256"`
}

// Install copies exactly one regular file to Dest/ToolFile, checks that it is
// an executable for Arch, and writes its digest to Dest/ResultFile.
func Install(req Request, limits Limits) (Result, error) {
	// Lstat before opening: opening a named pipe would block and opening a
	// device could have side effects. SameFile then rules out a swap.
	info, err := os.Lstat(req.Source)
	if err != nil {
		return Result{}, err
	}
	if !info.Mode().IsRegular() {
		return Result{}, fmt.Errorf("the tool is a %s, not a regular file", fileType(info.Mode()))
	}
	src, err := os.Open(req.Source)
	if err != nil {
		return Result{}, err
	}
	defer src.Close()
	if opened, err := src.Stat(); err != nil || !os.SameFile(info, opened) {
		return Result{}, fmt.Errorf("%s changed while it was opened", req.Source)
	}

	out, err := os.OpenFile(filepath.Join(req.Dest, ToolFile), os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o555)
	if err != nil {
		return Result{}, err
	}
	defer out.Close()
	digest := sha256.New()
	write := func(r io.Reader) error {
		n, err := io.Copy(io.MultiWriter(out, digest), io.LimitReader(r, limits.MaxFileBytes+1))
		if err != nil {
			return err
		}
		if n > limits.MaxFileBytes {
			return fmt.Errorf("the tool is larger than %d bytes", limits.MaxFileBytes)
		}
		return nil
	}

	if req.Member == "" {
		err = write(src)
	} else {
		if info.Size() > limits.MaxArchiveBytes {
			return Result{}, fmt.Errorf("the archive is larger than %d bytes", limits.MaxArchiveBytes)
		}
		err = extractMember(src, info.Size(), req.Member, write, limits)
	}
	if err != nil {
		return Result{}, err
	}
	if err := checkExecutable(out, req.Arch); err != nil {
		return Result{}, err
	}
	if err := out.Close(); err != nil {
		return Result{}, err
	}

	result := Result{SHA256: hex.EncodeToString(digest.Sum(nil))}
	dt, err := json.Marshal(result)
	if err != nil {
		return Result{}, err
	}
	return result, os.WriteFile(filepath.Join(req.Dest, ResultFile), dt, 0o600)
}

// extractMember streams the archive and passes the single entry named member
// to write. Every entry is checked, so a malformed or oversized archive fails
// even when the member itself looks fine.
func extractMember(src *os.File, size int64, member string, write func(io.Reader) error, limits Limits) error {
	br := bufio.NewReader(src)
	magic, _ := br.Peek(6)
	switch {
	case bytes.HasPrefix(magic, []byte("PK\x03\x04")), bytes.HasPrefix(magic, []byte("PK\x05\x06")):
		return extractZipMember(src, size, member, write, limits)
	case bytes.HasPrefix(magic, []byte{0x1f, 0x8b}):
		zr, err := gzip.NewReader(br)
		if err != nil {
			return fmt.Errorf("reading gzip archive: %w", err)
		}
		defer zr.Close()
		return extractTarMember(zr, member, write, limits)
	case bytes.HasPrefix(magic, []byte("BZh")):
		return extractTarMember(bzip2.NewReader(br), member, write, limits)
	case bytes.HasPrefix(magic, []byte{0x28, 0xb5, 0x2f, 0xfd}):
		zr, err := zstd.NewReader(br, zstd.WithDecoderConcurrency(1), zstd.WithDecoderMaxMemory(maxZstdWindow))
		if err != nil {
			return fmt.Errorf("reading zstd archive: %w", err)
		}
		defer zr.Close()
		return extractTarMember(zr, member, write, limits)
	case bytes.HasPrefix(magic, []byte{0xfd, '7', 'z', 'X', 'Z', 0x00}):
		return errors.New("xz archives are not supported; use a tar, tar.gz, tar.bz2, tar.zst, or zip download")
	default:
		return extractTarMember(br, member, write, limits)
	}
}

func extractTarMember(r io.Reader, member string, write func(io.Reader) error, limits Limits) error {
	expanded := &limitedReader{
		r:         r,
		remaining: limits.MaxUncompressedBytes,
		err:       fmt.Errorf("the archive expands to more than %d bytes", limits.MaxUncompressedBytes),
	}
	tr := tar.NewReader(expanded)
	scan := memberScan{member: member, limits: limits}
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		// ErrInsecurePath still returns the header; cleanEntryName rejects it.
		if err != nil && !errors.Is(err, tar.ErrInsecurePath) {
			if errors.Is(err, expanded.err) {
				return err
			}
			if scan.entries == 0 {
				return errors.New("the download is not a supported archive (tar, tar.gz, tar.bz2, tar.zst, or zip)")
			}
			return fmt.Errorf("reading tar archive: %w", err)
		}
		match, err := scan.next(hdr.Name, tarEntryType(hdr), hdr.Size)
		if err != nil {
			return err
		}
		if match {
			if err := write(tr); err != nil {
				return err
			}
		}
	}
	return scan.finish()
}

func extractZipMember(src io.ReaderAt, size int64, member string, write func(io.Reader) error, limits Limits) error {
	zr, err := zip.NewReader(src, size)
	if err != nil && !errors.Is(err, zip.ErrInsecurePath) {
		return fmt.Errorf("reading zip archive: %w", err)
	}
	if len(zr.File) > limits.MaxEntries {
		return fmt.Errorf("the archive has more than %d entries", limits.MaxEntries)
	}
	scan := memberScan{member: member, limits: limits}
	var expanded int64
	var match *zip.File
	for _, f := range zr.File {
		size := zipEntrySize(f)
		if size > limits.MaxUncompressedBytes-expanded {
			return fmt.Errorf("the archive expands to more than %d bytes", limits.MaxUncompressedBytes)
		}
		expanded += size
		ok, err := scan.next(f.Name, fileType(f.Mode()), size)
		if err != nil {
			return err
		}
		if ok {
			match = f
		}
	}
	if err := scan.finish(); err != nil {
		return err
	}
	rc, err := match.Open()
	if err != nil {
		return err
	}
	defer rc.Close()
	return write(rc)
}

// zipEntrySize is an entry's declared uncompressed size, clamped to int64.
// write's limit still bounds what a lying entry can expand to.
func zipEntrySize(f *zip.File) int64 {
	if f.UncompressedSize64 > math.MaxInt64 {
		return math.MaxInt64
	}
	return int64(f.UncompressedSize64)
}

// memberScan applies the per-entry rules shared by tar and zip.
type memberScan struct {
	member      string
	limits      Limits
	entries     int
	found       bool
	suggestions []string
}

// next checks one entry; kind is fileType's description of it.
func (s *memberScan) next(name, kind string, size int64) (bool, error) {
	s.entries++
	if s.entries > s.limits.MaxEntries {
		return false, fmt.Errorf("the archive has more than %d entries", s.limits.MaxEntries)
	}
	clean, err := cleanEntryName(name)
	if err != nil {
		return false, err
	}
	if clean != s.member {
		if pathpkg.Base(clean) == pathpkg.Base(s.member) && len(s.suggestions) < maxSuggestions {
			s.suggestions = append(s.suggestions, clean)
		}
		return false, nil
	}
	if s.found {
		return false, fmt.Errorf("the archive contains %q more than once", s.member)
	}
	s.found = true
	if kind != regularFile {
		return false, fmt.Errorf("archive member %q is a %s, not a regular file", s.member, kind)
	}
	if size > s.limits.MaxFileBytes {
		return false, fmt.Errorf("archive member %q is larger than %d bytes", s.member, s.limits.MaxFileBytes)
	}
	return true, nil
}

func (s *memberScan) finish() error {
	if s.found {
		return nil
	}
	if len(s.suggestions) > 0 {
		return fmt.Errorf("the archive has no %q; entries with the same name: %s", s.member, strings.Join(s.suggestions, ", "))
	}
	return fmt.Errorf("the archive has no %q", s.member)
}

// cleanEntryName rejects absolute names and any ".." component, then strips
// "./" prefixes and trailing slashes so names compare like extract paths.
func cleanEntryName(name string) (string, error) {
	if strings.HasPrefix(name, "/") {
		return "", fmt.Errorf("the archive contains an absolute path %q", name)
	}
	for _, part := range strings.Split(name, "/") {
		if part == ".." {
			return "", fmt.Errorf("the archive contains a path with '..': %q", name)
		}
	}
	return pathpkg.Clean(name), nil
}

// checkExecutable requires an ELF binary for arch or a #! script. Anything
// else is almost always an archive downloaded without extract.
func checkExecutable(f *os.File, arch string) error {
	var header [20]byte
	n, err := f.ReadAt(header[:], 0)
	if err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	if n >= 2 && string(header[:2]) == "#!" {
		return nil
	}
	if n < len(header) || string(header[:4]) != elf.ELFMAG {
		return errors.New("the tool is neither an ELF binary nor a #! script; set extract to take a file from an archive")
	}

	want, known := elfArchs[arch]
	if !known {
		return nil
	}
	var order binary.ByteOrder = binary.LittleEndian
	if elf.Data(header[elf.EI_DATA]) == elf.ELFDATA2MSB {
		order = binary.BigEndian
	}
	got := elfArch{
		class:   elf.Class(header[elf.EI_CLASS]),
		data:    elf.Data(header[elf.EI_DATA]),
		machine: elf.Machine(order.Uint16(header[18:20])),
	}
	if got != want {
		return fmt.Errorf("the tool is an ELF binary for %s, not %s", got, arch)
	}
	return nil
}

type elfArch struct {
	class   elf.Class
	data    elf.Data
	machine elf.Machine
}

func (a elfArch) String() string {
	for name, known := range elfArchs {
		if a == known {
			return name
		}
	}
	return fmt.Sprintf("%s (%s, %s)", a.machine, a.class, a.data)
}

// elfArchs maps GOARCH values to the ELF header a binary for them carries.
var elfArchs = map[string]elfArch{
	"386":               {elf.ELFCLASS32, elf.ELFDATA2LSB, elf.EM_386},
	utils.PlatformAMD64: {elf.ELFCLASS64, elf.ELFDATA2LSB, elf.EM_X86_64},
	"arm":               {elf.ELFCLASS32, elf.ELFDATA2LSB, elf.EM_ARM},
	utils.PlatformARM64: {elf.ELFCLASS64, elf.ELFDATA2LSB, elf.EM_AARCH64},
	"ppc64le":           {elf.ELFCLASS64, elf.ELFDATA2LSB, elf.EM_PPC64},
	"riscv64":           {elf.ELFCLASS64, elf.ELFDATA2LSB, elf.EM_RISCV},
	"s390x":             {elf.ELFCLASS64, elf.ELFDATA2MSB, elf.EM_S390},
}

const regularFile = "regular file"

// tarEntryType describes a tar entry by its type flag; FileInfo().Mode()
// reports hard links as regular files.
func tarEntryType(hdr *tar.Header) string {
	switch hdr.Typeflag {
	case tar.TypeReg:
		return regularFile
	case tar.TypeLink:
		return "hard link"
	default:
		return fileType(hdr.FileInfo().Mode())
	}
}

func fileType(mode fs.FileMode) string {
	switch {
	case mode&fs.ModeSymlink != 0:
		return "symlink"
	case mode.IsDir():
		return "directory"
	case mode&fs.ModeDevice != 0:
		return "device"
	case mode&fs.ModeNamedPipe != 0:
		return "named pipe"
	case mode&fs.ModeSocket != 0:
		return "socket"
	case mode.IsRegular():
		return regularFile
	default:
		return "special file"
	}
}

// limitedReader returns err, rather than truncating, once remaining runs out.
type limitedReader struct {
	r         io.Reader
	remaining int64
	err       error
}

func (l *limitedReader) Read(p []byte) (int, error) {
	if l.remaining <= 0 {
		return 0, l.err
	}
	if int64(len(p)) > l.remaining {
		p = p[:l.remaining]
	}
	n, err := l.r.Read(p)
	l.remaining -= int64(n)
	return n, err
}

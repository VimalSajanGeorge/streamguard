import os, json, requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
tokens = [t.strip() for t in os.environ.get("GITHUB_TOKEN", "").split(",") if t.strip()]
token = tokens[0]

GRAPHQL_ENDPOINT = "https://api.github.com/graphql"
VALID_CWE = {"CWE-89", "CWE-78", "CWE-79", "CWE-119", "CWE-120", "CWE-121", "CWE-122", "CWE-125", "CWE-134", "CWE-190", "CWE-416", "CWE-476"}

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Step 1: introspect the SecurityAdvisoryEcosystem enum to see valid values
INTROSPECT_QUERY = """
query {
  __type(name: "SecurityAdvisoryEcosystem") {
    enumValues { name description }
  }
  rateLimit { remaining }
}
"""

print("=== Step 1: Introspect SecurityAdvisoryEcosystem enum ===")
resp = requests.post(GRAPHQL_ENDPOINT, headers=headers,
                     json={"query": INTROSPECT_QUERY}, timeout=30)
data = resp.json()
enum_type = data.get("data", {}).get("__type", {})
print(f"Enum values for SecurityAdvisoryEcosystem:")
valid_ecosystems = []
for ev in enum_type.get("enumValues", []):
    print(f"  {ev['name']}: {ev.get('description','')}")
    valid_ecosystems.append(ev['name'])
remaining = data.get("data", {}).get("rateLimit", {}).get("remaining", "?")
print(f"Rate limit remaining: {remaining}")

# Step 2: scan ALL advisories (no ecosystem filter) via securityAdvisories
# to find entries with commit URLs and C/C++ signals
QUERY_ALL = """
query($first: Int!, $after: String) {
  securityAdvisories(first: $first, after: $after, orderBy: {field: PUBLISHED_AT, direction: DESC}) {
    nodes {
      ghsaId
      summary
      description
      vulnerabilities(first: 10) {
        nodes {
          package { name ecosystem }
        }
      }
      references { url }
      cwes(first: 5) { nodes { cweId } }
      cvss { score }
    }
    pageInfo { hasNextPage endCursor }
    totalCount
  }
  rateLimit { remaining resetAt }
}
"""

print("\n=== Step 2: Scan ALL advisories for C-language signals ===")

# C/C++ keywords to detect in package names or summaries
C_KEYWORDS = {
    # known C project names (existing list)
    "linux", "kernel", "ffmpeg", "imagemagick", "libpng", "libjpeg", "openssl",
    "curl", "libcurl", "php", "nginx", "apache", "sqlite", "tcpdump", "wireshark",
    "vlc", "libvlc", "glibc", "libc", "zlib", "openexr", "openjpeg", "libwebp",
    "libtiff", "libxml2", "libxslt", "libgd", "libssh", "libssh2", "libftp",
    "libpcap", "nmap", "git", "samba", "openssh", "bind", "named", "dhcp",
    "dovecot", "postfix", "exim", "proftpd", "vsftpd", "mutt", "wget",
    "python", "ruby", "perl", "php", "vim", "emacs", "screen",
    "gstreamer", "cairo", "pango", "gtk", "glib", "harfbuzz",
    "freetype", "fontconfig", "poppler", "ghostscript", "cups", "avahi",
    "dbus", "systemd", "sudo", "bash", "zsh", "cpio", "tar", "gzip",
    "bzip2", "xz", "lz4", "snappy", "lzo", "zstd",
    "gpsd", "ntp", "dhcpcd", "isc-dhcp", "strongswan", "openvpn",
    "tcpreplay", "iperf", "lftp", "irssi", "bitlbee", "znc",
    "mpg123", "sox", "lame", "faad", "flac", "speex", "opus", "vorbis",
    "x264", "x265", "aom", "dav1d", "vpx", "theora",
    "node", "v8", "webkitgtk", "webkit", "chromium", "firefox",
    "thunderbird", "libreoffice", "libressl",
    "openbsd", "freebsd", "netbsd", "freertos",
    "asterisk", "kamailio", "sofia-sip",
    "redis", "memcached", "nginx", "haproxy", "squid", "varnish",
    "postgresql", "mysql", "mariadb", "sqlite",
    "subversion", "mercurial", "cvs",
    "rpm", "dpkg", "yum", "apt",
    "qemu", "xen", "kvm", "virtualbox", "vmware",
    "xorg", "xserver", "libx11", "xlib",
    "racket", "scheme", "lua", "tcl",
    "gdb", "binutils", "gcc", "llvm", "clang",
    "wireshark", "tcpdump", "snort", "suricata",
    "libarchive", "p7zip", "unrar", "unzip",
}

cursor = None
total_scanned = 0
total_with_commit = 0
commit_with_valid_cwe = 0
c_project_matches = 0
c_project_with_commit = 0
c_project_commit_cwe = 0

ecosystem_counter = Counter()
cwe_dist_all_commit = Counter()
cwe_dist_c_commit = Counter()
c_pkg_commit_cwe_counter = Counter()

# For detailed sample logging
c_commit_cwe_samples = []

pages = 0
MAX_PAGES = 300  # up to 30K advisories

while pages < MAX_PAGES:
    variables = {"first": 100}
    if cursor:
        variables["after"] = cursor

    resp = requests.post(
        GRAPHQL_ENDPOINT,
        headers=headers,
        json={"query": QUERY_ALL, "variables": variables},
        timeout=60,
    )
    if resp.status_code != 200:
        print(f"  HTTP Error {resp.status_code}: {resp.text[:300]}")
        break

    data = resp.json()
    if "errors" in data:
        print(f"  GraphQL errors: {data['errors'][:2]}")
        break

    adv_data = data.get("data", {}).get("securityAdvisories", {})
    nodes = adv_data.get("nodes", [])
    page_info = adv_data.get("pageInfo", {})

    if pages == 0:
        print(f"  Total advisory count reported by API: {adv_data.get('totalCount', '?')}")

    if not nodes:
        print("  No nodes returned, stopping.")
        break

    for adv in nodes:
        total_scanned += 1
        ghsa_id = adv.get("ghsaId", "")
        summary = (adv.get("summary") or "").lower()
        description = (adv.get("description") or "").lower()

        # Collect ecosystems
        for vuln in adv.get("vulnerabilities", {}).get("nodes", []):
            pkg = vuln.get("package", {})
            eco = pkg.get("ecosystem", "")
            if eco:
                ecosystem_counter[eco] += 1

        # Check for GitHub commit URL
        has_commit = False
        commit_urls = []
        for ref in adv.get("references", []):
            url = ref.get("url", "")
            if "github.com" in url and "/commit/" in url:
                has_commit = True
                commit_urls.append(url)

        if has_commit:
            total_with_commit += 1

            # Collect CWEs
            cwes_here = []
            for cwe_node in adv.get("cwes", {}).get("nodes", []):
                cwe_id = cwe_node.get("cweId", "")
                cwe_dist_all_commit[cwe_id] += 1
                cwes_here.append(cwe_id)

            if any(c in VALID_CWE for c in cwes_here):
                commit_with_valid_cwe += 1

        # Check for C-language signal
        pkg_names = []
        for vuln in adv.get("vulnerabilities", {}).get("nodes", []):
            pkg = vuln.get("package", {})
            name = (pkg.get("name") or "").lower()
            if name:
                pkg_names.append(name)

        is_c_project = False
        matched_keyword = ""
        for kw in C_KEYWORDS:
            for name in pkg_names:
                if kw in name:
                    is_c_project = True
                    matched_keyword = kw
                    break
            if not is_c_project:
                # Also check summary/description
                if kw in summary or kw in description:
                    is_c_project = True
                    matched_keyword = kw
                    break
            if is_c_project:
                break

        if is_c_project:
            c_project_matches += 1
            if has_commit:
                c_project_with_commit += 1
                cwes_here = [
                    cn.get("cweId", "")
                    for cn in adv.get("cwes", {}).get("nodes", [])
                ]
                for cwe in cwes_here:
                    cwe_dist_c_commit[cwe] += 1
                if any(c in VALID_CWE for c in cwes_here):
                    c_project_commit_cwe += 1
                    for name in pkg_names:
                        c_pkg_commit_cwe_counter[name] += 1
                    if len(c_commit_cwe_samples) < 200:
                        c_commit_cwe_samples.append({
                            "ghsaId": ghsa_id,
                            "summary": adv.get("summary", "")[:120],
                            "cwes": cwes_here,
                            "pkg_names": pkg_names,
                            "matched_kw": matched_keyword,
                            "commit_urls": commit_urls[:3],
                        })

    cursor = page_info.get("endCursor")
    pages += 1

    if pages % 50 == 0:
        remaining = data.get("data", {}).get("rateLimit", {}).get("remaining", "?")
        print(f"  Page {pages:3d} | scanned: {total_scanned:6d} | with_commit: {total_with_commit:5d} | "
              f"commit+cwe: {commit_with_valid_cwe:4d} | c_proj: {c_project_matches:4d} | "
              f"c+commit: {c_project_with_commit:4d} | c+commit+cwe: {c_project_commit_cwe:4d} | "
              f"rate_remaining: {remaining}")

    if not page_info.get("hasNextPage", False):
        print(f"  Exhausted all pages at page {pages}.")
        break

print(f"\n{'='*60}")
print(f"=== FINAL RESULTS ===")
print(f"{'='*60}")
print(f"Total advisories scanned:          {total_scanned:7,d}")
print(f"With GitHub commit URL:            {total_with_commit:7,d}  ({100*total_with_commit/max(total_scanned,1):.1f}%)")
print(f"  ... AND valid CWE:               {commit_with_valid_cwe:7,d}  ({100*commit_with_valid_cwe/max(total_with_commit,1):.1f}% of commit)")
print(f"C-project keyword match:           {c_project_matches:7,d}  ({100*c_project_matches/max(total_scanned,1):.1f}%)")
print(f"  ... AND commit URL:              {c_project_with_commit:7,d}  ({100*c_project_with_commit/max(c_project_matches,1):.1f}% of C-proj)")
print(f"  ... AND valid CWE:               {c_project_commit_cwe:7,d}  ({100*c_project_commit_cwe/max(c_project_with_commit,1):.1f}% of C+commit)")

print(f"\n--- Ecosystem distribution (all advisories with vulns) ---")
for eco, cnt in ecosystem_counter.most_common(20):
    print(f"  {eco:<20s}: {cnt:6,d}")

print(f"\n--- CWE distribution in advisories WITH commit URL (top 25) ---")
for cwe, cnt in cwe_dist_all_commit.most_common(25):
    marker = " <-- TARGET" if cwe in VALID_CWE else ""
    print(f"  {cwe:<15s}: {cnt:5d}{marker}")

print(f"\n--- CWE distribution in C-project advisories WITH commit URL (top 25) ---")
for cwe, cnt in cwe_dist_c_commit.most_common(25):
    marker = " <-- TARGET" if cwe in VALID_CWE else ""
    print(f"  {cwe:<15s}: {cnt:5d}{marker}")

print(f"\n--- Package names in C+commit+validCWE entries (top 60) ---")
for name, cnt in c_pkg_commit_cwe_counter.most_common(60):
    print(f"  {name}: {cnt}")

print(f"\n--- Sample advisories: C+commit+validCWE (first 30) ---")
for s in c_commit_cwe_samples[:30]:
    print(f"  {s['ghsaId']} | {s['cwes']} | pkg={s['pkg_names'][:3]} | kw={s['matched_kw']}")
    print(f"    summary: {s['summary'][:100]}")
    print(f"    commits: {s['commit_urls'][:2]}")

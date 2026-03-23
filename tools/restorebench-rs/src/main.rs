//! Rust URPC client for gvisor restore benchmark.
//!
//! Talks directly to the sandbox's control socket — no shelling out for the
//! restore hot-path. Pre-creation and checkpoint still shell out to `runsc`.
//!
//! Dependencies: serde, serde_json, libc (no clap/nix/tempfile).
//!
//! Usage:
//!   cargo build --release --manifest-path tools/restorebench-rs/Cargo.toml
//!   sudo ./tools/restorebench-rs/target/release/restorebench-rs \
//!       --runsc ./bazel-out/k8-opt/bin/runsc/runsc_/runsc \
//!       --mem 1024 --precreate --background --iterations 5

use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::mem;
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CLI (manual parsing — no clap)
// ---------------------------------------------------------------------------

struct Cli {
    runsc: PathBuf,
    iterations: usize,
    mem: usize,
    touch_pct: usize,
    background: bool,
    precreate: bool,
    settle: f64,
    keep_tmp: bool,
    compression: String,
}

impl Cli {
    fn parse() -> Self {
        let args: Vec<String> = env::args().collect();
        let mut cli = Cli {
            runsc: PathBuf::new(),
            iterations: 5,
            mem: 0,
            touch_pct: 100,
            background: false,
            precreate: false,
            settle: 2.0,
            keep_tmp: false,
            compression: "none".into(),
        };
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--runsc" => { i += 1; cli.runsc = PathBuf::from(&args[i]); }
                "--iterations" => { i += 1; cli.iterations = args[i].parse().unwrap(); }
                "--mem" => { i += 1; cli.mem = args[i].parse().unwrap(); }
                "--touch-pct" => { i += 1; cli.touch_pct = args[i].parse().unwrap(); }
                "--settle" => { i += 1; cli.settle = args[i].parse().unwrap(); }
                "--compression" => { i += 1; cli.compression = args[i].clone(); }
                "--background" => cli.background = true,
                "--precreate" => cli.precreate = true,
                "--keep-tmp" => cli.keep_tmp = true,
                s if s.starts_with("--runsc=") => cli.runsc = PathBuf::from(&s[8..]),
                s if s.starts_with("--iterations=") => cli.iterations = s[13..].parse().unwrap(),
                s if s.starts_with("--mem=") => cli.mem = s[6..].parse().unwrap(),
                s if s.starts_with("--touch-pct=") => cli.touch_pct = s[12..].parse().unwrap(),
                s if s.starts_with("--settle=") => cli.settle = s[9..].parse().unwrap(),
                s if s.starts_with("--compression=") => cli.compression = s[14..].to_string(),
                other => {
                    eprintln!("unknown arg: {}", other);
                    eprintln!("usage: restorebench-rs --runsc PATH [--mem N] [--iterations N] [--precreate] [--background] [--keep-tmp]");
                    std::process::exit(1);
                }
            }
            i += 1;
        }
        if cli.runsc.as_os_str().is_empty() {
            eprintln!("error: --runsc is required");
            std::process::exit(1);
        }
        cli
    }
}

// ---------------------------------------------------------------------------
// URPC wire protocol
// ---------------------------------------------------------------------------

/// Client→server call envelope.
#[derive(Serialize)]
struct ClientCall<'a, T: Serialize> {
    method: &'a str,
    arg: &'a T,
}

/// Server→client result envelope.
#[derive(Deserialize)]
struct CallResult {
    success: bool,
    #[serde(default)]
    err: String,
}

/// Send a JSON-encoded URPC call, optionally with file descriptors via
/// SCM_RIGHTS. Returns an `io::Result`.
fn urpc_send<T: Serialize>(
    stream: &UnixStream,
    method: &str,
    arg: &T,
    fds: &[RawFd],
) -> io::Result<()> {
    let call = ClientCall { method, arg };
    let data = serde_json::to_vec(&call).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    if fds.is_empty() {
        // No FDs — plain write.
        let mut stream_ref: &UnixStream = stream;
        stream_ref.write_all(&data)?;
        return Ok(());
    }

    // sendmsg with SCM_RIGHTS.
    unsafe {
        let iov = libc::iovec {
            iov_base: data.as_ptr() as *mut libc::c_void,
            iov_len: data.len(),
        };

        let fds_bytes = fds.len() * mem::size_of::<RawFd>();
        let cmsg_space = libc::CMSG_SPACE(fds_bytes as libc::c_uint) as usize;
        let mut cmsg_buf = vec![0u8; cmsg_space];

        let mut msg: libc::msghdr = mem::zeroed();
        msg.msg_iov = &iov as *const _ as *mut _;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
        msg.msg_controllen = cmsg_space as _;

        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(fds_bytes as libc::c_uint) as _;

        std::ptr::copy_nonoverlapping(fds.as_ptr() as *const u8, libc::CMSG_DATA(cmsg), fds_bytes);

        let mut sent = 0usize;
        while sent < data.len() {
            let cur_iov = libc::iovec {
                iov_base: data[sent..].as_ptr() as *mut libc::c_void,
                iov_len: data.len() - sent,
            };
            let mut m: libc::msghdr = mem::zeroed();
            m.msg_iov = &cur_iov as *const _ as *mut _;
            m.msg_iovlen = 1;
            if sent == 0 {
                // Attach FDs only on the first sendmsg.
                m.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
                m.msg_controllen = cmsg_space as _;
            }
            let n = libc::sendmsg(stream.as_raw_fd(), &m, 0);
            if n < 0 {
                return Err(io::Error::last_os_error());
            }
            sent += n as usize;
        }
    }

    Ok(())
}

/// Receive a URPC response. Returns the parsed `CallResult`.
/// We don't expect any FDs back for our RPCs.
fn urpc_recv(stream: &UnixStream) -> io::Result<CallResult> {
    // Read response bytes. The response is a single JSON object.
    // Typical responses are <1KB. Use a 64KB buffer and read in a loop
    // until serde can parse a complete object.
    let mut buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 8192];
    loop {
        let mut stream_ref: &UnixStream = stream;
        let n = stream_ref.read(&mut tmp)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "urpc: connection closed before response",
            ));
        }
        buf.extend_from_slice(&tmp[..n]);

        match serde_json::from_slice::<CallResult>(&buf) {
            Ok(result) => return Ok(result),
            Err(e) if e.is_eof() => continue, // Need more data.
            Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e)),
        }
    }
}

/// Make a complete URPC call: send request (with optional FDs), receive
/// response, check for errors.
fn urpc_call<T: Serialize>(
    stream: &UnixStream,
    method: &str,
    arg: &T,
    fds: &[RawFd],
) -> io::Result<()> {
    urpc_send(stream, method, arg, fds)?;
    let result = urpc_recv(stream)?;
    if !result.success {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("urpc {}: {}", method, result.err),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// URPC argument types (matching gVisor's JSON wire format)
// ---------------------------------------------------------------------------

/// Argument for containerManager.Restore.
#[derive(Serialize)]
struct RestoreOpts {
    #[serde(rename = "HavePagesFile")]
    have_pages_file: bool,
    #[serde(rename = "HaveDeviceFile")]
    have_device_file: bool,
    #[serde(rename = "Background")]
    background: bool,
}

/// Argument for Network.CreateLinksAndRoutes (loopback-only case).
#[derive(Serialize)]
struct CreateLinksAndRoutesArgs {
    #[serde(rename = "LoopbackLinks")]
    loopback_links: Vec<LoopbackLink>,
}

#[derive(Serialize)]
struct LoopbackLink {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Addresses")]
    addresses: Vec<IpWithPrefix>,
    #[serde(rename = "Routes")]
    routes: Vec<Route>,
    #[serde(rename = "GVisorGRO")]
    gvisor_gro: bool,
    #[serde(rename = "GVisorGSOMaxSize")]
    #[serde(skip_serializing_if = "Option::is_none")]
    gvisor_gso_max_size: Option<u32>,
}

#[derive(Serialize)]
struct IpWithPrefix {
    #[serde(rename = "Address")]
    address: String,
    #[serde(rename = "PrefixLen")]
    prefix_len: i32,
}

/// Go's net.IPNet serializes as {"IP":"...", "Mask":"<base64>"}.
#[derive(Serialize)]
struct IpNet {
    #[serde(rename = "IP")]
    ip: String,
    #[serde(rename = "Mask")]
    mask: String,
}

#[derive(Serialize)]
struct Route {
    #[serde(rename = "Destination")]
    destination: IpNet,
    /// Go's net.IP zero-value serialises as "".
    #[serde(rename = "Gateway")]
    gateway: String,
    #[serde(rename = "MTU")]
    mtu: u32,
}

fn default_loopback_link() -> LoopbackLink {
    use std::io::Write as _;
    // base64 of 0xff000000 (IPv4 /8 mask)
    let mask_v4 = base64_encode(&[0xff, 0x00, 0x00, 0x00]);
    // base64 of 16x 0xff (IPv6 /128 mask)
    let mask_v6 = base64_encode(&[0xff; 16]);

    LoopbackLink {
        name: "lo".into(),
        addresses: vec![
            IpWithPrefix { address: "127.0.0.1".into(), prefix_len: 8 },
            IpWithPrefix { address: "::1".into(), prefix_len: 128 },
        ],
        routes: vec![
            Route {
                destination: IpNet { ip: "127.0.0.0".into(), mask: mask_v4 },
                gateway: String::new(),
                mtu: 0,
            },
            Route {
                destination: IpNet { ip: "::1".into(), mask: mask_v6 },
                gateway: String::new(),
                mtu: 0,
            },
        ],
        gvisor_gro: false,
        gvisor_gso_max_size: None,
    }
}

/// Minimal base64 encoder (standard alphabet, with padding).
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((triple >> 18) & 0x3F) as usize]);
        out.push(ALPHABET[((triple >> 12) & 0x3F) as usize]);
        if chunk.len() > 1 {
            out.push(ALPHABET[((triple >> 6) & 0x3F) as usize]);
        } else {
            out.push(b'=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(triple & 0x3F) as usize]);
        } else {
            out.push(b'=');
        }
    }
    String::from_utf8(out).unwrap()
}

// ---------------------------------------------------------------------------
// Container state file — extract control socket path
// ---------------------------------------------------------------------------

/// Find the sandbox control socket path.
/// gvisor stores it at `<rootDir>/runsc-<sandboxID>.sock`.
/// For root containers, sandboxID == containerID.
fn find_control_socket(root_dir: &Path, container_id: &str) -> io::Result<PathBuf> {
    // Direct path: runsc-<id>.sock
    let sock = root_dir.join(format!("runsc-{}.sock", container_id));
    if sock.exists() {
        return Ok(sock);
    }
    // Fallback: scan for any .sock file.
    if let Ok(entries) = fs::read_dir(root_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().ends_with(".sock") {
                return Ok(entry.path());
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("no control socket found for {} in {}", container_id, root_dir.display()),
    ))
}

// ---------------------------------------------------------------------------
// Shell helpers (used for non-hot-path operations)
// ---------------------------------------------------------------------------

fn run_runsc(runsc: &Path, root_dir: &Path, args: &[&str]) -> io::Result<()> {
    let status = Command::new(runsc)
        .arg("--rootless=false")
        .arg(format!("--root={}", root_dir.display()))
        .arg("--platform=systrap")
        .arg("--network=none")
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::inherit())
        .status()?;
    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("runsc {:?} failed: {}", args, status),
        ));
    }
    Ok(())
}

fn human_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.2} MiB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.2} KiB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{} B", b)
    }
}

// ---------------------------------------------------------------------------
// Alloc helper
// ---------------------------------------------------------------------------

const ALLOC_SRC: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char **argv) {
    if (argc < 2) { pause(); return 0; }
    long mib = atol(argv[1]);
    if (mib <= 0) { pause(); return 0; }
    int touch_pct = (argc >= 3) ? atoi(argv[2]) : 100;
    if (touch_pct <= 0) touch_pct = 1;
    if (touch_pct > 100) touch_pct = 100;
    long bytes = mib * 1024L * 1024L;
    char *p = malloc(bytes);
    if (!p) { perror("malloc"); return 1; }
    long pgsz = sysconf(_SC_PAGESIZE);
    long total = bytes / pgsz;
    long step = (touch_pct < 100) ? (100 / touch_pct) : 1;
    if (step < 1) step = 1;
    long touched = 0;
    for (long pg = 0; pg < total; pg += step) {
        p[pg * pgsz] = (char)(pg >> 4);
        touched++;
    }
    fprintf(stderr, "allocated %ld MiB, touched %ld/%ld pages (%d%%)\n",
            mib, touched, total, touch_pct);
    for (;;) pause();
    return 0;
}
"#;

fn build_alloc(dir: &Path) -> io::Result<PathBuf> {
    let src = dir.join("alloc.c");
    let bin = dir.join("alloc");
    fs::write(&src, ALLOC_SRC)?;
    let status = Command::new("cc")
        .args(["-static", "-O2", "-o"])
        .arg(&bin)
        .arg(&src)
        .status()?;
    if !status.success() {
        return Err(io::Error::new(io::ErrorKind::Other, "cc failed"));
    }
    Ok(bin)
}

// ---------------------------------------------------------------------------
// OCI spec
// ---------------------------------------------------------------------------

fn write_oci_spec(
    bundle_dir: &Path,
    args: &[String],
    bind_src: Option<&Path>,
) -> io::Result<()> {
    let mut mounts = serde_json::json!([
        {"destination": "/proc", "type": "proc", "source": "proc"},
        {"destination": "/tmp",  "type": "tmpfs", "source": "tmpfs"}
    ]);
    if let Some(src) = bind_src {
        mounts.as_array_mut().unwrap().push(serde_json::json!({
            "destination": "/workload",
            "type": "bind",
            "source": src.to_str().unwrap(),
            "options": ["rbind", "ro"]
        }));
    }
    let spec = serde_json::json!({
        "ociVersion": "1.0.0",
        "process": {
            "terminal": false,
            "user": {"uid": 0, "gid": 0},
            "args": args,
            "env": ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],
            "cwd": "/",
            "capabilities": {
                "bounding": ["CAP_AUDIT_WRITE","CAP_CHOWN","CAP_DAC_OVERRIDE",
                    "CAP_FOWNER","CAP_FSETID","CAP_KILL","CAP_MKNOD",
                    "CAP_NET_BIND_SERVICE","CAP_NET_RAW","CAP_SETFCAP",
                    "CAP_SETGID","CAP_SETPCAP","CAP_SETUID","CAP_SYS_CHROOT"],
                "effective": ["CAP_AUDIT_WRITE","CAP_CHOWN","CAP_DAC_OVERRIDE",
                    "CAP_FOWNER","CAP_FSETID","CAP_KILL","CAP_MKNOD",
                    "CAP_NET_BIND_SERVICE","CAP_NET_RAW","CAP_SETFCAP",
                    "CAP_SETGID","CAP_SETPCAP","CAP_SETUID","CAP_SYS_CHROOT"],
                "permitted": ["CAP_AUDIT_WRITE","CAP_CHOWN","CAP_DAC_OVERRIDE",
                    "CAP_FOWNER","CAP_FSETID","CAP_KILL","CAP_MKNOD",
                    "CAP_NET_BIND_SERVICE","CAP_NET_RAW","CAP_SETFCAP",
                    "CAP_SETGID","CAP_SETPCAP","CAP_SETUID","CAP_SYS_CHROOT"]
            }
        },
        "root": {"path": "/", "readonly": true},
        "hostname": "restorebench",
        "mounts": mounts,
        "annotations": {},
        "linux": {
            "namespaces": [
                {"type": "pid"}, {"type": "ipc"}, {"type": "uts"}, {"type": "mount"}
            ]
        }
    });
    fs::write(bundle_dir.join("config.json"), serde_json::to_vec_pretty(&spec).unwrap())
}

// ---------------------------------------------------------------------------
// Direct restore via URPC
// ---------------------------------------------------------------------------

/// Perform a restore by talking directly to the sandbox's URPC socket.
/// This is the hot-path — no process spawning.
/// Timing breakdown from the last restore_via_urpc call.
struct RestoreTiming {
    connect: Duration,
    network: Duration,
    restore: Duration,
    total: Duration,
}

fn restore_via_urpc(
    root_dir: &Path,
    container_id: &str,
    checkpoint_dir: &Path,
    background: bool,
) -> io::Result<RestoreTiming> {
    let t_start = Instant::now();
    // 1. Find the control socket.
    let sock_path = find_control_socket(root_dir, container_id)?;
    let sock_str = sock_path.to_string_lossy();

    // Handle long socket paths (>= UNIX_PATH_MAX = 108).
    let t_after_find = Instant::now();
    let conn = if sock_str.len() >= 108 {
        let c_path = std::ffi::CString::new(sock_str.as_ref()).unwrap();
        let sock_fd = unsafe { libc::open(c_path.as_ptr(), libc::O_PATH) };
        if sock_fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let proc_path = format!("/proc/self/fd/{}", sock_fd);
        let stream = UnixStream::connect(&proc_path)?;
        unsafe { libc::close(sock_fd); }
        stream
    } else {
        UnixStream::connect(&*sock_path)?
    };

    let t_connected = Instant::now();

    // 2. Network setup (loopback only — matches --network=none).
    urpc_call(
        &conn,
        "Network.CreateLinksAndRoutes",
        &CreateLinksAndRoutesArgs {
            loopback_links: vec![default_loopback_link()],
        },
        &[],
    )?;

    let t_network_done = Instant::now();

    // 3. Open checkpoint files.
    let state_file = fs::File::open(checkpoint_dir.join("checkpoint.img"))?;
    let mut fds: Vec<RawFd> = vec![state_file.as_raw_fd()];

    let pages_meta_file = fs::File::open(checkpoint_dir.join("pages_meta.img")).ok();
    let pages_file = fs::File::open(checkpoint_dir.join("pages.img")).ok();
    let have_pages = pages_meta_file.is_some() && pages_file.is_some();
    if have_pages {
        fds.push(pages_meta_file.as_ref().unwrap().as_raw_fd());
        fds.push(pages_file.as_ref().unwrap().as_raw_fd());
    }

    // 4. Send Restore RPC with checkpoint FDs.
    urpc_call(
        &conn,
        "containerManager.Restore",
        &RestoreOpts {
            have_pages_file: have_pages,
            have_device_file: false,
            background,
        },
        &fds,
    )?;

    let t_restore_done = Instant::now();

    // Files are closed when dropped.
    drop(state_file);
    drop(pages_meta_file);
    drop(pages_file);
    drop(conn);

    Ok(RestoreTiming {
        connect: t_connected.duration_since(t_start),
        network: t_network_done.duration_since(t_connected),
        restore: t_restore_done.duration_since(t_network_done),
        total: t_restore_done.duration_since(t_start),
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    if unsafe { libc::getuid() } != 0 {
        eprintln!("error: must be run as root");
        std::process::exit(1);
    }

    let runsc = fs::canonicalize(&cli.runsc).expect("resolving runsc path");
    eprintln!("runsc: {}", runsc.display());

    // ---- Temp dir ----
    let tmp: PathBuf = {
        let mut template = b"/tmp/restorebench-rs-XXXXXX\0".to_vec();
        let ptr = unsafe { libc::mkdtemp(template.as_mut_ptr() as *mut libc::c_char) };
        if ptr.is_null() {
            panic!("mkdtemp failed: {}", io::Error::last_os_error());
        }
        let c_str = unsafe { std::ffi::CStr::from_ptr(ptr) };
        PathBuf::from(c_str.to_str().unwrap())
    };
    if cli.keep_tmp {
        eprintln!("temp dir: {}", tmp.display());
    }
    struct TmpGuard { path: PathBuf, keep: bool }
    impl Drop for TmpGuard {
        fn drop(&mut self) {
            if !self.keep { let _ = fs::remove_dir_all(&self.path); }
        }
    }
    let _tmp_guard = TmpGuard { path: tmp.clone(), keep: cli.keep_tmp };

    // ---- Build workload ----
    let (container_args, bind_src): (Vec<String>, Option<PathBuf>) = if cli.mem > 0 {
        let alloc_bin = build_alloc(&tmp).expect("building alloc helper");
        let wl_dir = tmp.join("workload");
        fs::create_dir_all(&wl_dir).unwrap();
        fs::copy(&alloc_bin, wl_dir.join("alloc")).unwrap();
        eprintln!("workload: allocate {} MiB, touch {}%", cli.mem, cli.touch_pct);
        (
            vec![
                "/workload/alloc".into(),
                cli.mem.to_string(),
                cli.touch_pct.to_string(),
            ],
            Some(wl_dir),
        )
    } else {
        eprintln!("workload: sleep (no memory allocation)");
        (vec!["sleep".into(), "infinity".into()], None)
    };

    // ---- Source container: create + start + checkpoint ----
    let src_root = tmp.join("root-src");
    let src_bundle = tmp.join("bundle-src");
    let ckpt_dir = tmp.join("checkpoint");
    fs::create_dir_all(&src_root).unwrap();
    fs::create_dir_all(&src_bundle).unwrap();
    fs::create_dir_all(&ckpt_dir).unwrap();

    write_oci_spec(&src_bundle, &container_args, bind_src.as_deref()).unwrap();

    let src_id = format!("src-{}", std::process::id());

    eprintln!("=== Creating source container ===");
    run_runsc(
        &runsc,
        &src_root,
        &["create", &format!("--bundle={}", src_bundle.display()), &src_id],
    )
    .expect("runsc create (source)");

    run_runsc(&runsc, &src_root, &["start", &src_id]).expect("runsc start (source)");

    let settle = if cli.mem >= 512 {
        cli.settle.max(2.0 + (cli.mem as f64 / 512.0))
    } else if cli.mem > 0 {
        cli.settle.max(2.0)
    } else {
        cli.settle
    };
    eprintln!("waiting {:.1}s for workload to settle...", settle);
    std::thread::sleep(Duration::from_secs_f64(settle));

    eprintln!("=== Checkpointing ===");
    let ckpt_start = Instant::now();
    run_runsc(
        &runsc,
        &src_root,
        &[
            "checkpoint",
            &format!("--image-path={}", ckpt_dir.display()),
            &format!("--compression={}", cli.compression),
            &src_id,
        ],
    )
    .expect("runsc checkpoint");
    eprintln!("checkpoint completed in {:?}", ckpt_start.elapsed());

    // Print checkpoint sizes.
    let mut total_bytes = 0u64;
    if let Ok(entries) = fs::read_dir(&ckpt_dir) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                let sz = meta.len();
                total_bytes += sz;
                eprintln!("  {}: {}", entry.file_name().to_string_lossy(), human_bytes(sz));
            }
        }
    }
    eprintln!("  total: {}", human_bytes(total_bytes));

    // ---- Measure restore ----
    let mode = match (cli.precreate, cli.background) {
        (true, true) => "precreate+background (URPC direct)",
        (true, false) => "precreate (URPC direct)",
        (false, true) => "background (URPC direct)",
        (false, false) => "sync (URPC direct)",
    };
    eprintln!(
        "=== Measuring restore ({}, {} iterations) ===",
        mode, cli.iterations
    );

    let mut durations = Vec::with_capacity(cli.iterations);
    let bundle_for_restore = tmp.join("bundle-restore");
    fs::create_dir_all(&bundle_for_restore).unwrap();
    write_oci_spec(&bundle_for_restore, &container_args, bind_src.as_deref()).unwrap();

    for i in 0..cli.iterations {
        let restore_id = format!("dst-{}-{}", i, std::process::id());
        let restore_root = tmp.join(format!("root-restore-{}", i));
        fs::create_dir_all(&restore_root).unwrap();

        // Pre-create sandbox (shell out — not timed).
        // This is always needed: creates sandbox + gofer.
        if cli.precreate {
            run_runsc(
                &runsc,
                &restore_root,
                &[
                    "create",
                    &format!("--bundle={}", bundle_for_restore.display()),
                    &restore_id,
                ],
            )
            .expect("runsc create (precreate)");
        }

        // ---- TIMED SECTION ----
        let start = Instant::now();

        if !cli.precreate {
            // Create sandbox inline (timed).
            run_runsc(
                &runsc,
                &restore_root,
                &[
                    "create",
                    &format!("--bundle={}", bundle_for_restore.display()),
                    &restore_id,
                ],
            )
            .expect("runsc create (inline)");
        }

        // Direct URPC restore — no process spawn.
        let timing = restore_via_urpc(&restore_root, &restore_id, &ckpt_dir, cli.background)
            .expect("restore via URPC");

        let elapsed = start.elapsed();
        // ---- END TIMED SECTION ----

        durations.push(elapsed);
        eprintln!("  restore {}/{}: {:?}  (connect {:?}, network {:?}, restore {:?})",
            i + 1, cli.iterations, elapsed,
            timing.connect, timing.network, timing.restore);

        // Cleanup.
        let _ = run_runsc(&runsc, &restore_root, &["kill", &restore_id, "SIGKILL"]);
        std::thread::sleep(Duration::from_millis(200));
        let _ = run_runsc(&runsc, &restore_root, &["delete", "-force", &restore_id]);
    }

    // Cleanup source.
    let _ = run_runsc(&runsc, &src_root, &["kill", &src_id, "SIGKILL"]);
    std::thread::sleep(Duration::from_millis(200));
    let _ = run_runsc(&runsc, &src_root, &["delete", "-force", &src_id]);

    // ---- Summary ----
    println!();
    println!("=== Restore Benchmark Results ===");
    println!("Platform:       systrap");
    println!("Compression:    {}", cli.compression);
    println!("Memory:         {} MiB", cli.mem);
    println!("Checkpoint:     {}", human_bytes(total_bytes));
    println!("Iterations:     {}", cli.iterations);
    println!("Mode:           {}", mode);
    println!();

    let min = durations.iter().min().copied().unwrap_or_default();
    let max = durations.iter().max().copied().unwrap_or_default();
    let total: Duration = durations.iter().sum();
    let avg = total / durations.len() as u32;

    println!("Min:     {:?}", min);
    println!("Max:     {:?}", max);
    println!("Avg:     {:?}", avg);
    println!("Total:   {:?}", total);
    for (i, d) in durations.iter().enumerate() {
        println!("  [{}] {:?}", i, d);
    }
}

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};

static CLOSING: AtomicBool = AtomicBool::new(false);

const HIDDEN: u32 = 0x08000000;

struct Started(Mutex<Vec<Child>>);

fn listening(port: u16) -> bool {
  let address: SocketAddr = ([127, 0, 0, 1], port).into();
  TcpStream::connect_timeout(&address, Duration::from_millis(250)).is_ok()
}

fn sneo_ui() -> bool {
  let address: SocketAddr = ([127, 0, 0, 1], 3001).into();
  let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(250)) else {
    return false;
  };
  let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
  if stream.write_all(b"GET /api/sneo HTTP/1.1\r\nHost: 127.0.0.1:3001\r\nConnection: close\r\n\r\n").is_err() {
    return false;
  }
  let mut reply = Vec::new();
  let _ = stream.read_to_end(&mut reply);
  String::from_utf8_lossy(&reply).contains("\"sneo\":true")
}

fn engine_root() -> PathBuf {
  let mut here = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
  while let Some(dir) = here {
    if dir.join("master.exe").exists() && dir.join("ui").exists() {
      return dir;
    }
    here = dir.parent().map(Path::to_path_buf);
  }
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn stop_engine() {
  let address: SocketAddr = ([127, 0, 0, 1], 3001).into();
  let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(500)) else {
    return;
  };
  let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
  let request = "POST /api/engine/shutdown HTTP/1.1\r\nHost: 127.0.0.1:3001\r\nX-Sneo: shutdown\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
  if stream.write_all(request.as_bytes()).is_err() {
    return;
  }
  let mut reply = Vec::new();
  let _ = stream.read_to_end(&mut reply);
}

fn launch(program: &Path, args: &[&str], dir: &Path, log: &Path) -> Option<Child> {
  let out = std::fs::File::create(log).ok()?;
  let err = out.try_clone().ok()?;
  Command::new(program).args(args).current_dir(dir).stdout(out).stderr(err).creation_flags(HIDDEN).spawn().ok()
}

pub fn run() {
  let root = engine_root();
  let ui = root.join("ui");
  let mut children = Vec::new();

  if !listening(8000) {
    if let Some(child) = launch(&root.join("master.exe"), &["8000", "-", "-", "modelos"], &root, &root.join("master.log")) {
      children.push(child);
    }
  }

  if !sneo_ui() {
    if let Some(child) = launch(Path::new("node"), &["server/server.js"], &ui, &ui.join("server.log")) {
      children.push(child);
    }
  }

  let app = tauri::Builder::default()
    .manage(Started(Mutex::new(children)))
    .setup(|app| {
      let handle = app.handle().clone();
      thread::spawn(move || {
        let start = Instant::now();
        while !sneo_ui() && start.elapsed() < Duration::from_secs(30) {
          thread::sleep(Duration::from_millis(200));
        }
        let url = WebviewUrl::External("http://127.0.0.1:3001".parse().unwrap());
        let _ = WebviewWindowBuilder::new(&handle, "main", url).title("Sneo").inner_size(1280.0, 840.0).build();

        let ui = engine_root().join("ui");
        let mut misses = 0;
        while !CLOSING.load(Ordering::SeqCst) {
          thread::sleep(Duration::from_secs(3));
          misses = if sneo_ui() { 0 } else { misses + 1 };
          if misses < 3 || CLOSING.load(Ordering::SeqCst) {
            continue;
          }
          misses = 0;
          if let Some(child) = launch(Path::new("node"), &["server/server.js"], &ui, &ui.join("server.log")) {
            if let Some(started) = handle.try_state::<Started>() {
              started.0.lock().unwrap().push(child);
            }
          }
        }
      });
      Ok(())
    })
    .build(tauri::generate_context!())
    .expect("could not start Sneo");

  app.run(|handle, event| {
    match event {
      RunEvent::WindowEvent { label, event: WindowEvent::CloseRequested { api, .. }, .. } => {
        api.prevent_close();
        if let Some(window) = handle.get_webview_window(&label) {
          let _ = window.hide();
        }
        if CLOSING.swap(true, Ordering::SeqCst) {
          return;
        }
        let handle = handle.clone();
        thread::spawn(move || {
          stop_engine();
          handle.exit(0);
        });
      }
      RunEvent::Exit => {
        if let Some(started) = handle.try_state::<Started>() {
          for child in started.0.lock().unwrap().iter_mut() {
            let _ = child.kill();
          }
        }
      }
      _ => {}
    }
  });
}

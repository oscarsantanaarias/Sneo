import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';

export const onWindows = process.platform === 'win32';

// el motor y el worker en windows terminan en .exe y en linux no
export const programPath = (dir, name) => path.join(dir, onWindows ? `${name}.exe` : name);

// cierra solo el programa de esa ruta, nunca por nombre: otro programa del usuario puede llamarse igual
export const stopProgram = (file) => new Promise(resolve => {
  if(onWindows){
    const quoted = file.replace(/'/g, "''");
    const name = path.basename(file, '.exe').replace(/[^\w.-]/g, '');
    execFile('powershell.exe', ['-NoProfile', '-Command', `Get-Process ${name} -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq '${quoted}' } | Stop-Process -Force`], {windowsHide: true}, () => resolve());
    return;
  }

  // en linux cada proceso dice en /proc/<pid>/exe de que archivo salio, y si se recompilo termina en (deleted)
  let ids = [];
  try {
    ids = fs.readdirSync('/proc').filter(name => /^\d+$/.test(name));
  } catch(e) {
    return resolve();
  }

  for(const id of ids){
    let exe = '';
    try {
      exe = fs.readlinkSync(`/proc/${id}/exe`).replace(/ \(deleted\)$/, '');
    } catch(e) {
      continue;
    }
    if(exe !== file) continue;
    try {
      process.kill(Number(id), 'SIGKILL');
    } catch(e) {}
  }
  resolve();
});

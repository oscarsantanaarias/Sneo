const { spawn } = require('child_process');
const fs = require('fs');

const child = process.platform === 'win32'
  ? spawn('ping', ['-n', '120', '127.0.0.1'], {stdio: 'ignore', windowsHide: true})
  : spawn('sleep', ['120'], {stdio: 'ignore'});

fs.writeFileSync(process.argv[2], String(child.pid));
setInterval(() => {}, 1000);

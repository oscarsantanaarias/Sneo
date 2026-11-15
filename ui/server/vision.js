import fs from 'fs/promises';
import path from 'path';
import { pathToFileURL } from 'url';
import { getProjectPath, safeJoin } from './projects.js';
import { askVision } from './model.js';
import { cleanText } from './storage.js';

let puppeteer = null;

const browserPaths = [
  path.join(process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)', 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
  path.join(process.env.ProgramFiles || 'C:\\Program Files', 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
  path.join(process.env.ProgramFiles || 'C:\\Program Files', 'Google', 'Chrome', 'Application', 'chrome.exe'),
  path.join(process.env.LOCALAPPDATA || '', 'Google', 'Chrome', 'Application', 'chrome.exe')
];

const systemBrowser = async () => {
  for(const candidate of browserPaths){
    const found = await fs.access(candidate).then(() => true).catch(() => false);
    if(found) return candidate;
  }

  return undefined;
};

const loadPuppeteer = async () => {
  if(!puppeteer){
    puppeteer = (await import('puppeteer')).default;
  }

  return puppeteer;
};

const resolveTarget = async (project, file, url) => {
  if(url && /^https?:\/\//i.test(url)){
    return url;
  }

  const projectPath = await getProjectPath(project);
  const filePath = safeJoin(projectPath, file || 'index.html');
  await fs.access(filePath);
  return pathToFileURL(filePath).href;
};

export const toolVisualCheck = async ({project, file, url, question}) => {
  const target = await resolveTarget(project, file, url);
  const pptr = await loadPuppeteer();
  const browser = await pptr.launch({headless: 'new', executablePath: await systemBrowser()});

  try {
    const page = await browser.newPage();
    await page.setViewport({width: 1280, height: 800});

    const consoleErrors = [];
    page.on('console', msg => msg.type() === 'error' && consoleErrors.push(msg.text()));
    page.on('pageerror', err => consoleErrors.push(String(err.message || err)));

    await page.goto(target, {waitUntil: 'networkidle2', timeout: 30000}).catch(() => {});
    await new Promise(resolve => setTimeout(resolve, 600));

    const layout = await page.evaluate(() => {
      const vw = window.innerWidth;
      const problems = [];
      const scrollX = document.documentElement.scrollWidth > window.innerWidth + 2;

      for(const el of document.body.querySelectorAll('*')){
        const r = el.getBoundingClientRect();

        if(r.width === 0 || r.height === 0){
          continue;
        }

        if(r.right > vw + 4 || r.left < -4){
          problems.push(`${el.tagName.toLowerCase()}${el.id ? '#' + el.id : ''} se sale horizontal (left ${Math.round(r.left)}, right ${Math.round(r.right)}, viewport ${vw})`);
        }

        if(problems.length >= 12){
          break;
        }
      }

      return {overflowX: scrollX, problems, title: document.title, bodyText: document.body.innerText.slice(0, 400)};
    });

    const shotRel = `screenshot_${(file || 'page').replace(/[^a-z0-9]/gi, '_')}.png`;
    const projectPath = await getProjectPath(project);
    const shotPath = url ? path.join(projectPath, shotRel) : safeJoin(projectPath, shotRel);
    await page.screenshot({path: shotPath, fullPage: false});
    const imageBase64 = (await fs.readFile(shotPath)).toString('base64');

    let visionFeedback = '';

    try {
      const q = cleanText(question) || 'Eres disenador UI experto. Mira esta captura de una web. Describe brevemente que se ve y senala problemas concretos de diseno o layout (elementos cortados, superpuestos, ilegibles, feos, mal alineados). Se especifico y directo.';
      visionFeedback = await askVision(q, imageBase64);
    } catch(e) {
      visionFeedback = `(modelo de vision no disponible: ${e.message})`;
    }

    return {
      target,
      screenshot: shotRel,
      title: layout.title,
      consoleErrors: consoleErrors.slice(0, 10),
      overflowHorizontal: layout.overflowX,
      layoutProblems: layout.problems,
      visionFeedback
    };
  } finally {
    await browser.close();
  }
};

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const FPS = 24;
const WIDTH = 1280;
const HEIGHT = 720;
const TOTAL_DURATION = 160;
const CHUNK_SEC = 5;
const FRAMES_DIR = path.join(__dirname, 'frames');

if (!fs.existsSync(FRAMES_DIR)) fs.mkdirSync(FRAMES_DIR, { recursive: true });

const HTML_PATH = path.resolve(__dirname, 'scenes.html');
const CHROME = '/root/.cache/puppeteer/chrome/linux-146.0.7680.153/chrome-linux64/chrome';

async function renderChunk(startSec, endSec) {
  const startFrame = Math.floor(startSec * FPS);
  const endFrame = Math.floor(endSec * FPS);
  
  let allExist = true;
  for (let f = startFrame; f < endFrame; f++) {
    if (!fs.existsSync(path.join(FRAMES_DIR, `frame_${String(f).padStart(6, '0')}.png`))) {
      allExist = false; break;
    }
  }
  if (allExist) { console.log(`  Skip ${startFrame}-${endFrame - 1}`); return; }

  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: [
      `--window-size=${WIDTH},${HEIGHT}`,
      '--no-sandbox', '--disable-setuid-sandbox',
      '--disable-dev-shm-usage', '--disable-gpu',
      '--hide-scrollbars', '--mute-audio'
    ]
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 1 });
    await page.goto(`file://${HTML_PATH}`, { waitUntil: 'networkidle0', timeout: 30000 });
    await new Promise(r => setTimeout(r, 300));

    for (let f = startFrame; f < endFrame; f++) {
      const t = f / FPS;
      await page.evaluate((time) => window.seekTo(time), t);
      const fp = path.join(FRAMES_DIR, `frame_${String(f).padStart(6, '0')}.png`);
      await page.screenshot({ path: fp, type: 'png', clip: { x: 0, y: 0, width: WIDTH, height: HEIGHT } });
      if (f % (FPS * 30) === 0) console.log(`  ${f}/${endFrame - 1} (${(f / FPS).toFixed(1)}s)`);
    }
  } finally {
    await browser.close();
  }
}

async function main() {
  const totalFrames = TOTAL_DURATION * FPS;
  console.log(`Rendering ${totalFrames} frames (${TOTAL_DURATION}s @ ${FPS}fps, ${Math.ceil(TOTAL_DURATION / CHUNK_SEC)} chunks)`);
  const startTime = Date.now();

  for (let sec = 0; sec < TOTAL_DURATION; sec += CHUNK_SEC) {
    const chunkEnd = Math.min(sec + CHUNK_SEC, TOTAL_DURATION);
    const cs = Date.now();
    process.stdout.write(`Chunk ${sec}-${chunkEnd}s ... `);
    await renderChunk(sec, chunkEnd);
    console.log(`${((Date.now() - cs) / 1000).toFixed(1)}s (total: ${((Date.now() - startTime) / 1000).toFixed(0)}s, ${((sec / TOTAL_DURATION) * 100).toFixed(0)}%)`);
  }

  console.log(`\nDone in ${((Date.now() - startTime) / 1000 / 60).toFixed(1)}min`);
  console.log(`ffmpeg -y -framerate ${FPS} -i ${FRAMES_DIR}/frame_%06d.png -c:v libx264 -crf 18 -pix_fmt yuv420p -preset slow ../assets/splatdb-explainer.mp4`);
}

main().catch(e => { console.error(e); process.exit(1); });

const puppeteer = require('puppeteer');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const FPS = 30;
const WIDTH = 1920;
const HEIGHT = 1080;
const OUTPUT = path.join(__dirname, 'splatdb-explainer.mp4');
const FRAMES_DIR = path.join(__dirname, 'frames');
const CHROME = '/root/.cache/puppeteer/chrome/linux-146.0.7680.153/chrome-linux64/chrome';
const CHUNK_SEC = 10;
const TOTAL_DURATION = 75;

async function renderChunk(startSec, endSec, retry = 0) {
  console.log(`Rendering ${startSec}s-${endSec}s${retry ? ` (retry ${retry})` : ''}`);
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu',
           '--disable-dev-shm-usage', '--js-flags=--max-old-space-size=512',
           `--window-size=${WIDTH},${HEIGHT}`],
    executablePath: CHROME,
  });
  const page = await browser.newPage();
  await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 1 });
  const html = fs.readFileSync(path.join(__dirname, 'scenes.html'), 'utf8');
  await page.setContent(html, { waitUntil: 'networkidle0' });

  const startFrame = Math.floor(startSec * FPS);
  const endFrame = Math.floor(endSec * FPS);
  try {
    for (let i = startFrame; i <= endFrame; i++) {
      const time = i / FPS;
      await page.evaluate((t) => window.seekTo(t), time);
      await page.screenshot({
        path: path.join(FRAMES_DIR, `frame_${String(i).padStart(6, '0')}.png`),
        type: 'png',
      });
    }
    console.log(`  Done: frames ${startFrame}-${endFrame}`);
  } catch (e) {
    console.log(`  Error at frame level, will retry: ${e.message.slice(0, 80)}`);
    if (retry < 2) {
      // Find last successful frame
      const frames = fs.readdirSync(FRAMES_DIR).filter(f => f.endsWith('.png')).sort();
      const lastFrame = frames.length ? parseInt(frames[frames.length - 1].match(/\d+/)[0]) : startFrame;
      const resumeSec = lastFrame / FPS;
      console.log(`  Resuming from ${resumeSec}s in this chunk`);
      await browser.close();
      return renderChunk(Math.max(resumeSec, startSec), endSec, retry + 1);
    }
    throw e;
  }
  await browser.close();
}

async function main() {
  if (fs.existsSync(FRAMES_DIR)) fs.rmSync(FRAMES_DIR, { recursive: true });
  fs.mkdirSync(FRAMES_DIR, { recursive: true });

  for (let s = 0; s < TOTAL_DURATION; s += CHUNK_SEC) {
    const end = Math.min(s + CHUNK_SEC, TOTAL_DURATION);
    await renderChunk(s, end);
  }

  const frameCount = fs.readdirSync(FRAMES_DIR).filter(f => f.endsWith('.png')).length;
  console.log(`\nTotal frames: ${frameCount}`);
  console.log('Encoding MP4...');
  execSync(
    `ffmpeg -y -framerate ${FPS} -i "${FRAMES_DIR}/frame_%06d.png" ` +
    `-c:v libx264 -pix_fmt yuv420p -preset slow -crf 18 -movflags +faststart "${OUTPUT}"`,
    { stdio: 'pipe' }
  );
  console.log(`Done: ${OUTPUT} (${(fs.statSync(OUTPUT).size / 1024 / 1024).toFixed(1)}MB)`);
}

main().catch(e => { console.error(e.message); process.exit(1); });

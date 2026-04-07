const puppeteer = require('puppeteer');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const FPS = 30;
const WIDTH = 1920;
const HEIGHT = 1080;
const OUTPUT = path.join(__dirname, 'splatdb-explainer.mp4');
const FRAMES_DIR = path.join(__dirname, 'frames');

async function main() {
  if (fs.existsSync(FRAMES_DIR)) fs.rmSync(FRAMES_DIR, { recursive: true });
  fs.mkdirSync(FRAMES_DIR, { recursive: true });

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu',
           `--window-size=${WIDTH},${HEIGHT}`],
    executablePath: process.env.CHROME_PATH || undefined,
  });

  const page = await browser.newPage();
  await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 1 });

  const html = fs.readFileSync(path.join(__dirname, 'scenes.html'), 'utf8');
  await page.setContent(html, { waitUntil: 'networkidle0' });

  const totalDuration = await page.evaluate(() => window.TOTAL_DURATION);
  const totalFrames = Math.ceil(totalDuration * FPS);

  console.log(`Rendering ${totalFrames} frames (${totalDuration}s at ${FPS}fps)...`);

  for (let i = 0; i <= totalFrames; i++) {
    const time = i / FPS;
    await page.evaluate((t) => window.seekTo(t), time);
    const frameNum = String(i).padStart(6, '0');
    await page.screenshot({
      path: path.join(FRAMES_DIR, `frame_${frameNum}.png`),
      type: 'png',
    });
    if (i % (FPS * 5) === 0) {
      console.log(`  Frame ${i}/${totalFrames} (${time.toFixed(1)}s)`);
    }
  }

  console.log('Encoding MP4...');
  execSync(
    `ffmpeg -y -framerate ${FPS} -i "${FRAMES_DIR}/frame_%06d.png" ` +
    `-c:v libx264 -pix_fmt yuv420p -preset slow -crf 18 ` +
    `-movflags +faststart "${OUTPUT}"`,
    { stdio: 'inherit' }
  );

  console.log(`Done: ${OUTPUT}`);
  await browser.close();
}

main().catch(console.error);

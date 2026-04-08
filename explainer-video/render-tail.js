const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const FPS = 30;
const WIDTH = 1920;
const HEIGHT = 1080;
const FRAMES_DIR = path.join(__dirname, 'frames');
const CHROME = '/root/.cache/puppeteer/chrome/linux-146.0.7680.153/chrome-linux64/chrome';

async function main() {
  const existing = fs.readdirSync(FRAMES_DIR).filter(f => f.endsWith('.png')).length;
  const lastFrame = existing ? parseInt(
    fs.readdirSync(FRAMES_DIR).filter(f => f.endsWith('.png')).sort().pop().match(/\d+/)[0]
  ) : 0;
  
  console.log(`Existing frames: ${existing}, last: ${lastFrame}`);
  
  // We need frames 2022 to 2250 (67s to 75s)
  const startFrame = lastFrame + 1;
  const endFrame = 2250;
  
  if (startFrame > endFrame) {
    console.log('All frames already rendered!');
    return;
  }
  
  console.log(`Rendering frames ${startFrame}-${endFrame}`);
  
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu',
           '--disable-dev-shm-usage', `--window-size=${WIDTH},${HEIGHT}`],
    executablePath: CHROME,
  });
  const page = await browser.newPage();
  await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 1 });
  const html = fs.readFileSync(path.join(__dirname, 'scenes.html'), 'utf8');
  await page.setContent(html, { waitUntil: 'networkidle0' });

  for (let i = startFrame; i <= endFrame; i++) {
    const time = i / FPS;
    await page.evaluate((t) => window.seekTo(t), time);
    await page.screenshot({
      path: path.join(FRAMES_DIR, `frame_${String(i).padStart(6, '0')}.png`),
      type: 'png',
    });
    if (i % (FPS * 2) === 0) console.log(`  Frame ${i}/${endFrame}`);
  }
  console.log('Done!');
  await browser.close();
}

main().catch(e => { console.error(e.message); process.exit(1); });

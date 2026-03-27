#!/usr/bin/env node
/**
 * Generate PDF from paper_ndna.html using Puppeteer.
 * Waits for MathJax to finish rendering before printing.
 */

import puppeteer from 'puppeteer';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const htmlPath = path.join(__dirname, 'paper_ndna.html');
const pdfPath = path.join(__dirname, 'paper_ndna.pdf');

async function main() {
  console.log('Launching browser...');
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // Set viewport to A4-ish dimensions
  await page.setViewport({ width: 794, height: 1123 });

  const fileUrl = `file://${htmlPath}`;
  console.log(`Loading ${fileUrl}`);

  await page.goto(fileUrl, {
    waitUntil: 'networkidle0',
    timeout: 60000
  });

  console.log('Waiting for MathJax to load...');

  // Wait for MathJax object to exist
  await page.waitForFunction(
    () => typeof window.MathJax !== 'undefined' && window.MathJax.startup,
    { timeout: 30000 }
  );

  console.log('MathJax loaded. Waiting for rendering to complete...');

  // Wait for MathJax rendering to complete
  await page.waitForFunction(
    () => window.__mathjax_ready === true,
    { timeout: 60000 }
  );

  // Extra wait to ensure all SVG rendering is flushed
  await new Promise(r => setTimeout(r, 2000));

  console.log('MathJax rendering complete. Generating PDF...');

  await page.pdf({
    path: pdfPath,
    format: 'A4',
    margin: {
      top: '2cm',
      bottom: '2cm',
      left: '2cm',
      right: '2cm'
    },
    printBackground: true,
    displayHeaderFooter: false,
    preferCSSPageSize: false
  });

  console.log(`PDF written to ${pdfPath}`);

  await browser.close();
  console.log('Done.');
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});

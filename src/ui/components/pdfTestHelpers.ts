import { jsPDF } from 'jspdf';

interface Draw {
  x: number;
  y: number;
  size: number;
  text: string;
}

/**
 * Extract every text draw (x, y baseline, font size, string) from an
 * uncompressed jsPDF content stream. jsPDF emits one `BT … Td … Tj … ET` block
 * per positioned draw, and a `T*` for every hard line break inside a string.
 */
/** Concatenated (uncompressed) page content streams. */
export function rawContent(doc: jsPDF): string {
  const bytes = new Uint8Array(doc.output('arraybuffer'));
  let raw = '';
  for (const byte of bytes) raw += String.fromCharCode(byte); // latin1
  return [...raw.matchAll(/stream\r?\n([\s\S]*?)endstream/g)].map((m) => m[1]).join('\n');
}

export function extractDraws(doc: jsPDF): Draw[] {
  const draws: Draw[] = [];

  {
    let size = 0;
    let x = 0;
    let y = 0;
    for (const line of rawContent(doc).split(/\r?\n/)) {
      const tf = line.match(/^\/F\d+ ([\d.]+) Tf/);
      if (tf) { size = parseFloat(tf[1]); continue; }
      const td = line.match(/^([\d.-]+) ([\d.-]+) Td/);
      if (td) { x = parseFloat(td[1]); y = parseFloat(td[2]); continue; }
      if (line === 'ET') { x = 0; y = 0; continue; }

      // A single line can hold several ops, e.g. `T* (The) Tj`.
      const ops = /T\*|\((?:\\.|[^()\\])*\) Tj/g;
      let op: RegExpExecArray | null;
      while ((op = ops.exec(line)) !== null) {
        if (op[0] === 'T*') { y -= size * 1.15; continue; }
        const text = op[0].slice(1, -4).replace(/\\([()\\])/g, '$1');
        draws.push({ x, y, size, text });
        x += textWidth(text, size);
      }
    }
  }
  return draws;
}

// Rough advance widths: space is ~0.28em, everything else ~0.5em for Helvetica.
function textWidth(text: string, size: number): number {
  let w = 0;
  for (const ch of text) w += ch === ' ' ? size * 0.278 : size * 0.5;
  return w;
}

/**
 * Draws that sit on *different* baselines but whose bounding boxes still
 * intersect — this is the text-on-text collision the user sees in the report.
 * Adjacent draws sharing one baseline are excluded: the rough width estimate
 * makes neighbouring words look like they touch.
 */
export function findCrossBaselineCollisions(draws: Draw[]): string[] {
  const boxes = draws
    .filter((d) => d.text.trim() !== '')
    .map((d) => ({
      ...d,
      left: d.x,
      right: d.x + textWidth(d.text, d.size),
      top: d.y + d.size * 0.25,
      bottom: d.y - d.size * 0.8,
    }));

  const collisions: string[] = [];
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i];
      const b = boxes[j];
      if (Math.abs(a.y - b.y) < 0.01) continue;
      const vOverlap = Math.min(a.top, b.top) - Math.max(a.bottom, b.bottom);
      const hOverlap = Math.min(a.right, b.right) - Math.max(a.left, b.left);
      if (vOverlap > 1 && hOverlap > 1) {
        collisions.push(
          `${JSON.stringify(a.text)} @(${a.x.toFixed(1)},${a.y.toFixed(1)}) overlaps ${JSON.stringify(b.text)} @(${b.x.toFixed(1)},${b.y.toFixed(1)})`,
        );
      }
    }
  }
  return collisions;
}

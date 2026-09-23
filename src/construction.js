const CROWN = 1.00;
const HAIRLINE = 0.68;
const BROW = 0.18;
const EYE = 0.07;
const NOSE = -0.32;
const MOUTH = -0.49;
const CHIN = -0.76;
const BOTTOM = -0.94;

function smooth01(t) {
  t = Math.max(0, Math.min(1, t));
  return t * t * (3 - 2 * t);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function segment(y, yTop, yBottom, valueTop, valueBottom) {
  return lerp(valueTop, valueBottom, smooth01((yTop - y) / (yTop - yBottom)));
}

function compactEllipse(x, y, cx, cy, rx, ry, inner) {
  const dx = (x - cx) / rx;
  const dy = (y - cy) / ry;
  const d = dx * dx + dy * dy;
  if (d >= 1) return 0;
  if (d <= inner) return 1;
  return 1 - smooth01((d - inner) / (1 - inner));
}

// Exact JS mirrors of the construction constraints in kernels/face.cu.
function silhouetteHalfWidth(y, p) {
  const temple = p.templeWidth / 0.88;
  const cheek = p.cheekWidth / 1.06;
  const jaw = p.jawWidth / 0.72;
  let scale;

  if (y > 0.78) scale = segment(y, 1.00, 0.78, 0.30, 0.93);
  else if (y > 0.30) scale = segment(y, 0.78, 0.30, 0.93, 0.84 * temple);
  else if (y > -0.08) scale = segment(y, 0.30, -0.08, 0.84 * temple, 0.90 * cheek);
  else if (y > -0.52) scale = segment(y, -0.08, -0.52, 0.90 * cheek, 0.72 * jaw);
  else if (y > -0.76) scale = segment(y, -0.52, -0.76, 0.72 * jaw, 0.40);
  else scale = segment(y, -0.76, -0.94, 0.40, 0.16);

  return p.headWidth * scale;
}

function baseProfileDepth(y, p) {
  const depthScale = p.headDepth / 0.90;
  const chin = p.chinProjection;
  const muzzle = p.muzzleProjection;
  let z;

  if (y > 0.68) z = segment(y, 1.00, 0.68, 0.10, 0.31);
  else if (y > 0.18) z = segment(y, 0.68, 0.18, 0.31, 0.36);
  else if (y > 0.08) z = segment(y, 0.18, 0.08, 0.36, 0.30);
  else if (y > -0.32) z = segment(y, 0.08, -0.32, 0.30, 0.29);
  else if (y > -0.49) z = segment(y, -0.32, -0.49, 0.29, 0.33 * muzzle);
  else if (y > -0.60) z = segment(y, -0.49, -0.60, 0.33 * muzzle, 0.30);
  else if (y > -0.72) z = segment(y, -0.60, -0.72, 0.27, 0.36 * chin);
  else z = segment(y, -0.72, -0.94, 0.36 * chin, 0.10);

  return z * depthScale;
}

function centerFeatureDepth(y, p) {
  const eyeHalf = p.headWidth * 0.18 * (p.eyeWidth / 0.92);
  const eyeCenter = eyeHalf * 2 * (p.eyeSpacing / 0.29);
  const innerCorner = Math.max(eyeCenter - eyeHalf, p.headWidth * 0.10);
  const wingHalf = innerCorner * p.noseWidth;
  const mouthHalf = eyeCenter * p.mouthWidth;

  let z = 0;

  z += 0.014 * p.browProjection *
    compactEllipse(0, y, 0, 0.155, innerCorner * 0.55, 0.070, 0.10);
  z -= 0.020 *
    compactEllipse(0, y, 0, 0.085, innerCorner * 0.48, 0.055, 0.10);

  if (y <= 0.09 && y >= -0.245) {
    const t = Math.max(0, Math.min(1, (0.09 - y) / 0.335));
    z += p.noseProjection * lerp(0.012, 0.115, smooth01(t));
  }

  z += p.noseProjection * 0.078 * compactEllipse(
    0, y, 0, -0.275, Math.max(wingHalf * 0.62, 0.050), 0.060, 0.12
  );

  z += 0.026 * compactEllipse(
    0, y, 0, -0.465, Math.max(mouthHalf * 1.10, 0.18), 0.165, 0.12
  );

  z -= 0.007 * compactEllipse(
    0, y, 0, -0.405, Math.max(wingHalf * 0.28, 0.034), 0.050, 0.10
  );

  z += p.upperLip * 0.011 * compactEllipse(
    0, y, 0, -0.485, Math.max(mouthHalf * 0.24, 0.042), 0.030, 0.10
  );

  const lowerRx = Math.max(mouthHalf * 0.44, 0.060);
  z += p.lowerLip * 0.014 * (
    compactEllipse(0, y, -mouthHalf * 0.21, -0.548, lowerRx, 0.038, 0.08) +
    compactEllipse(0, y,  mouthHalf * 0.21, -0.548, lowerRx, 0.038, 0.08)
  );

  if (y > -0.53 && y < -0.50) {
    z -= 0.0045 * (1 - smooth01(Math.abs(y + 0.515) / 0.015));
  }

  return z;
}

function centerProfileDepth(y, p) {
  return baseProfileDepth(y, p) + centerFeatureDepth(y, p);
}

function setup(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0a0d12';
  ctx.fillRect(0, 0, w, h);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  return { ctx, w, h };
}

function guide(ctx, y, x0, x1, label, strong = false) {
  ctx.save();
  ctx.strokeStyle = strong ? '#33445a' : '#243041';
  ctx.lineWidth = strong ? 1.1 : 1;
  ctx.setLineDash([3, 5]);
  ctx.beginPath();
  ctx.moveTo(x0, y);
  ctx.lineTo(x1, y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = strong ? '#77869b' : '#536177';
  ctx.font = '8px ui-monospace, monospace';
  ctx.fillText(label, x0 + 3, y - 4);
  ctx.restore();
}

function strokePolyline(ctx, points, color, width = 1.2) {
  if (!points.length) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  ctx.stroke();
  ctx.restore();
}

function dot(ctx, x, y, color, r = 2.3) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

export function drawConstruction(frontCanvas, profileCanvas, p) {
  drawFront(frontCanvas, p);
  drawProfile(profileCanvas, p);
}

function drawFront(canvas, p) {
  const { ctx, w, h } = setup(canvas);
  const cx = w * 0.5;
  const top = 27;
  const bottom = h - 27;
  const H = bottom - top;
  const sy = y => top + ((CROWN - y) / (CROWN - BOTTOM)) * H;
  const xScale = 103;
  const sx = x => cx + x * xScale;

  for (const [y, label, strong] of [
    [HAIRLINE, 'HAIR', false],
    [BROW, 'BROW', true],
    [EYE, 'EYES / MID', true],
    [NOSE, 'NOSE', true],
    [MOUTH, 'MOUTH', false],
    [CHIN, 'CHIN', true]
  ]) guide(ctx, sy(y), 17, w - 17, label, strong);

  const right = [];
  const left = [];
  const sideRight = [];
  const sideLeft = [];

  for (let i = 0; i <= 110; i++) {
    const y = CROWN + (BOTTOM - CROWN) * (i / 110);
    const half = silhouetteHalfWidth(y, p);
    right.push([sx(half), sy(y)]);
    left.push([sx(-half), sy(y)]);
    sideRight.push([sx(half * 0.52), sy(y)]);
    sideLeft.push([sx(-half * 0.52), sy(y)]);
  }

  strokePolyline(ctx, right, '#a6b1c0', 1.7);
  strokePolyline(ctx, left, '#a6b1c0', 1.7);
  strokePolyline(ctx, sideRight, '#45566e', 0.9);
  strokePolyline(ctx, sideLeft, '#45566e', 0.9);

  ctx.strokeStyle = '#3c4b5e';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx, sy(CROWN));
  ctx.lineTo(cx, sy(BOTTOM));
  ctx.stroke();

  const eyeHalf = p.headWidth * 0.18 * (p.eyeWidth / 0.92);
  const eyeCenter = eyeHalf * 2 * (p.eyeSpacing / 0.29);
  const innerCorner = Math.max(eyeCenter - eyeHalf, p.headWidth * 0.10);
  const wingHalf = innerCorner * p.noseWidth;
  const mouthHalf = eyeCenter * p.mouthWidth;
  const socketRx = eyeHalf * 1.22;

  // Five-eye scaffold.
  ctx.save();
  ctx.strokeStyle = '#2c394c';
  ctx.lineWidth = 0.8;
  ctx.setLineDash([2, 4]);
  for (let i = -2; i <= 2; i++) {
    const center = i * eyeHalf * 2;
    const x0 = sx(center - eyeHalf);
    const x1 = sx(center + eyeHalf);
    ctx.strokeRect(x0, sy(EYE) - 9, x1 - x0, 18);
  }
  ctx.restore();

  // Socket boundaries.
  for (const sign of [-1, 1]) {
    ctx.save();
    ctx.strokeStyle = '#75b9e9';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.ellipse(sx(sign * eyeCenter), sy(EYE), socketRx * xScale, 17, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // Nose top/side planes from the same eye-derived dimensions as CUDA.
  const rootHalf = wingHalf * 0.27;
  const lowerTopHalf = wingHalf * 0.43;
  const lowerSideHalf = wingHalf * 0.80;
  const rootY = sy(0.09);
  const lowerY = sy(-0.245);
  const baseY = sy(NOSE);

  strokePolyline(ctx, [[sx(-rootHalf), rootY], [sx(-lowerTopHalf), lowerY], [sx(-wingHalf * 0.62), baseY]], '#d6a27e', 1.3);
  strokePolyline(ctx, [[sx( rootHalf), rootY], [sx( lowerTopHalf), lowerY], [sx( wingHalf * 0.62), baseY]], '#d6a27e', 1.3);
  strokePolyline(ctx, [[sx(-rootHalf), rootY], [sx(-lowerSideHalf), lowerY], [sx(-wingHalf), baseY]], '#8f6d59', 0.95);
  strokePolyline(ctx, [[sx( rootHalf), rootY], [sx( lowerSideHalf), lowerY], [sx( wingHalf), baseY]], '#8f6d59', 0.95);

  // Denture cylinder and mouth construction.
  ctx.save();
  ctx.strokeStyle = '#405067';
  ctx.setLineDash([3, 4]);
  ctx.lineWidth = 0.9;
  ctx.beginPath();
  ctx.ellipse(cx, sy(-0.465), mouthHalf * 1.10 * xScale, 25, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  ctx.strokeStyle = '#bc7c86';
  ctx.lineWidth = 1.15;
  ctx.beginPath();
  ctx.moveTo(sx(-mouthHalf), sy(MOUTH));
  ctx.bezierCurveTo(
    sx(-mouthHalf * 0.45), sy(MOUTH + 0.015),
    sx(-mouthHalf * 0.18), sy(MOUTH + 0.022),
    cx, sy(MOUTH + 0.012)
  );
  ctx.bezierCurveTo(
    sx(mouthHalf * 0.18), sy(MOUTH + 0.022),
    sx(mouthHalf * 0.45), sy(MOUTH + 0.015),
    sx(mouthHalf), sy(MOUTH)
  );
  ctx.stroke();

  ctx.fillStyle = '#6a7789';
  ctx.font = '8px ui-monospace, monospace';
  ctx.fillText('exact CUDA silhouette + landmark scaffold', 10, h - 9);
}

function drawProfile(canvas, p) {
  const { ctx, w, h } = setup(canvas);
  const baseX = 108;
  const top = 27;
  const bottom = h - 27;
  const H = bottom - top;
  const sy = y => top + ((CROWN - y) / (CROWN - BOTTOM)) * H;
  const depthScale = 120;
  const sz = z => baseX + z * depthScale;

  for (const [y, label, strong] of [
    [HAIRLINE, 'HAIR', false],
    [BROW, 'BROW', true],
    [EYE, 'EYES / MID', true],
    [NOSE, 'NOSE', true],
    [MOUTH, 'MOUTH', false],
    [CHIN, 'CHIN', true]
  ]) guide(ctx, sy(y), 15, w - 15, label, strong);

  // Exact CUDA center-profile constraint.
  const profile = [];
  for (let i = 0; i <= 150; i++) {
    const y = CROWN + (BOTTOM - CROWN) * (i / 150);
    profile.push([sz(centerProfileDepth(y, p)), sy(y)]);
  }
  strokePolyline(ctx, profile, '#d7e0ea', 1.8);

  // Schematic back of cranium only; the front curve above is exact.
  ctx.save();
  ctx.strokeStyle = '#56667b';
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  ctx.moveTo(baseX, sy(CROWN));
  ctx.bezierCurveTo(baseX - 54, sy(0.84), baseX - 82, sy(0.40), baseX - 73, sy(-0.08));
  ctx.bezierCurveTo(baseX - 62, sy(-0.42), baseX - 34, sy(-0.70), baseX - 10, sy(BOTTOM));
  ctx.stroke();
  ctx.restore();

  for (const [y, label, color] of [
    [0.18, 'GLABELLA', '#91a8c0'],
    [0.085, 'NASION', '#d6a27e'],
    [-0.245, 'DORSUM', '#d6a27e'],
    [-0.275, 'TIP', '#d6a27e'],
    [-0.32, 'BASE', '#d6a27e'],
    [-0.49, 'MOUTH', '#bc7c86'],
    [-0.60, 'GROOVE', '#8d817b'],
    [-0.72, 'CHIN', '#8d817b']
  ]) {
    const x = sz(centerProfileDepth(y, p));
    dot(ctx, x, sy(y), color);
    ctx.fillStyle = color;
    ctx.font = '7px ui-monospace, monospace';
    ctx.fillText(label, x + 5, sy(y) - 3);
  }

  // Construction masses.
  const eyeHalf = p.headWidth * 0.18 * (p.eyeWidth / 0.92);

  ctx.save();
  ctx.strokeStyle = '#75b9e9';
  ctx.globalAlpha = 0.8;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(sz(baseProfileDepth(EYE, p) - 0.01), sy(EYE), eyeHalf * 34, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = '#405067';
  ctx.setLineDash([3, 4]);
  ctx.lineWidth = 0.9;
  ctx.beginPath();
  ctx.ellipse(sz(baseProfileDepth(-0.47, p) - 0.015), sy(-0.47), 18, 37, 0.08, -Math.PI / 2, Math.PI / 2);
  ctx.stroke();
  ctx.restore();

  ctx.fillStyle = '#6a7789';
  ctx.font = '8px ui-monospace, monospace';
  ctx.fillText('exact CUDA center-profile constraint', 10, h - 9);
}

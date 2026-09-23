const TAU = Math.PI * 2;

// The same construction landmarks used by kernels/face.cu.
// Crown-to-hairline is approximately the top sixth; the visible face then
// falls into roughly equal hairline/brow/nose/chin thirds.
const CROWN = 0.96;
const HAIRLINE = 0.68;
const BROW = 0.18;
const EYE = 0.07;
const NOSE = -0.32;
const MOUTH = -0.49;
const CHIN = -0.82;

function setup(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
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
  ctx.lineWidth = strong ? 1.15 : 1;
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

function ellipse(ctx, x, y, rx, ry, stroke = '#7ce2c0', alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.ellipse(x, y, rx, ry, 0, 0, TAU);
  ctx.stroke();
  ctx.restore();
}

function strokePath(ctx, points, color, width = 1.2, close = false) {
  if (!points.length) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  if (close) ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

function mirrorPoints(points, cx) {
  return points.map(([x, y]) => [cx - (x - cx), y]);
}

export function drawConstruction(frontCanvas, profileCanvas, p) {
  drawFront(frontCanvas, p);
  drawProfile(profileCanvas, p);
}

function drawFront(canvas, p) {
  const { ctx, w, h } = setup(canvas);
  const cx = w * 0.5;
  const top = 28;
  const bottom = h - 28;
  const H = bottom - top;

  const sy = n => top + ((CROWN - n) / (CROWN - CHIN)) * H;
  const sx = n => cx + n * 94;

  for (const [n, name, strong] of [
    [HAIRLINE, 'HAIR', false],
    [BROW, 'BROW', true],
    [EYE, 'EYES / MID', true],
    [NOSE, 'NOSE', true],
    [MOUTH, 'MOUTH', false],
    [CHIN, 'CHIN', true]
  ]) {
    guide(ctx, sy(n), 19, w - 19, name, strong);
  }

  // ------------------------------------------------------------------
  // 1) HELMET HEAD / SILHOUETTE
  // ------------------------------------------------------------------
  const rightOutline = [
    [cx, sy(CROWN)],
    [sx(0.53 * p.headWidth), sy(0.82)],
    [sx(0.82 * p.headWidth), sy(0.43)],
    [sx(0.70 * p.headWidth * p.templeWidth), sy(BROW)],
    [sx(0.72 * p.headWidth * p.cheekWidth), sy(-0.08)],
    [sx(0.55 * p.headWidth * p.jawWidth), sy(-0.61)],
    [sx(0.20 * p.headWidth), sy(CHIN)],
    [cx, sy(CHIN + 0.015)]
  ];
  strokePath(ctx, rightOutline, '#a6b1c0', 1.6);
  strokePath(ctx, mirrorPoints(rightOutline, cx), '#a6b1c0', 1.6);

  // Slice/cut line for the side plane: hairline -> temple -> cheek.
  const rightSidePlane = [
    [sx(0.50 * p.headWidth), sy(HAIRLINE)],
    [sx(0.60 * p.headWidth * p.templeWidth), sy(BROW)],
    [sx(0.61 * p.headWidth * p.cheekWidth), sy(-0.07)]
  ];
  strokePath(ctx, rightSidePlane, '#516077', 1.0);
  strokePath(ctx, mirrorPoints(rightSidePlane, cx), '#516077', 1.0);

  // Center line.
  ctx.strokeStyle = '#3c4b5e';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx, sy(CROWN));
  ctx.lineTo(cx, sy(CHIN));
  ctx.stroke();

  // ------------------------------------------------------------------
  // 2) FIVE-EYE CONSTRUCTION
  // ------------------------------------------------------------------
  // The default head is constructed around five eye widths, with one
  // eye-width between the eyes. Controls are deviations from that scaffold.
  const eyeHalf = 15.0 * p.eyeWidth;
  const eyeCenter = eyeHalf * 2.0 * (p.eyeSpacing / 0.29);
  const eyeY = sy(EYE);
  const eyeH = 8.0;

  ctx.save();
  ctx.strokeStyle = '#304058';
  ctx.lineWidth = 0.8;
  ctx.setLineDash([2, 4]);
  for (let i = -2; i <= 2; i++) {
    const x = cx + i * eyeHalf * 2;
    ctx.strokeRect(x - eyeHalf, eyeY - 11, eyeHalf * 2, 22);
  }
  ctx.restore();

  // Eye sockets first, not almond-shaped eyes pasted on the face.
  const socket = (centerX, flip) => {
    const outer = centerX + flip * eyeHalf * 1.22;
    const inner = centerX - flip * eyeHalf * 1.03;
    const points = [
      [inner, eyeY - 4],
      [centerX, eyeY - 9],
      [outer, eyeY - 5],
      [outer - flip * 3, eyeY + 7],
      [centerX, eyeY + 10],
      [inner + flip * 2, eyeY + 6]
    ];
    strokePath(ctx, points, '#75b9e9', 1.25, true);
  };
  socket(cx - eyeCenter, -1);
  socket(cx + eyeCenter, 1);

  // Brow blocks / socket roof.
  strokePath(ctx, [
    [cx - eyeCenter - eyeHalf, sy(BROW) + 2],
    [cx - eyeCenter, sy(BROW) - 4],
    [cx - 4, sy(BROW) + 1]
  ], '#7a9fc2', 1.1);
  strokePath(ctx, mirrorPoints([
    [cx - eyeCenter - eyeHalf, sy(BROW) + 2],
    [cx - eyeCenter, sy(BROW) - 4],
    [cx - 4, sy(BROW) + 1]
  ], cx), '#7a9fc2', 1.1);

  // Keystone / glabella is its own small plane.
  strokePath(ctx, [
    [cx, sy(BROW) - 2],
    [cx - 7, sy(0.12)],
    [cx, sy(0.07)],
    [cx + 7, sy(0.12)]
  ], '#d1a177', 1.15, true);

  // ------------------------------------------------------------------
  // 3) NOSE AS A BOX/WEDGE
  // ------------------------------------------------------------------
  const innerCorner = Math.max(eyeCenter - eyeHalf, 9);
  const wingHalf = innerCorner * p.noseWidth;
  const rootHalf = wingHalf * 0.28;
  const dorsumHalf = wingHalf * 0.44;
  const rootY = sy(0.085);
  const lowerDorsumY = sy(-0.235);
  const noseY = sy(NOSE);

  // Top plane edges.
  strokePath(ctx, [
    [cx - rootHalf, rootY],
    [cx - dorsumHalf, lowerDorsumY],
    [cx - wingHalf * 0.45, noseY - 6]
  ], '#d6a27e', 1.35);
  strokePath(ctx, mirrorPoints([
    [cx - rootHalf, rootY],
    [cx - dorsumHalf, lowerDorsumY],
    [cx - wingHalf * 0.45, noseY - 6]
  ], cx), '#d6a27e', 1.35);

  // Wide side planes and alar wings.
  strokePath(ctx, [
    [cx - rootHalf, rootY],
    [cx - wingHalf * 0.72, lowerDorsumY],
    [cx - wingHalf, noseY],
    [cx - wingHalf * 0.45, noseY - 6]
  ], '#936f58', 1.0);
  strokePath(ctx, mirrorPoints([
    [cx - rootHalf, rootY],
    [cx - wingHalf * 0.72, lowerDorsumY],
    [cx - wingHalf, noseY],
    [cx - wingHalf * 0.45, noseY - 6]
  ], cx), '#936f58', 1.0);

  // Bottom plane / rhythm between nostrils.
  ctx.strokeStyle = '#b98469';
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  ctx.moveTo(cx - wingHalf, noseY);
  ctx.quadraticCurveTo(cx - wingHalf * 0.45, noseY + 4, cx, noseY + 1);
  ctx.quadraticCurveTo(cx + wingHalf * 0.45, noseY + 4, cx + wingHalf, noseY);
  ctx.stroke();

  // ------------------------------------------------------------------
  // 4) DENTURE CYLINDER + MOUTH
  // ------------------------------------------------------------------
  const mouthHalf = eyeCenter * p.mouthWidth;
  const mouthY = sy(MOUTH);
  ctx.save();
  ctx.strokeStyle = '#3f4d60';
  ctx.lineWidth = 0.9;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.ellipse(cx, mouthY - 3, mouthHalf * 1.12, 28, 0, Math.PI * 0.10, Math.PI * 0.90);
  ctx.stroke();
  ctx.restore();

  // Mouth corners track approximately under pupil/eye-center construction.
  ctx.strokeStyle = '#bc7c86';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(cx - mouthHalf, mouthY);
  ctx.bezierCurveTo(
    cx - mouthHalf * 0.45, mouthY - 2.5 * p.upperLip,
    cx - mouthHalf * 0.18, mouthY - 4.0 * p.upperLip,
    cx, mouthY - 2.0 * p.upperLip);
  ctx.bezierCurveTo(
    cx + mouthHalf * 0.18, mouthY - 4.0 * p.upperLip,
    cx + mouthHalf * 0.45, mouthY - 2.5 * p.upperLip,
    cx + mouthHalf, mouthY);
  ctx.bezierCurveTo(
    cx + mouthHalf * 0.38, mouthY + 5.0 * p.lowerLip,
    cx - mouthHalf * 0.38, mouthY + 5.0 * p.lowerLip,
    cx - mouthHalf, mouthY);
  ctx.stroke();

  // Chin mass and labiomental break.
  ellipse(ctx, cx, sy(-0.72), 24, 17, '#8a776d', 0.65);
  strokePath(ctx, [
    [cx - 21, sy(-0.61)],
    [cx, sy(-0.625)],
    [cx + 21, sy(-0.61)]
  ], '#5b4f4a', 0.9);

  ctx.fillStyle = '#6a7789';
  ctx.font = '8px ui-monospace,monospace';
  ctx.fillText('helmet → sockets → wedge → denture', 12, h - 10);
}

function drawProfile(canvas, p) {
  const { ctx, w, h } = setup(canvas);
  const cx = 128;
  const top = 28;
  const bottom = h - 28;
  const H = bottom - top;

  const sy = n => top + ((CROWN - n) / (CROWN - CHIN)) * H;
  const front = d => cx + d * 72;
  const back = d => cx - d * 80;

  for (const [n, name, strong] of [
    [HAIRLINE, 'HAIR', false],
    [BROW, 'BROW', true],
    [EYE, 'EYES / MID', true],
    [NOSE, 'NOSE', true],
    [MOUTH, 'MOUTH', false],
    [CHIN, 'CHIN', true]
  ]) {
    guide(ctx, sy(n), 17, w - 17, name, strong);
  }

  // ------------------------------------------------------------------
  // 1) CRANIUM + SIDE-PLANE CUT
  // ------------------------------------------------------------------
  ctx.strokeStyle = '#66758a';
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.moveTo(cx, sy(CROWN));
  ctx.bezierCurveTo(
    back(0.67 * p.headDepth), sy(0.88),
    back(0.93 * p.headDepth), sy(0.46),
    back(0.83 * p.headDepth), sy(0.03));
  ctx.bezierCurveTo(
    back(0.73 * p.headDepth), sy(-0.32),
    back(0.46 * p.headDepth), sy(-0.66),
    back(0.15), sy(CHIN));
  ctx.stroke();

  // Temporal side plane ellipse/cut.
  ctx.save();
  ctx.strokeStyle = '#3f4d61';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.ellipse(back(0.33 * p.headDepth), sy(0.26), 38, 70, -0.03, 0, TAU);
  ctx.stroke();
  ctx.restore();

  // ------------------------------------------------------------------
  // 2) PROFILE LANDMARKS / ANGLE CHANGES
  // ------------------------------------------------------------------
  // Plot the profile as a sequence of actual construction landmarks:
  // forehead -> glabella -> nasion -> dorsum -> tip -> subnasale ->
  // philtrum -> lips -> labiomental groove -> chin.
  const forehead = [front(0.17), sy(0.55)];
  const glabella = [front(0.22 * p.browProjection), sy(BROW)];
  const nasion = [front(0.12), sy(0.095)];
  const dorsum = [front(0.30 * p.noseProjection), sy(-0.13)];
  const tip = [front(0.47 * p.noseProjection), sy(-0.275)];
  const subnasale = [front(0.25 * p.noseProjection), sy(-0.35)];
  const philtrum = [front(0.23 * p.muzzleProjection), sy(-0.42)];
  const upperLip = [front(0.30 * p.muzzleProjection), sy(MOUTH)];
  const lowerLip = [front(0.29 * p.muzzleProjection), sy(MOUTH - 0.06)];
  const groove = [front(0.20), sy(-0.615)];
  const chin = [front(0.27 * p.chinProjection), sy(-0.72)];
  const chinBottom = [front(0.17 * p.chinProjection), sy(CHIN)];

  ctx.strokeStyle = '#d4dee9';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(cx, sy(CROWN));
  ctx.bezierCurveTo(front(0.24), sy(0.82), front(0.20), sy(0.68), forehead[0], forehead[1]);
  ctx.lineTo(glabella[0], glabella[1]);
  ctx.lineTo(nasion[0], nasion[1]);
  ctx.lineTo(dorsum[0], dorsum[1]);
  ctx.lineTo(tip[0], tip[1]);
  ctx.lineTo(subnasale[0], subnasale[1]);
  ctx.lineTo(philtrum[0], philtrum[1]);
  ctx.lineTo(upperLip[0], upperLip[1]);
  ctx.lineTo(lowerLip[0], lowerLip[1]);
  ctx.lineTo(groove[0], groove[1]);
  ctx.lineTo(chin[0], chin[1]);
  ctx.lineTo(chinBottom[0], chinBottom[1]);
  ctx.stroke();

  // Nasion/saddle is intentionally marked: the nose starts after a recess,
  // not as one uninterrupted forehead-to-tip bridge.
  ellipse(ctx, nasion[0], nasion[1], 3.1, 3.1, '#d6a27e', 1);
  ctx.fillStyle = '#b98c6d';
  ctx.font = '7px ui-monospace,monospace';
  ctx.fillText('NASION', nasion[0] + 7, nasion[1] - 3);

  // Nose box/wedge construction. The side plane is deliberately broad.
  strokePath(ctx, [
    nasion,
    dorsum,
    tip,
    subnasale
  ], '#d6a27e', 1.25);
  strokePath(ctx, [
    [nasion[0] - 8, nasion[1] + 2],
    [dorsum[0] - 11, dorsum[1] + 4],
    [tip[0] - 13, tip[1] + 2],
    subnasale
  ], '#8f6d59', 1.0);

  // Socket/eyeball relationship: sphere tucked behind the brow awning.
  ellipse(ctx, front(0.08), sy(EYE), 10, 10, '#75b9e9', 0.85);
  strokePath(ctx, [
    [front(0.02), sy(BROW)],
    [front(0.15 * p.browProjection), sy(BROW) + 2],
    [front(0.13), sy(EYE) - 8]
  ], '#7596b5', 1.0);

  // Denture / tooth cylinder under the lips.
  ctx.save();
  ctx.strokeStyle = '#455469';
  ctx.setLineDash([3, 4]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.ellipse(front(0.11), sy(-0.47), 22, 42, 0.10, -Math.PI * 0.50, Math.PI * 0.50);
  ctx.stroke();
  ctx.restore();

  // Jaw construction from the side plane toward the chin.
  strokePath(ctx, [
    [back(0.28), sy(-0.34)],
    [back(0.22), sy(-0.57)],
    [front(0.02), sy(-0.76)],
    chinBottom
  ], '#7a8798', 1.15);

  ctx.fillStyle = '#6a7789';
  ctx.font = '8px ui-monospace,monospace';
  ctx.fillText('profile = angle changes, not one curve', 12, h - 10);
}

const TAU = Math.PI * 2;

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

function guide(ctx, y, x0, x1, label) {
  ctx.save();
  ctx.strokeStyle = '#243041';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 5]);
  ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#536177';
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

export function drawConstruction(frontCanvas, profileCanvas, p) {
  drawFront(frontCanvas, p);
  drawProfile(profileCanvas, p);
}

function drawFront(canvas, p) {
  const { ctx, w, h } = setup(canvas);
  const cx = w * 0.5;
  const top = 31, bottom = h - 27, H = bottom - top;
  const sy = n => top + (1 - (n + 1) * 0.5) * H;
  const sx = n => cx + n * 92;

  for (const [n, name] of [[0.46,'BROW'],[0.27,'EYES'],[-0.07,'NOSE'],[-0.34,'MOUTH'],[-0.72,'CHIN']]) {
    guide(ctx, sy(n), 24, w - 24, name);
  }
  guide(ctx, sy(0), 24, w - 24, 'MID');

  ctx.strokeStyle = '#9aa9bb';
  ctx.lineWidth = 1.6;

  ctx.beginPath();
  ctx.moveTo(cx, top);
  ctx.bezierCurveTo(sx(.72*p.headWidth), top+4, sx(.86*p.headWidth), sy(.35), sx(.73*p.cheekWidth), sy(.05));
  ctx.bezierCurveTo(sx(.67*p.cheekWidth), sy(-.22), sx(.55*p.jawWidth), sy(-.58), sx(.18), bottom-3);
  ctx.bezierCurveTo(cx+9, bottom+1, cx+5, bottom+3, cx, bottom+3);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx, top);
  ctx.bezierCurveTo(sx(-.72*p.headWidth), top+4, sx(-.86*p.headWidth), sy(.35), sx(-.73*p.cheekWidth), sy(.05));
  ctx.bezierCurveTo(sx(-.67*p.cheekWidth), sy(-.22), sx(-.55*p.jawWidth), sy(-.58), sx(-.18), bottom-3);
  ctx.bezierCurveTo(cx-9, bottom+1, cx-5, bottom+3, cx, bottom+3);
  ctx.stroke();

  ctx.strokeStyle = '#3c4b5e';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, top+3); ctx.lineTo(cx,bottom); ctx.stroke();

  const eyeY = sy(.27), eyeX = p.eyeSpacing * 145, eyeRX = p.eyeWidth * 45;
  ellipse(ctx, cx-eyeX, eyeY, eyeRX, 7.5, '#75b9e9');
  ellipse(ctx, cx+eyeX, eyeY, eyeRX, 7.5, '#75b9e9');
  ellipse(ctx, cx-eyeX, eyeY, 3.5, 3.5, '#75b9e9', .6);
  ellipse(ctx, cx+eyeX, eyeY, 3.5, 3.5, '#75b9e9', .6);

  ctx.strokeStyle = '#d2a783';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(cx, sy(.47));
  ctx.bezierCurveTo(cx-8,sy(.28),cx-12,sy(.07),cx-15*p.noseWidth,sy(-.07));
  ctx.bezierCurveTo(cx-8,sy(-.1),cx+8,sy(-.1),cx+15*p.noseWidth,sy(-.07));
  ctx.bezierCurveTo(cx+12,sy(.07),cx+8,sy(.28),cx,sy(.47));
  ctx.stroke();

  const mouthY = sy(-.34), mouthW = 36*p.mouthWidth;
  ctx.strokeStyle = '#bb7781';
  ctx.beginPath();
  ctx.moveTo(cx-mouthW,mouthY);
  ctx.quadraticCurveTo(cx,mouthY-3.5*p.upperLip,cx+mouthW,mouthY);
  ctx.quadraticCurveTo(cx,mouthY+4.5*p.lowerLip,cx-mouthW,mouthY);
  ctx.stroke();

  ctx.fillStyle='#6a7789'; ctx.font='8px ui-monospace,monospace';
  ctx.fillText('front plane + landmarks', 12, h-10);
}

function drawProfile(canvas, p) {
  const { ctx, w, h } = setup(canvas);
  const cx = 132;
  const top = 31, bottom = h - 27, H = bottom - top;
  const sy = n => top + (1 - (n + 1) * .5) * H;
  const front = d => cx + d * 66;
  const back = d => cx - d * 82;

  for (const [n,name] of [[.46,'BROW'],[.27,'EYES'],[-.07,'NOSE'],[-.34,'MOUTH'],[-.72,'CHIN']]) {
    guide(ctx,sy(n),22,w-22,name);
  }

  ctx.strokeStyle='#7d8b9e';ctx.lineWidth=1.5;
  ctx.beginPath();
  ctx.moveTo(cx,top);
  ctx.bezierCurveTo(back(.75*p.headDepth),top+8,back(.92*p.headDepth),sy(.25),back(.77*p.headDepth),sy(-.15));
  ctx.bezierCurveTo(back(.61*p.headDepth),sy(-.5),back(.24),sy(-.85),cx, bottom);
  ctx.stroke();

  ctx.strokeStyle='#d0dae5';ctx.lineWidth=1.8;
  ctx.beginPath();
  ctx.moveTo(cx,top);
  ctx.bezierCurveTo(front(.36),sy(.72),front(.24),sy(.53),front(.20*p.browProjection),sy(.44));
  ctx.bezierCurveTo(front(.12),sy(.34),front(.11),sy(.29),front(.13),sy(.24));
  ctx.bezierCurveTo(front(.33*p.noseProjection),sy(.16),front(.55*p.noseProjection),sy(.02),front(.57*p.noseProjection),sy(-.07));
  ctx.bezierCurveTo(front(.42*p.noseProjection),sy(-.14),front(.24),sy(-.18),front(.22),sy(-.27));
  ctx.bezierCurveTo(front(.34*p.muzzleProjection),sy(-.32),front(.38*p.muzzleProjection),sy(-.37),front(.29*p.muzzleProjection),sy(-.43));
  ctx.bezierCurveTo(front(.22),sy(-.54),front(.26*p.chinProjection),sy(-.67),front(.25*p.chinProjection),sy(-.75));
  ctx.bezierCurveTo(front(.12),sy(-.88),front(.05),bottom-4,cx,bottom);
  ctx.stroke();

  ctx.strokeStyle='#75b9e9';ctx.lineWidth=1.2;
  ctx.beginPath();ctx.arc(front(.13),sy(.27),7,0,TAU);ctx.stroke();

  ctx.strokeStyle='#c58a92';
  ctx.beginPath();
  ctx.moveTo(front(.23*p.muzzleProjection),sy(-.34));
  ctx.quadraticCurveTo(front(.34*p.muzzleProjection),sy(-.35),front(.24*p.muzzleProjection),sy(-.38));
  ctx.stroke();

  ctx.fillStyle='#6a7789';ctx.font='8px ui-monospace,monospace';
  ctx.fillText('profile controls projection', 12, h-10);
}

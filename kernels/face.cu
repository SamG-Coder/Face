// Iterative clay head for CUDA WebShader.
// Geometry evolves through repeated blockout -> secondary -> feature -> polish
// passes. No pass solves the whole face in one shot.

__device__ float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

__device__ float absf_local(float v) {
    return v < 0.0f ? -v : v;
}

__device__ float smooth01(float t) {
    t = clampf(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ float segment(
    float y,
    float yTop,
    float yBottom,
    float valueTop,
    float valueBottom)
{
    float t = (yTop - y) / (yTop - yBottom);
    return lerpf(valueTop, valueBottom, smooth01(t));
}

__device__ float compactEllipse(
    float x,
    float y,
    float cx,
    float cy,
    float rx,
    float ry,
    float inner)
{
    float dx = (x - cx) / rx;
    float dy = (y - cy) / ry;
    float d = dx * dx + dy * dy;

    if (d >= 1.0f) return 0.0f;
    if (d <= inner) return 1.0f;

    return 1.0f - smooth01((d - inner) / (1.0f - inner));
}

__device__ float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// -----------------------------------------------------------------------------
// CONSTRUCTION GUIDES
// -----------------------------------------------------------------------------

__device__ float silhouetteHalfWidth(
    float y,
    float headWidth,
    float templeWidth,
    float cheekWidth,
    float jawWidth)
{
    float temple = templeWidth / 0.88f;
    float cheek = cheekWidth / 1.06f;
    float jaw = jawWidth / 0.72f;
    float scale;

    // Rounded crown first. The earlier cage left a broad flat cap.
    if (y > 0.88f) {
        scale = segment(y, 1.00f, 0.88f, 0.08f, 0.72f);
    } else if (y > 0.72f) {
        scale = segment(y, 0.88f, 0.72f, 0.72f, 0.94f);
    } else if (y > 0.30f) {
        scale = segment(y, 0.72f, 0.30f, 0.94f, 0.84f * temple);
    } else if (y > -0.08f) {
        scale = segment(y, 0.30f, -0.08f, 0.84f * temple, 0.90f * cheek);
    } else if (y > -0.52f) {
        scale = segment(y, -0.08f, -0.52f, 0.90f * cheek, 0.72f * jaw);
    } else if (y > -0.76f) {
        scale = segment(y, -0.52f, -0.76f, 0.72f * jaw, 0.42f);
    } else {
        scale = segment(y, -0.76f, -0.94f, 0.42f, 0.06f);
    }

    return headWidth * scale;
}

__device__ float neutralHalfWidth(float y, float headWidth) {
    return silhouetteHalfWidth(y, headWidth, 0.88f, 1.06f, 0.72f);
}

__device__ float baseProfileDepth(
    float y,
    float headDepth,
    float chinProjection,
    float muzzleProjection)
{
    float depthScale = headDepth / 0.90f;
    float z;

    if (y > 0.72f) {
        z = segment(y, 1.00f, 0.72f, 0.06f, 0.29f);
    } else if (y > 0.18f) {
        z = segment(y, 0.72f, 0.18f, 0.29f, 0.35f);
    } else if (y > 0.08f) {
        z = segment(y, 0.18f, 0.08f, 0.35f, 0.30f);
    } else if (y > -0.32f) {
        z = segment(y, 0.08f, -0.32f, 0.30f, 0.29f);
    } else if (y > -0.49f) {
        z = segment(
            y, -0.32f, -0.49f,
            0.29f, 0.32f * muzzleProjection);
    } else if (y > -0.60f) {
        z = segment(
            y, -0.49f, -0.60f,
            0.32f * muzzleProjection, 0.29f);
    } else if (y > -0.72f) {
        z = segment(
            y, -0.60f, -0.72f,
            0.27f, 0.34f * chinProjection);
    } else {
        z = segment(
            y, -0.72f, -0.94f,
            0.34f * chinProjection, 0.06f);
    }

    return z * depthScale;
}

__device__ float neutralProfileDepth(float y, float headDepth) {
    return baseProfileDepth(y, headDepth, 1.0f, 1.0f);
}

__device__ float sidePlaneDepth(float y, float headDepth) {
    float depthScale = headDepth / 0.90f;

    if (y > 0.58f) {
        return segment(y, 1.00f, 0.58f, -0.02f, -0.075f) * depthScale;
    }

    if (y > -0.46f) {
        return -0.075f * depthScale;
    }

    return segment(y, -0.46f, -0.94f, -0.075f, -0.015f) * depthScale;
}

__device__ float backCraniumDepth(float y, float headDepth) {
    float depthScale = headDepth / 0.90f;
    float d;

    if (y > 0.86f) {
        d = segment(y, 1.00f, 0.86f, 0.08f, 0.54f);
    } else if (y > 0.62f) {
        d = segment(y, 0.86f, 0.62f, 0.54f, 0.77f);
    } else if (y > 0.18f) {
        d = segment(y, 0.62f, 0.18f, 0.77f, 0.82f);
    } else if (y > -0.42f) {
        d = segment(y, 0.18f, -0.42f, 0.82f, 0.65f);
    } else if (y > -0.72f) {
        d = segment(y, -0.42f, -0.72f, 0.65f, 0.37f);
    } else {
        d = segment(y, -0.72f, -0.94f, 0.37f, 0.06f);
    }

    return d * depthScale;
}

__device__ float4 constructionPoint(
    float u,
    float v,
    float headWidth,
    float headDepth,
    float headHeight,
    float templeWidth,
    float cheekWidth,
    float jawWidth,
    float muzzleProjection,
    float chinProjection,
    int neutral)
{
    const float PI = 3.14159265358979323846f;

    float yN = lerpf(-0.94f, 1.00f, v);
    float y = yN * headHeight;

    float halfWidth = neutral != 0
        ? neutralHalfWidth(yN, headWidth)
        : silhouetteHalfWidth(
            yN, headWidth, templeWidth, cheekWidth, jawWidth);

    float theta = u * PI * 2.0f;
    float s = sinf(theta);
    float c = cosf(theta);
    float x = s * halfWidth;

    float sideZ = sidePlaneDepth(yN, headDepth);
    float z;

    if (c >= 0.0f) {
        // Neutral lump stays rounder. Primary sculpt later cuts the side planes.
        float xn = absf_local(x) / fmaxf(halfWidth, 0.0001f);
        float start = neutral != 0 ? 0.18f : 0.38f;
        float frontBlend = 1.0f;

        if (xn > start) {
            frontBlend = 1.0f - smooth01(
                (xn - start) / (1.0f - start));
        }

        float centerZ = neutral != 0
            ? neutralProfileDepth(yN, headDepth)
            : baseProfileDepth(
                yN, headDepth, chinProjection, muzzleProjection);

        z = lerpf(sideZ, centerZ, frontBlend);
    } else {
        float back = backCraniumDepth(yN, headDepth);
        z = sideZ - back * smooth01(-c);
    }

    return make_float4(x, y, z, 1.0f);
}

// -----------------------------------------------------------------------------
// SECONDARY FORMS
// -----------------------------------------------------------------------------

__device__ float secondaryDepth(
    float x,
    float y,
    float headWidth,
    float eyeSpacing,
    float eyeWidth,
    float socketDepth,
    float browProjection,
    float noseProjection,
    float noseWidth,
    float cheekProjection,
    float mouthWidth)
{
    float z = 0.0f;

    float eyeHalf = headWidth * 0.18f * (eyeWidth / 0.92f);
    float eyeCenter = eyeHalf * 2.0f * (eyeSpacing / 0.29f);
    float innerCorner = fmaxf(
        eyeCenter - eyeHalf,
        headWidth * 0.10f);

    float socketRx = eyeHalf * 1.28f;
    float socketRy = 0.125f;

    float socketL = compactEllipse(
        x, y, -eyeCenter, 0.065f,
        socketRx, socketRy, 0.16f);
    float socketR = compactEllipse(
        x, y, eyeCenter, 0.065f,
        socketRx, socketRy, 0.16f);

    // Broad carving, not a hard hole.
    z -= socketDepth * 0.31f * (socketL + socketR);

    float browL = compactEllipse(
        x, y, -eyeCenter, 0.185f,
        socketRx * 1.04f, 0.080f, 0.08f);
    float browR = compactEllipse(
        x, y, eyeCenter, 0.185f,
        socketRx * 1.04f, 0.080f, 0.08f);

    z += 0.023f * browProjection * (browL + browR);

    float cheekX = eyeCenter * 1.28f;
    float cheekL = compactEllipse(
        x, y, -cheekX, -0.105f,
        headWidth * 0.28f, 0.24f, 0.05f);
    float cheekR = compactEllipse(
        x, y, cheekX, -0.105f,
        headWidth * 0.28f, 0.24f, 0.05f);

    z += 0.024f * cheekProjection * (cheekL + cheekR);

    // First nose pass is only the large wedge.
    float wingHalf = innerCorner * noseWidth;

    if (y <= 0.09f && y >= -0.25f) {
        float t = clampf((0.09f - y) / 0.34f, 0.0f, 1.0f);
        float topHalf = lerpf(
            wingHalf * 0.30f,
            wingHalf * 0.44f,
            t);
        float sideHalf = lerpf(
            wingHalf * 0.66f,
            wingHalf * 0.84f,
            t);
        float outerHalf = lerpf(
            wingHalf * 0.82f,
            wingHalf * 1.02f,
            t);

        float ax = absf_local(x);
        float across = 0.0f;

        if (ax <= topHalf) {
            across = 1.0f;
        } else if (ax <= sideHalf) {
            float q = (ax - topHalf) /
                fmaxf(sideHalf - topHalf, 0.0001f);
            across = lerpf(1.0f, 0.45f, smooth01(q));
        } else if (ax <= outerHalf) {
            float q = (ax - sideHalf) /
                fmaxf(outerHalf - sideHalf, 0.0001f);
            across = 0.45f * (1.0f - smooth01(q));
        }

        z += across * noseProjection *
            lerpf(0.008f, 0.092f, smooth01(t));
    }

    // Broad tooth-cylinder support only.
    float mouthHalf = eyeCenter * mouthWidth;
    float muzzle = compactEllipse(
        x, y, 0.0f, -0.465f,
        fmaxf(mouthHalf * 1.18f, 0.20f),
        0.185f, 0.04f);

    z += 0.020f * muzzle;

    return z;
}

// -----------------------------------------------------------------------------
// TERTIARY FORMS
// -----------------------------------------------------------------------------

__device__ float tertiaryDepth(
    float x,
    float y,
    float headWidth,
    float eyeSpacing,
    float eyeWidth,
    float browProjection,
    float noseProjection,
    float noseWidth,
    float mouthWidth,
    float upperLip,
    float lowerLip)
{
    float z = 0.0f;

    float eyeHalf = headWidth * 0.18f * (eyeWidth / 0.92f);
    float eyeCenter = eyeHalf * 2.0f * (eyeSpacing / 0.29f);
    float innerCorner = fmaxf(
        eyeCenter - eyeHalf,
        headWidth * 0.10f);
    float wingHalf = innerCorner * noseWidth;
    float mouthHalf = eyeCenter * mouthWidth;

    // Subtle orbital fullness so sockets stop reading as empty pits.
    float eyeBallRx = eyeHalf * 0.76f;
    float eyeBallRy = 0.058f;
    float eyeL = compactEllipse(
        x, y, -eyeCenter, 0.050f,
        eyeBallRx, eyeBallRy, 0.12f);
    float eyeR = compactEllipse(
        x, y, eyeCenter, 0.050f,
        eyeBallRx, eyeBallRy, 0.12f);
    z += 0.016f * (eyeL + eyeR);

    float glabella = compactEllipse(
        x, y, 0.0f, 0.150f,
        innerCorner * 0.58f, 0.070f, 0.05f);
    float nasion = compactEllipse(
        x, y, 0.0f, 0.083f,
        innerCorner * 0.50f, 0.052f, 0.05f);

    z += 0.010f * browProjection * glabella;
    z -= 0.012f * nasion;

    float tip = compactEllipse(
        x, y, 0.0f, -0.278f,
        fmaxf(wingHalf * 0.66f, 0.052f),
        0.062f, 0.08f);
    z += noseProjection * 0.054f * tip;

    float alaX = wingHalf * 0.68f;
    float alaRx = fmaxf(wingHalf * 0.42f, 0.040f);
    float alaL = compactEllipse(
        x, y, -alaX, -0.315f,
        alaRx, 0.050f, 0.04f);
    float alaR = compactEllipse(
        x, y, alaX, -0.315f,
        alaRx, 0.050f, 0.04f);
    z += noseProjection * 0.018f * (alaL + alaR);

    float philtrum = compactEllipse(
        x, y, 0.0f, -0.405f,
        fmaxf(wingHalf * 0.30f, 0.036f),
        0.050f, 0.04f);
    z -= 0.005f * philtrum;

    float upperCenter = compactEllipse(
        x, y, 0.0f, -0.485f,
        fmaxf(mouthHalf * 0.24f, 0.044f),
        0.032f, 0.04f);
    float upperL = compactEllipse(
        x, y, -mouthHalf * 0.46f, -0.490f,
        fmaxf(mouthHalf * 0.39f, 0.052f),
        0.034f, 0.04f);
    float upperR = compactEllipse(
        x, y, mouthHalf * 0.46f, -0.490f,
        fmaxf(mouthHalf * 0.39f, 0.052f),
        0.034f, 0.04f);

    z += upperLip * 0.008f *
        (1.05f * upperCenter + 0.66f * (upperL + upperR));

    float lowerL = compactEllipse(
        x, y, -mouthHalf * 0.20f, -0.548f,
        fmaxf(mouthHalf * 0.45f, 0.060f),
        0.040f, 0.04f);
    float lowerR = compactEllipse(
        x, y, mouthHalf * 0.20f, -0.548f,
        fmaxf(mouthHalf * 0.45f, 0.060f),
        0.040f, 0.04f);

    z += lowerLip * 0.010f * (lowerL + lowerR);

    // Very shallow curved mouth seam.
    float ax = absf_local(x);
    if (ax < mouthHalf) {
        float xt = ax / fmaxf(mouthHalf, 0.0001f);
        float seamY = -0.515f + 0.010f * xt * xt;
        float dy = absf_local(y - seamY) / 0.016f;
        float seam = (1.0f - smooth01(dy)) *
            (1.0f - smooth01(xt));
        z -= 0.0030f * seam;
    }

    // Labiomental break.
    float groove = compactEllipse(
        x, y, 0.0f, -0.615f,
        headWidth * 0.25f, 0.058f, 0.03f);
    z -= 0.010f * groove;

    return z;
}

__device__ float4 targetPoint(
    float u,
    float v,
    int stage,
    float headWidth,
    float headDepth,
    float headHeight,
    float templeWidth,
    float cheekWidth,
    float jawWidth,
    float eyeSpacing,
    float eyeWidth,
    float socketDepth,
    float browProjection,
    float noseProjection,
    float noseWidth,
    float cheekProjection,
    float muzzleProjection,
    float mouthWidth,
    float upperLip,
    float lowerLip,
    float chinProjection)
{
    float4 p = constructionPoint(
        u, v,
        headWidth, headDepth, headHeight,
        templeWidth, cheekWidth, jawWidth,
        muzzleProjection, chinProjection,
        0);

    if (stage <= 0) {
        return p;
    }

    const float PI = 3.14159265358979323846f;
    float yN = lerpf(-0.94f, 1.00f, v);
    float theta = u * PI * 2.0f;
    float c = cosf(theta);

    // Features only affect the actual front mask.
    float facing = smooth01(c / 0.34f);

    p.z += facing * secondaryDepth(
        p.x, yN,
        headWidth,
        eyeSpacing,
        eyeWidth,
        socketDepth,
        browProjection,
        noseProjection,
        noseWidth,
        cheekProjection,
        mouthWidth);

    if (stage >= 2) {
        p.z += facing * tertiaryDepth(
            p.x, yN,
            headWidth,
            eyeSpacing,
            eyeWidth,
            browProjection,
            noseProjection,
            noseWidth,
            mouthWidth,
            upperLip,
            lowerLip);
    }

    return p;
}

// -----------------------------------------------------------------------------
// GPU PASSES
// -----------------------------------------------------------------------------

__global__ void GenerateBaseHead(
    float4* positions,
    unsigned int rings,
    unsigned int levels,
    float headWidth,
    float headDepth,
    float headHeight)
{
    unsigned int id =
        blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int count = rings * levels;

    if (id >= count) return;

    unsigned int ix = id % rings;
    unsigned int iy = id / rings;

    float u = (float)ix / (float)rings;
    float v = (float)iy / (float)(levels - 1);

    positions[id] = constructionPoint(
        u, v,
        headWidth, headDepth, headHeight,
        0.88f, 1.06f, 0.72f,
        1.0f, 1.0f,
        1);
}

__global__ void SculptStage(
    const float4* source,
    float4* destination,
    unsigned int rings,
    unsigned int levels,
    int stage,
    float strength,
    float headWidth,
    float headDepth,
    float headHeight,
    float templeWidth,
    float cheekWidth,
    float jawWidth,
    float eyeSpacing,
    float eyeWidth,
    float socketDepth,
    float browProjection,
    float noseProjection,
    float noseWidth,
    float cheekProjection,
    float muzzleProjection,
    float mouthWidth,
    float upperLip,
    float lowerLip,
    float chinProjection)
{
    unsigned int id =
        blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int count = rings * levels;

    if (id >= count) return;

    unsigned int ix = id % rings;
    unsigned int iy = id / rings;

    float u = (float)ix / (float)rings;
    float v = (float)iy / (float)(levels - 1);

    float4 current = source[id];
    float4 target = targetPoint(
        u, v, stage,
        headWidth, headDepth, headHeight,
        templeWidth, cheekWidth, jawWidth,
        eyeSpacing, eyeWidth, socketDepth,
        browProjection, noseProjection, noseWidth,
        cheekProjection, muzzleProjection,
        mouthWidth, upperLip, lowerLip,
        chinProjection);

    float s = clampf(strength, 0.0f, 1.0f);

    destination[id] = make_float4(
        lerpf(current.x, target.x, s),
        lerpf(current.y, target.y, s * 0.18f),
        lerpf(current.z, target.z, s),
        1.0f);
}

__global__ void SmoothClay(
    const float4* source,
    float4* destination,
    unsigned int rings,
    unsigned int levels,
    float strength)
{
    unsigned int id =
        blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int count = rings * levels;

    if (id >= count) return;

    unsigned int ix = id % rings;
    unsigned int iy = id / rings;

    unsigned int leftX =
        ix == 0 ? rings - 1 : ix - 1;
    unsigned int rightX =
        ix + 1 == rings ? 0 : ix + 1;

    unsigned int left = iy * rings + leftX;
    unsigned int right = iy * rings + rightX;
    unsigned int down =
        iy == 0 ? id : (iy - 1) * rings + ix;
    unsigned int up =
        iy + 1 == levels ? id : (iy + 1) * rings + ix;

    float4 p = source[id];
    float4 a = source[left];
    float4 b = source[right];
    float4 c = source[down];
    float4 d = source[up];

    float avgX = (a.x + b.x + c.x + d.x) * 0.25f;
    float avgZ = (a.z + b.z + c.z + d.z) * 0.25f;

    // Preserve drawing landmark heights; blend mostly in depth, with a smaller
    // silhouette relaxation in x.
    float s = clampf(strength, 0.0f, 0.45f);

    destination[id] = make_float4(
        lerpf(p.x, avgX, s * 0.28f),
        p.y,
        lerpf(p.z, avgZ, s),
        1.0f);
}

__global__ void ComputeNormals(
    const float4* positions,
    float4* normals,
    unsigned int rings,
    unsigned int levels)
{
    unsigned int id =
        blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int count = rings * levels;

    if (id >= count) return;

    unsigned int ix = id % rings;
    unsigned int iy = id / rings;

    unsigned int leftX =
        ix == 0 ? rings - 1 : ix - 1;
    unsigned int rightX =
        ix + 1 == rings ? 0 : ix + 1;

    unsigned int left = iy * rings + leftX;
    unsigned int right = iy * rings + rightX;
    unsigned int down =
        iy == 0 ? id : (iy - 1) * rings + ix;
    unsigned int up =
        iy + 1 == levels ? id : (iy + 1) * rings + ix;

    float4 l = positions[left];
    float4 r = positions[right];
    float4 d = positions[down];
    float4 u = positions[up];

    float3 tangentU = make_float3(
        r.x - l.x,
        r.y - l.y,
        r.z - l.z);
    float3 tangentV = make_float3(
        u.x - d.x,
        u.y - d.y,
        u.z - d.z);

    float3 n = normalize(cross3(tangentU, tangentV));

    normals[id] = make_float4(
        n.x, n.y, n.z, 0.0f);
}

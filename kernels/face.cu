// Procedural human head construction for CUDA WebShader.
//
// IMPORTANT: the head is NOT a deformed UV sphere.
// Each horizontal row is built from two drawing constraints:
//   - front-view silhouette width
//   - side-view/profile depth
// The ring between them is then constructed as a cross-section.
//
// That makes the same front/profile construction used by portrait artists the
// actual geometry generator rather than a diagram beside an unrelated sphere.

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
// FRONT DRAWING: outer silhouette
// -----------------------------------------------------------------------------

__device__ float silhouetteHalfWidth(
    float y,
    float headWidth,
    float templeWidth,
    float cheekWidth,
    float jawWidth)
{
    // Controls are normalized around the project's neutral/default face.
    float temple = templeWidth / 0.88f;
    float cheek = cheekWidth / 1.06f;
    float jaw = jawWidth / 0.72f;

    float scale;

    if (y > 0.78f) {
        // crown -> widest upper cranium
        scale = segment(y, 1.00f, 0.78f, 0.30f, 0.93f);
    } else if (y > 0.30f) {
        // upper cranium -> temple
        scale = segment(y, 0.78f, 0.30f, 0.93f, 0.84f * temple);
    } else if (y > -0.08f) {
        // temple -> zygomatic width
        scale = segment(y, 0.30f, -0.08f, 0.84f * temple, 0.90f * cheek);
    } else if (y > -0.52f) {
        // cheek -> mandibular body
        scale = segment(y, -0.08f, -0.52f, 0.90f * cheek, 0.72f * jaw);
    } else if (y > -0.76f) {
        // jaw -> chin
        scale = segment(y, -0.52f, -0.76f, 0.72f * jaw, 0.40f);
    } else {
        // close below the chin; this is under the visible face, not a neck nub
        scale = segment(y, -0.76f, -0.94f, 0.40f, 0.16f);
    }

    return headWidth * scale;
}

// -----------------------------------------------------------------------------
// PROFILE DRAWING: large front-face plane and back cranium
// -----------------------------------------------------------------------------

__device__ float baseProfileDepth(
    float y,
    float headDepth,
    float chinProjection,
    float muzzleProjection)
{
    float depthScale = headDepth / 0.90f;
    float chin = chinProjection / 1.00f;
    float muzzle = muzzleProjection / 1.00f;

    float z;

    if (y > 0.68f) {
        z = segment(y, 1.00f, 0.68f, 0.10f, 0.31f);
    } else if (y > 0.18f) {
        // forehead gently recedes toward brow
        z = segment(y, 0.68f, 0.18f, 0.31f, 0.36f);
    } else if (y > 0.08f) {
        // glabella -> nasion break
        z = segment(y, 0.18f, 0.08f, 0.36f, 0.30f);
    } else if (y > -0.32f) {
        // maxilla plane behind the separate nose wedge
        z = segment(y, 0.08f, -0.32f, 0.30f, 0.29f);
    } else if (y > -0.49f) {
        // subnasal -> upper denture mass
        z = segment(y, -0.32f, -0.49f, 0.29f, 0.33f * muzzle);
    } else if (y > -0.60f) {
        z = segment(y, -0.49f, -0.60f, 0.33f * muzzle, 0.30f);
    } else if (y > -0.72f) {
        // labiomental recess -> chin
        z = segment(y, -0.60f, -0.72f, 0.27f, 0.36f * chin);
    } else {
        z = segment(y, -0.72f, -0.94f, 0.36f * chin, 0.10f);
    }

    return z * depthScale;
}

__device__ float sidePlaneDepth(float y, float headDepth) {
    float depthScale = headDepth / 0.90f;

    if (y > 0.55f) {
        return segment(y, 1.00f, 0.55f, -0.04f, -0.08f) * depthScale;
    }
    if (y > -0.45f) {
        return -0.08f * depthScale;
    }
    return segment(y, -0.45f, -0.94f, -0.08f, -0.02f) * depthScale;
}

__device__ float backCraniumDepth(float y, float headDepth) {
    float depthScale = headDepth / 0.90f;
    float d;

    if (y > 0.76f) {
        d = segment(y, 1.00f, 0.76f, 0.30f, 0.72f);
    } else if (y > 0.18f) {
        d = segment(y, 0.76f, 0.18f, 0.72f, 0.83f);
    } else if (y > -0.42f) {
        d = segment(y, 0.18f, -0.42f, 0.83f, 0.67f);
    } else if (y > -0.72f) {
        d = segment(y, -0.42f, -0.72f, 0.67f, 0.40f);
    } else {
        d = segment(y, -0.72f, -0.94f, 0.40f, 0.13f);
    }

    return d * depthScale;
}

// -----------------------------------------------------------------------------
// FRONT-FACE FORMS: still planar/constructional, but placed on the cage above.
// -----------------------------------------------------------------------------

__device__ float frontFeatureDepth(
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
    float mouthWidth,
    float upperLip,
    float lowerLip)
{
    float z = 0.0f;

    // Five-eye scaffold: default eye full width ~1/5 of the front head width.
    float eyeSizeScale = eyeWidth / 0.92f;
    float eyeHalf = headWidth * 0.18f * eyeSizeScale;
    float eyeCenter = eyeHalf * 2.0f * (eyeSpacing / 0.29f);
    float innerCorner = fmaxf(eyeCenter - eyeHalf, headWidth * 0.10f);

    // Eye orbit: a broad shallow cavity.
    float socketRx = eyeHalf * 1.22f;
    float socketRy = 0.115f;
    float leftSocket = compactEllipse(
        x, y, -eyeCenter, 0.07f,
        socketRx, socketRy, 0.20f);
    float rightSocket = compactEllipse(
        x, y, eyeCenter, 0.07f,
        socketRx, socketRy, 0.20f);

    z -= socketDepth * 0.48f * (leftSocket + rightSocket);

    // Eyeball mass implied inside the socket so the first blockout does not
    // read as two empty holes. Explicit eyeball geometry comes next.
    float eyeBallRx = eyeHalf * 0.78f;
    float eyeBallRy = 0.062f;
    float leftBall = compactEllipse(
        x, y, -eyeCenter, 0.055f,
        eyeBallRx, eyeBallRy, 0.16f);
    float rightBall = compactEllipse(
        x, y, eyeCenter, 0.055f,
        eyeBallRx, eyeBallRy, 0.16f);

    z += 0.022f * (leftBall + rightBall);

    // Brow roofs sit above the sockets. They are not connected into one
    // giant central bridge.
    float browRx = socketRx * 1.03f;
    float browRy = 0.075f;
    float leftBrow = compactEllipse(
        x, y, -eyeCenter, 0.18f,
        browRx, browRy, 0.12f);
    float rightBrow = compactEllipse(
        x, y, eyeCenter, 0.18f,
        browRx, browRy, 0.12f);

    z += 0.027f * browProjection * (leftBrow + rightBrow);

    // Small glabella / keystone and explicit nasion recess.
    float glabella = compactEllipse(
        x, y, 0.0f, 0.155f,
        innerCorner * 0.55f, 0.070f, 0.10f);
    float nasion = compactEllipse(
        x, y, 0.0f, 0.085f,
        innerCorner * 0.48f, 0.055f, 0.10f);

    z += 0.014f * browProjection * glabella;
    z -= 0.020f * nasion;

    // Zygomatic plane lies outside/below the orbit.
    float cheekX = eyeCenter * 1.30f;
    float cheekL = compactEllipse(
        x, y, -cheekX, -0.10f,
        headWidth * 0.25f, 0.20f, 0.10f);
    float cheekR = compactEllipse(
        x, y, cheekX, -0.10f,
        headWidth * 0.25f, 0.20f, 0.10f);

    z += 0.030f * cheekProjection * (cheekL + cheekR);

    // -----------------------------------------------------------------
    // Nose wedge: profile depth is generated here, not by stretching the
    // whole center of the head. Width is derived from inner eye corners.
    // -----------------------------------------------------------------
    float wingHalf = innerCorner * noseWidth;

    if (y <= 0.09f && y >= -0.245f) {
        float t = clampf((0.09f - y) / 0.335f, 0.0f, 1.0f);

        float topHalf = lerpf(
            wingHalf * 0.27f,
            wingHalf * 0.43f,
            t);
        float sideHalf = lerpf(
            wingHalf * 0.62f,
            wingHalf * 0.80f,
            t);
        float outerHalf = lerpf(
            wingHalf * 0.76f,
            wingHalf * 0.92f,
            t);

        float ax = absf_local(x);
        float across = 0.0f;

        if (ax <= topHalf) {
            across = 1.0f;
        } else if (ax <= sideHalf) {
            float s = (ax - topHalf) / fmaxf(sideHalf - topHalf, 0.0001f);
            across = lerpf(1.0f, 0.46f, smooth01(s));
        } else if (ax <= outerHalf) {
            float s = (ax - sideHalf) / fmaxf(outerHalf - sideHalf, 0.0001f);
            across = 0.46f * (1.0f - smooth01(s));
        }

        float projection = noseProjection *
            lerpf(0.012f, 0.115f, smooth01(t));
        z += across * projection;
    }

    // Ball and wings are separate from the wedge.
    float tip = compactEllipse(
        x, y, 0.0f, -0.275f,
        fmaxf(wingHalf * 0.62f, 0.050f),
        0.060f, 0.12f);
    z += noseProjection * 0.078f * tip;

    float alaX = wingHalf * 0.67f;
    float alaRx = fmaxf(wingHalf * 0.40f, 0.040f);
    float alaL = compactEllipse(
        x, y, -alaX, -0.315f,
        alaRx, 0.047f, 0.08f);
    float alaR = compactEllipse(
        x, y, alaX, -0.315f,
        alaRx, 0.047f, 0.08f);
    z += noseProjection * 0.024f * (alaL + alaR);

    // -----------------------------------------------------------------
    // Tooth cylinder / muzzle first, then subtle lip volumes.
    // -----------------------------------------------------------------
    float mouthHalf = eyeCenter * mouthWidth;

    float muzzle = compactEllipse(
        x, y, 0.0f, -0.465f,
        fmaxf(mouthHalf * 1.10f, 0.18f),
        0.165f, 0.12f);
    z += 0.026f * muzzle;

    float philtrum = compactEllipse(
        x, y, 0.0f, -0.405f,
        fmaxf(wingHalf * 0.28f, 0.034f),
        0.050f, 0.10f);
    z -= 0.007f * philtrum;

    float upperCenter = compactEllipse(
        x, y, 0.0f, -0.485f,
        fmaxf(mouthHalf * 0.24f, 0.042f),
        0.030f, 0.10f);
    float upperL = compactEllipse(
        x, y, -mouthHalf * 0.48f, -0.490f,
        fmaxf(mouthHalf * 0.38f, 0.052f),
        0.033f, 0.08f);
    float upperR = compactEllipse(
        x, y, mouthHalf * 0.48f, -0.490f,
        fmaxf(mouthHalf * 0.38f, 0.052f),
        0.033f, 0.08f);

    z += upperLip * 0.010f *
        (1.10f * upperCenter + 0.70f * (upperL + upperR));

    float lowerL = compactEllipse(
        x, y, -mouthHalf * 0.21f, -0.548f,
        fmaxf(mouthHalf * 0.44f, 0.060f),
        0.038f, 0.08f);
    float lowerR = compactEllipse(
        x, y, mouthHalf * 0.21f, -0.548f,
        fmaxf(mouthHalf * 0.44f, 0.060f),
        0.038f, 0.08f);

    z += lowerLip * 0.014f * (lowerL + lowerR);

    // Curved mouth seam around the tooth cylinder. Very shallow at blockout
    // level; no giant black horizontal trench.
    float axMouth = absf_local(x);
    if (axMouth < mouthHalf) {
        float xt = axMouth / fmaxf(mouthHalf, 0.0001f);
        float seamY = -0.515f + 0.012f * xt * xt;
        float dy = absf_local(y - seamY) / 0.015f;
        float seam = (1.0f - smooth01(dy)) * (1.0f - smooth01(xt));
        z -= 0.0045f * seam;
    }

    return z;
}

__device__ float4 facePoint(
    float u,
    float v,
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
    const float PI = 3.14159265358979323846f;

    // Linear rows: drawing landmarks stay where they were placed. This is the
    // key change from the old spherical latitude mapping.
    float yN = lerpf(-0.94f, 1.00f, v);
    float y = yN * headHeight;

    float halfWidth = silhouetteHalfWidth(
        yN,
        headWidth,
        templeWidth,
        cheekWidth,
        jawWidth);

    float theta = u * PI * 2.0f;
    float s = sinf(theta);
    float c = cosf(theta);

    float x = s * halfWidth;
    float sideZ = sidePlaneDepth(yN, headDepth);

    float z;

    if (c >= 0.0f) {
        // Front mask: a broad center plane with explicit side-plane falloff.
        // This is the 3D equivalent of slicing the sides off the helmethead.
        float xn = absf_local(x) / fmaxf(halfWidth, 0.0001f);
        float frontPlane = 1.0f;

        if (xn > 0.52f) {
            frontPlane = 1.0f - smooth01((xn - 0.52f) / 0.48f);
        }

        float centerZ = baseProfileDepth(
            yN,
            headDepth,
            chinProjection,
            muzzleProjection);

        z = lerpf(sideZ, centerZ, frontPlane);

        float featureFacing = smooth01(c / 0.28f);
        z += featureFacing * frontFeatureDepth(
            x,
            yN,
            headWidth,
            eyeSpacing,
            eyeWidth,
            socketDepth,
            browProjection,
            noseProjection,
            noseWidth,
            cheekProjection,
            mouthWidth,
            upperLip,
            lowerLip);
    } else {
        // Back of cranium is its own smooth cross-section. It does not inherit
        // any face features or the old sphere's latitude pinching.
        float back = backCraniumDepth(yN, headDepth);
        float backT = smooth01((-c));
        z = sideZ - back * backT;
    }

    return make_float4(x, y, z, 1.0f);
}

__global__ void GenerateFace(
    float4* positions,
    float4* normals,
    unsigned int rings,
    unsigned int levels,
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
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int count = rings * levels;
    if (id >= count) return;

    unsigned int ix = id % rings;
    unsigned int iy = id / rings;

    float u = (float)ix / (float)rings;
    float v = (float)iy / (float)(levels - 1);

    float4 p = facePoint(
        u, v,
        headWidth, headDepth, headHeight,
        templeWidth, cheekWidth, jawWidth,
        eyeSpacing, eyeWidth, socketDepth,
        browProjection, noseProjection, noseWidth,
        cheekProjection, muzzleProjection,
        mouthWidth, upperLip, lowerLip, chinProjection);

    positions[id] = p;

    // Finite-difference normal from the exact same construction surface.
    float du = 0.0022f;
    float dv = 0.0022f;
    float v0 = clampf(v - dv, 0.0f, 1.0f);
    float v1 = clampf(v + dv, 0.0f, 1.0f);

    float4 pu0 = facePoint(
        u - du, v,
        headWidth, headDepth, headHeight,
        templeWidth, cheekWidth, jawWidth,
        eyeSpacing, eyeWidth, socketDepth,
        browProjection, noseProjection, noseWidth,
        cheekProjection, muzzleProjection,
        mouthWidth, upperLip, lowerLip, chinProjection);
    float4 pu1 = facePoint(
        u + du, v,
        headWidth, headDepth, headHeight,
        templeWidth, cheekWidth, jawWidth,
        eyeSpacing, eyeWidth, socketDepth,
        browProjection, noseProjection, noseWidth,
        cheekProjection, muzzleProjection,
        mouthWidth, upperLip, lowerLip, chinProjection);
    float4 pv0 = facePoint(
        u, v0,
        headWidth, headDepth, headHeight,
        templeWidth, cheekWidth, jawWidth,
        eyeSpacing, eyeWidth, socketDepth,
        browProjection, noseProjection, noseWidth,
        cheekProjection, muzzleProjection,
        mouthWidth, upperLip, lowerLip, chinProjection);
    float4 pv1 = facePoint(
        u, v1,
        headWidth, headDepth, headHeight,
        templeWidth, cheekWidth, jawWidth,
        eyeSpacing, eyeWidth, socketDepth,
        browProjection, noseProjection, noseWidth,
        cheekProjection, muzzleProjection,
        mouthWidth, upperLip, lowerLip, chinProjection);

    float3 tangentU = make_float3(
        pu1.x - pu0.x,
        pu1.y - pu0.y,
        pu1.z - pu0.z);
    float3 tangentV = make_float3(
        pv1.x - pv0.x,
        pv1.y - pv0.y,
        pv1.z - pv0.z);

    float3 n = normalize(cross3(tangentU, tangentV));
    normals[id] = make_float4(n.x, n.y, n.z, 0.0f);
}

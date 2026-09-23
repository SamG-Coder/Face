// Procedural human head construction for CUDA WebShader.
//
// Construction is based on portrait-drawing / sculpting practice:
//   1. establish cranium + cut side planes
//   2. place the face by large proportional landmarks
//   3. carve the sockets and brow before adding features
//   4. construct the nose as a wedge with top/side/bottom relationships
//   5. place the mouth on the denture/tooth-cylinder mass
//   6. add smaller soft-tissue forms only after the blockout reads correctly
//
// No model data is loaded. Each invocation generates one vertex and normal.

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

// Compact, controllable form field. Unlike a Gaussian it reaches exactly zero
// outside the specified ellipse, which makes it much easier to preserve large
// head planes instead of melting every feature into the surface.
__device__ float compactEllipse(
    float x, float y,
    float cx, float cy,
    float rx, float ry,
    float inner)
{
    float dx = (x - cx) / rx;
    float dy = (y - cy) / ry;
    float d = dx * dx + dy * dy;
    if (d >= 1.0f) return 0.0f;
    if (d <= inner) return 1.0f;
    return 1.0f - smooth01((d - inner) / (1.0f - inner));
}

__device__ float band(float value, float center, float radius) {
    float d = absf_local(value - center) / radius;
    return 1.0f - smooth01(d);
}

__device__ float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
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

    // Construction landmarks in normalized crown-to-underjaw coordinates.
    // The visible chin is above the bottom mesh pole so the mesh can close
    // beneath the jaw rather than collapsing the chin itself to a point.
    const float HAIRLINE = 0.68f;
    const float BROW = 0.18f;
    const float EYE = 0.07f;
    const float NOSE_BASE = -0.32f;
    const float MOUTH = -0.49f;
    const float CHIN = -0.82f;

    float theta = u * PI * 2.0f;
    float phi = (v - 0.5f) * PI;

    float st = sinf(theta);
    float ct = cosf(theta);
    float sp = sinf(phi);
    float cp = fmaxf(cosf(phi), 0.0f);

    float yN = sp;
    float y = yN * headHeight;

    // ---------------------------------------------------------------------
    // 1) HELMET / EGG BLOCKOUT
    // ---------------------------------------------------------------------
    // Start from a cranium, then shape the silhouette by construction zones.
    // This is intentionally global (not only "front-facing" vertices) so the
    // temple, zygomatic, jaw and chin actually change the silhouette.
    float widthScale = 1.0f;

    float templeBand = band(yN, BROW + 0.11f, 0.28f);
    widthScale *= 1.0f - templeBand * (1.0f - templeWidth) * 0.72f;

    float cheekBand = band(yN, -0.08f, 0.30f);
    widthScale *= 1.0f + cheekBand * (cheekWidth - 1.0f) * 0.72f;

    float jawBand = band(yN, -0.62f, 0.23f);
    widthScale *= 1.0f - jawBand * (1.0f - jawWidth) * 0.95f;

    float chinBand = band(yN, CHIN, 0.15f);
    widthScale *= 1.0f - chinBand * 0.43f;

    float x = st * cp * headWidth * widthScale;
    float z = ct * cp * headDepth;

    // Front-facing influence. The side-plane cut remains visible because the
    // silhouette shaping above is not multiplied by this value.
    float front = smooth01((ct - 0.02f) / 0.98f);

    // Forehead is a plane that gently recedes toward the hairline/crown,
    // rather than another rounded bump pasted onto the ellipsoid.
    float foreheadT = clampf((yN - BROW) / (HAIRLINE - BROW), 0.0f, 1.0f);
    z -= front * foreheadT * 0.035f;

    // ---------------------------------------------------------------------
    // 2) PROPORTIONAL LANDMARKS
    // ---------------------------------------------------------------------
    // A useful drawing default is roughly five eye-widths across, with about
    // one eye-width between the eyes. eyeSpacing remains an artistic control,
    // normalized around its original default value of 0.29.
    float eyeHalf = headWidth * 0.185f * eyeWidth;
    float eyeSpacingScale = eyeSpacing / 0.29f;
    float eyeCenterX = eyeHalf * 2.0f * eyeSpacingScale;
    float eyeY = headHeight * EYE;

    // Inner eye corners are also the starting proportional reference for the
    // wings of the nose. This relationship is then allowed to vary.
    float innerCornerX = eyeCenterX - eyeHalf;
    innerCornerX = fmaxf(innerCornerX, 0.065f);
    float noseWingHalf = innerCornerX * noseWidth;

    // Mouth corners are initially related to the pupils / eye centers, then
    // varied by the mouthWidth control.
    float mouthHalf = eyeCenterX * mouthWidth;

    // ---------------------------------------------------------------------
    // 3) SOCKETS, BROW, GLABELLA, CHEEKBONES
    // ---------------------------------------------------------------------
    // The eye socket is the large shape. The eye itself belongs inside this
    // cavity later; it is not represented by a painted almond-shaped dent.
    float socketRx = eyeHalf * 1.18f;
    float socketRy = headHeight * 0.115f;

    float leftSocket = compactEllipse(
        x, y, -eyeCenterX, eyeY,
        socketRx, socketRy, 0.24f);
    float rightSocket = compactEllipse(
        x, y, eyeCenterX, eyeY,
        socketRx, socketRy, 0.24f);

    z -= front * socketDepth * 0.78f * (leftSocket + rightSocket);

    // Brow blocks sit over the sockets like an awning. Keep them separate
    // from the nose root so the brow does not turn into one continuous pillar.
    float browY = headHeight * BROW;
    float browRy = headHeight * 0.075f;
    float leftBrow = compactEllipse(
        x, y, -eyeCenterX, browY,
        socketRx * 1.06f, browRy, 0.18f);
    float rightBrow = compactEllipse(
        x, y, eyeCenterX, browY,
        socketRx * 1.06f, browRy, 0.18f);

    z += front * 0.040f * browProjection * (leftBrow + rightBrow);

    // Glabella / keystone: a small independent form between the brows.
    float glabella = compactEllipse(
        x, y, 0.0f, headHeight * 0.16f,
        fmaxf(noseWingHalf * 0.78f, 0.055f),
        headHeight * 0.075f, 0.16f);
    z += front * 0.018f * browProjection * glabella;

    // Nasion/saddle recess below the glabella. This is the explicit break
    // that prevents the forehead from becoming a giant nose bridge.
    float nasion = compactEllipse(
        x, y, 0.0f, headHeight * 0.095f,
        fmaxf(noseWingHalf * 0.70f, 0.050f),
        headHeight * 0.065f, 0.12f);
    z -= front * 0.026f * nasion;

    // Zygomatic / cheek plane. It belongs outside and below the socket.
    float cheekX = eyeCenterX * 1.28f;
    float cheekY = headHeight * -0.08f;
    float cheekL = compactEllipse(
        x, y, -cheekX, cheekY,
        headWidth * 0.25f, headHeight * 0.24f, 0.20f);
    float cheekR = compactEllipse(
        x, y, cheekX, cheekY,
        headWidth * 0.25f, headHeight * 0.24f, 0.20f);

    z += front * 0.048f * cheekProjection * (cheekL + cheekR);

    // ---------------------------------------------------------------------
    // 4) NOSE: BOX / WEDGE FIRST, ANATOMY SECOND
    // ---------------------------------------------------------------------
    // Portrait construction treats the nose as top, side and bottom planes.
    // The dorsum below is therefore a broad wedge. It is deliberately NOT a
    // long vertical Gaussian centered on x=0.
    float noseRootY = 0.085f;
    float lowerDorsumY = -0.235f;
    float dorsum = 0.0f;

    if (yN <= noseRootY && yN >= lowerDorsumY) {
        float t = clampf(
            (noseRootY - yN) / (noseRootY - lowerDorsumY),
            0.0f, 1.0f);

        float coreHalf = lerpf(
            noseWingHalf * 0.28f,
            noseWingHalf * 0.44f,
            t);
        float sideHalf = lerpf(
            noseWingHalf * 0.64f,
            noseWingHalf * 0.88f,
            t);
        float outerHalf = lerpf(
            noseWingHalf * 0.78f,
            noseWingHalf,
            t);

        float ax = absf_local(x);
        float across = 0.0f;

        if (ax <= coreHalf) {
            // Top plane.
            across = 1.0f;
        } else if (ax <= sideHalf) {
            // Broad side plane. This is intentionally wider than the top.
            float s = (ax - coreHalf) / fmaxf(sideHalf - coreHalf, 0.0001f);
            across = 1.0f - 0.52f * smooth01(s);
        } else if (ax <= outerHalf) {
            // Soft connection from nose side plane back into maxilla/cheek.
            float s = (ax - sideHalf) / fmaxf(outerHalf - sideHalf, 0.0001f);
            across = 0.48f * (1.0f - smooth01(s));
        }

        // Projection begins very shallow at the saddle and increases toward
        // the cartilaginous lower dorsum.
        float projection = noseProjection *
            lerpf(0.018f, 0.105f, smooth01(t));

        dorsum = across * projection;
    }

    z += front * dorsum;

    // Tip / ball. Separate from the dorsum, rather than the dorsum simply
    // continuing until it becomes a point.
    float tip = compactEllipse(
        x, y, 0.0f, headHeight * -0.275f,
        fmaxf(noseWingHalf * 0.66f, 0.060f),
        headHeight * 0.070f, 0.18f);
    z += front * noseProjection * 0.135f * tip;

    // Alar wings wrap laterally around the ball.
    float alaX = noseWingHalf * 0.68f;
    float alaY = headHeight * NOSE_BASE;
    float alaRx = fmaxf(noseWingHalf * 0.42f, 0.045f);
    float alaRy = headHeight * 0.050f;
    float alaL = compactEllipse(
        x, y, -alaX, alaY,
        alaRx, alaRy, 0.10f);
    float alaR = compactEllipse(
        x, y, alaX, alaY,
        alaRx, alaRy, 0.10f);
    z += front * noseProjection * 0.036f * (alaL + alaR);

    // Shallow subnasal break. A later topology pass can turn this into a real
    // underside / nostril undercut.
    float subnasal = compactEllipse(
        x, y, 0.0f, headHeight * -0.345f,
        fmaxf(noseWingHalf * 0.72f, 0.060f),
        headHeight * 0.045f, 0.12f);
    z -= front * 0.014f * subnasal;

    // ---------------------------------------------------------------------
    // 5) DENTURE / TOOTH CYLINDER, THEN LIPS
    // ---------------------------------------------------------------------
    // The mouth is carried by a curved muzzle/denture mass. The lips are not
    // horizontal strips pasted onto a flat face.
    float mouthY = headHeight * MOUTH;
    float denture = compactEllipse(
        x, y, 0.0f, headHeight * -0.455f,
        fmaxf(mouthHalf * 1.18f, 0.18f),
        headHeight * 0.185f, 0.18f);
    z += front * 0.046f * muzzleProjection * denture;

    // Philtrum concavity above the upper lip.
    float philtrum = compactEllipse(
        x, y, 0.0f, headHeight * -0.405f,
        fmaxf(noseWingHalf * 0.30f, 0.040f),
        headHeight * 0.065f, 0.15f);
    z -= front * 0.015f * philtrum;

    // Three upper-lip pillows.
    float upperCenter = compactEllipse(
        x, y, 0.0f, mouthY + headHeight * 0.018f,
        fmaxf(mouthHalf * 0.25f, 0.050f),
        headHeight * 0.037f, 0.15f);
    float upperSideL = compactEllipse(
        x, y, -mouthHalf * 0.48f, mouthY,
        fmaxf(mouthHalf * 0.39f, 0.060f),
        headHeight * 0.040f, 0.12f);
    float upperSideR = compactEllipse(
        x, y, mouthHalf * 0.48f, mouthY,
        fmaxf(mouthHalf * 0.39f, 0.060f),
        headHeight * 0.040f, 0.12f);

    z += front * 0.020f * upperLip *
        (1.10f * upperCenter + 0.72f * (upperSideL + upperSideR));

    // Two lower-lip pillows.
    float lowerY = mouthY - headHeight * 0.060f;
    float lowerL = compactEllipse(
        x, y, -mouthHalf * 0.22f, lowerY,
        fmaxf(mouthHalf * 0.46f, 0.070f),
        headHeight * 0.047f, 0.14f);
    float lowerR = compactEllipse(
        x, y, mouthHalf * 0.22f, lowerY,
        fmaxf(mouthHalf * 0.46f, 0.070f),
        headHeight * 0.047f, 0.14f);

    z += front * 0.027f * lowerLip * (lowerL + lowerR);

    // Mouth crease follows the curved tooth-cylinder rather than a straight
    // horizontal Gaussian stripe.
    float axMouth = absf_local(x);
    if (axMouth < mouthHalf) {
        float xt = axMouth / fmaxf(mouthHalf, 0.0001f);
        float creaseY = mouthY - headHeight * 0.010f
            + headHeight * 0.018f * xt * xt;
        float dy = absf_local(y - creaseY) / (headHeight * 0.018f);
        float edge = 1.0f - smooth01(xt);
        float crease = (1.0f - smooth01(dy)) * edge;
        z -= front * 0.016f * crease;
    }

    // Corners/nodes pinch inward slightly.
    float nodeL = compactEllipse(
        x, y, -mouthHalf, mouthY,
        fmaxf(mouthHalf * 0.18f, 0.040f),
        headHeight * 0.050f, 0.12f);
    float nodeR = compactEllipse(
        x, y, mouthHalf, mouthY,
        fmaxf(mouthHalf * 0.18f, 0.040f),
        headHeight * 0.050f, 0.12f);
    z -= front * 0.010f * (nodeL + nodeR);

    // ---------------------------------------------------------------------
    // 6) CHIN / LOWER-FACE STAIRCASE
    // ---------------------------------------------------------------------
    // Recess below the lower lip followed by the chin mass.
    float labiomental = compactEllipse(
        x, y, 0.0f, headHeight * -0.615f,
        headWidth * 0.24f, headHeight * 0.070f, 0.10f);
    z -= front * 0.022f * labiomental;

    float chin = compactEllipse(
        x, y, 0.0f, headHeight * -0.72f,
        headWidth * 0.29f, headHeight * 0.145f, 0.20f);
    z += front * 0.072f * chinProjection * chin;

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

    if (iy == 0) {
        normals[id] = make_float4(0.0f, -1.0f, 0.0f, 0.0f);
        return;
    }
    if (iy == levels - 1) {
        normals[id] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
        return;
    }

    float du = 0.0025f;
    float dv = 0.0025f;
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

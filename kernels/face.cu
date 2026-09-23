// Procedural human head construction for CUDA WebShader.
// No model data is loaded. Each invocation generates one vertex and normal.

__device__ float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ float gauss1(float x, float center, float radius) {
    float d = (x - center) / radius;
    return expf(-(d * d));
}

__device__ float gauss2(float x, float y, float cx, float cy, float rx, float ry) {
    float dx = (x - cx) / rx;
    float dy = (y - cy) / ry;
    return expf(-(dx * dx + dy * dy));
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

    float theta = u * PI * 2.0f;
    float phi = (v - 0.5f) * PI;

    float st = sinf(theta);
    float ct = cosf(theta);
    float sp = sinf(phi);
    float cp = fmaxf(cosf(phi), 0.0f);

    float y = sp * headHeight;
    float x = st * cp * headWidth;
    float z = ct * cp * headDepth;

    // Only the forward half receives facial anatomy. A soft transition
    // around the temporal plane prevents a hard seam into the cranium.
    float front = clampf((ct + 0.12f) / 1.12f, 0.0f, 1.0f);
    front = front * front;

    // Artist-style blockout: temple -> cheek -> jaw before details.
    float temple = gauss1(sp, 0.36f, 0.19f);
    float cheekBand = gauss1(sp, 0.02f, 0.24f);
    float jawBand = gauss1(sp, -0.56f, 0.23f);

    x = x * (1.0f - front * temple * (1.0f - templeWidth));
    x = x * (1.0f + front * cheekBand * (cheekWidth - 1.0f));
    x = x * (1.0f - front * jawBand * (1.0f - jawWidth) * 0.82f);

    float eyeY = headHeight * 0.27f;
    float ex = eyeSpacing * headWidth * 1.55f;
    float eyeRx = 0.155f * eyeWidth;
    float eyeRy = 0.105f;

    float leftSocket = gauss2(x, y, -ex, eyeY, eyeRx, eyeRy);
    float rightSocket = gauss2(x, y, ex, eyeY, eyeRx, eyeRy);
    z -= front * socketDepth * (leftSocket + rightSocket);

    // Brow shelf. This sits above the socket instead of simply enlarging it.
    float browY = headHeight * 0.43f;
    float leftBrow = gauss2(x, y, -ex, browY, eyeRx * 1.16f, 0.085f);
    float rightBrow = gauss2(x, y, ex, browY, eyeRx * 1.16f, 0.085f);
    z += front * 0.075f * browProjection * (leftBrow + rightBrow);

    // Zygomatic / cheek plane.
    float cheekX = headWidth * 0.46f;
    float cheekY = headHeight * 0.02f;
    float cheekL = gauss2(x, y, -cheekX, cheekY, 0.22f, 0.24f);
    float cheekR = gauss2(x, y, cheekX, cheekY, 0.22f, 0.24f);
    z += front * 0.075f * cheekProjection * (cheekL + cheekR);

    // Nose = bridge + tip + alar wings, not one generic bump.
    float bridge = gauss2(x, y, 0.0f, headHeight * 0.18f, 0.085f * noseWidth, 0.31f);
    float tip = gauss2(x, y, 0.0f, headHeight * -0.075f, 0.135f * noseWidth, 0.13f);
    float alaL = gauss2(x, y, -0.105f * noseWidth, headHeight * -0.095f, 0.075f * noseWidth, 0.075f);
    float alaR = gauss2(x, y, 0.105f * noseWidth, headHeight * -0.095f, 0.075f * noseWidth, 0.075f);
    z += front * noseProjection * (0.13f * bridge + 0.27f * tip + 0.035f * (alaL + alaR));

    // Muzzle/orbicularis mass gives the mouth a platform.
    float muzzleY = headHeight * -0.32f;
    float muzzle = gauss2(x, y, 0.0f, muzzleY, 0.32f * mouthWidth, 0.22f);
    z += front * 0.065f * muzzleProjection * muzzle;

    // Lip volumes and the crease between them.
    float upper = gauss2(x, y, 0.0f, headHeight * -0.305f, 0.245f * mouthWidth, 0.050f);
    float lower = gauss2(x, y, 0.0f, headHeight * -0.385f, 0.235f * mouthWidth, 0.060f);
    float crease = gauss2(x, y, 0.0f, headHeight * -0.345f, 0.27f * mouthWidth, 0.022f);
    z += front * (0.035f * upperLip * upper + 0.048f * lowerLip * lower - 0.022f * crease);

    // Chin projection completes the lower face plane.
    float chin = gauss2(x, y, 0.0f, headHeight * -0.67f, 0.25f, 0.16f);
    z += front * 0.105f * chinProjection * chin;

    // Slight forehead/frontal-bone support keeps the upper face from reading
    // as a perfect ellipsoid.
    float forehead = gauss2(x, y, 0.0f, headHeight * 0.62f, 0.42f, 0.30f);
    z += front * 0.035f * forehead;

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

    // Derive a smooth normal from the same procedural surface. The seam is
    // naturally periodic in u; v is clamped at the crown/chin poles.
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

    float3 tangentU = make_float3(pu1.x - pu0.x, pu1.y - pu0.y, pu1.z - pu0.z);
    float3 tangentV = make_float3(pv1.x - pv0.x, pv1.y - pv0.y, pv1.z - pv0.z);
    float3 n = normalize(cross3(tangentU, tangentV));

    normals[id] = make_float4(n.x, n.y, n.z, 0.0f);
}

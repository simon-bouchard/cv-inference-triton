#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

#define RETURN_IF_ERROR(X)                    \
    do {                                      \
        TRITONSERVER_Error* rie_ = (X);       \
        if (rie_ != nullptr) return rie_;     \
    } while (false)

#include "decode_jpeg.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

static constexpr int RESIZE  = 512;
static constexpr int CROP    = 480;
static constexpr int CROP_OFF = (RESIZE - CROP) / 2;  // 16

// ImageNet normalization constants (mean/std per channel, pre-scaled by 1/255).
static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

// Bilinear resize to RESIZE×RESIZE, center-crop to CROP×CROP,
// ImageNet normalize, HWC→CHW, write FP32.
// src: H×W×3 uint8  →  dst: 1×3×CROP×CROP float
static void resize_crop_normalize_chw(
    const uint8_t* src, int sw, int sh, float* dst)
{
    const float scale_x = static_cast<float>(sw) / RESIZE;
    const float scale_y = static_cast<float>(sh) / RESIZE;

    for (int oy = 0; oy < CROP; ++oy) {
        // oy in crop space → oy+CROP_OFF in resize space → source coords
        float fy = (oy + CROP_OFF + 0.5f) * scale_y - 0.5f;
        int y0 = static_cast<int>(std::floor(fy));
        int y1 = y0 + 1;
        float yw = fy - static_cast<float>(y0);
        y0 = (y0 < 0) ? 0 : (y0 >= sh ? sh - 1 : y0);
        y1 = (y1 < 0) ? 0 : (y1 >= sh ? sh - 1 : y1);
        const uint8_t* r0 = src + y0 * sw * 3;
        const uint8_t* r1 = src + y1 * sw * 3;

        for (int ox = 0; ox < CROP; ++ox) {
            float fx = (ox + CROP_OFF + 0.5f) * scale_x - 0.5f;
            int x0 = static_cast<int>(std::floor(fx));
            int x1 = x0 + 1;
            float xw = fx - static_cast<float>(x0);
            x0 = (x0 < 0) ? 0 : (x0 >= sw ? sw - 1 : x0);
            x1 = (x1 < 0) ? 0 : (x1 >= sw ? sw - 1 : x1);

            for (int c = 0; c < 3; ++c) {
                float p00 = r0[x0 * 3 + c];
                float p01 = r0[x1 * 3 + c];
                float p10 = r1[x0 * 3 + c];
                float p11 = r1[x1 * 3 + c];
                float v = (1.f - yw) * ((1.f - xw) * p00 + xw * p01)
                        +        yw  * ((1.f - xw) * p10 + xw * p11);
                v = (v / 255.0f - MEAN[c]) / STD[c];
                dst[c * CROP * CROP + oy * CROP + ox] = v;
            }
        }
    }
}

extern "C" {

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
    uint32_t major, minor;
    RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&major, &minor));
    if (major != TRITONBACKEND_API_VERSION_MAJOR)
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "Triton backend API version mismatch");
    return nullptr;
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) { return nullptr; }

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) { return nullptr; }

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
    return nullptr;
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
    return nullptr;
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance,
    TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Request* request = requests[r];

        TRITONBACKEND_Input* input;
        RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "image_raw", &input));

        uint64_t in_byte_size;
        uint32_t buf_count;
        RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, nullptr, nullptr, &in_byte_size, &buf_count));

        const void* in_buf;
        TRITONSERVER_MemoryType in_mtype = TRITONSERVER_MEMORY_CPU;
        int64_t in_mid = 0;
        RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
            input, 0, &in_buf, &in_byte_size, &in_mtype, &in_mid));

        // TYPE_STRING layout: [uint32 byte-length][raw bytes]
        const auto* raw = static_cast<const uint8_t*>(in_buf);
        uint32_t jpeg_len;
        std::memcpy(&jpeg_len, raw, sizeof(jpeg_len));
        const uint8_t* jpeg_data = raw + sizeof(uint32_t);

        TRITONBACKEND_Response* response;
        RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));

        TRITONSERVER_Error* proc_err = nullptr;
        int img_w = 0, img_h = 0;
        std::vector<uint8_t> rgb;

        try {
            rgb = decode_jpeg(jpeg_data, jpeg_len, img_w, img_h);
        } catch (const std::exception& ex) {
            proc_err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, ex.what());
        }

        if (!proc_err) {
            TRITONBACKEND_Output* output;
            int64_t out_shape[] = {1, 3, CROP, CROP};
            proc_err = TRITONBACKEND_ResponseOutput(
                response, &output, "image_tensor",
                TRITONSERVER_TYPE_FP32, out_shape, 4);

            if (!proc_err) {
                uint64_t out_bytes = 1 * 3 * CROP * CROP * sizeof(float);
                void* out_buf;
                TRITONSERVER_MemoryType out_mtype = TRITONSERVER_MEMORY_CPU;
                int64_t out_mid = 0;
                proc_err = TRITONBACKEND_OutputBuffer(
                    output, &out_buf, out_bytes, &out_mtype, &out_mid);

                if (!proc_err) {
                    try {
                        resize_crop_normalize_chw(
                            rgb.data(), img_w, img_h,
                            static_cast<float*>(out_buf));
                    } catch (const std::exception& ex) {
                        proc_err = TRITONSERVER_ErrorNew(
                            TRITONSERVER_ERROR_INTERNAL, ex.what());
                    }
                }
            }
        }

        auto send_err = TRITONBACKEND_ResponseSend(
            response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, proc_err);
        if (proc_err)
            TRITONSERVER_ErrorDelete(proc_err);
        RETURN_IF_ERROR(send_err);

        RETURN_IF_ERROR(TRITONBACKEND_RequestRelease(
            request, TRITONSERVER_REQUEST_RELEASE_ALL));
    }

    return nullptr;
}

} // extern "C"

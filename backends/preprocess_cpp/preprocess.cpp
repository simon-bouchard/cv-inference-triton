#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

#define RETURN_IF_ERROR(X)                    \
    do {                                      \
        TRITONSERVER_Error* rie_ = (X);       \
        if (rie_ != nullptr) return rie_;     \
    } while (false)

#include <cstdio>
#include <jpeglib.h>
#include <setjmp.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int OUT_W = 640;
static constexpr int OUT_H = 640;

// libjpeg error handler that throws instead of calling exit().
struct JpegErrorMgr {
    jpeg_error_mgr pub;
    jmp_buf jmp;
    char msg[JMSG_LENGTH_MAX];
};

static void jpeg_error_exit(j_common_ptr cinfo)
{
    auto* err = reinterpret_cast<JpegErrorMgr*>(cinfo->err);
    (*cinfo->err->format_message)(cinfo, err->msg);
    longjmp(err->jmp, 1);
}

// Decode JPEG bytes to an RGB HWC uint8 buffer.
static std::vector<uint8_t>
decode_jpeg(const uint8_t* data, size_t len, int& out_w, int& out_h)
{
    jpeg_decompress_struct cinfo;
    JpegErrorMgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;

    if (setjmp(jerr.jmp)) {
        jpeg_destroy_decompress(&cinfo);
        throw std::runtime_error(std::string("JPEG decode: ") + jerr.msg);
    }

    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, static_cast<unsigned long>(len));
    jpeg_read_header(&cinfo, TRUE);

    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    out_w = static_cast<int>(cinfo.output_width);
    out_h = static_cast<int>(cinfo.output_height);

    std::vector<uint8_t> rgb(out_w * out_h * 3);

    while (static_cast<int>(cinfo.output_scanline) < out_h) {
        uint8_t* row = rgb.data() + cinfo.output_scanline * out_w * 3;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    return rgb;
}

// Letterbox + bilinear resize + normalize to [0,1] + HWC→CHW.
// Preserves aspect ratio by scaling to fit within OUT_W×OUT_H and padding
// the remainder with YOLOv8's expected grey (114/255).
// src: H×W×3 uint8  →  dst: 1×3×OUT_H×OUT_W float32
static constexpr float PAD_VAL = 114.f / 255.f;

static void letterbox_normalize_chw(
    const uint8_t* src, int sw, int sh, float* dst)
{
    const float scale = std::min(static_cast<float>(OUT_W) / sw,
                                 static_cast<float>(OUT_H) / sh);
    const int new_w = static_cast<int>(sw * scale);
    const int new_h = static_cast<int>(sh * scale);
    const int pad_x = (OUT_W - new_w) / 2;
    const int pad_y = (OUT_H - new_h) / 2;

    std::fill(dst, dst + 3 * OUT_H * OUT_W, PAD_VAL);

    const float sx = static_cast<float>(sw) / new_w;
    const float sy = static_cast<float>(sh) / new_h;

    for (int oy = pad_y; oy < pad_y + new_h; ++oy) {
        float fy = (oy - pad_y + 0.5f) * sy - 0.5f;
        int y0 = static_cast<int>(std::floor(fy));
        int y1 = y0 + 1;
        float yw = fy - static_cast<float>(y0);
        y0 = (y0 < 0) ? 0 : (y0 >= sh ? sh - 1 : y0);
        y1 = (y1 < 0) ? 0 : (y1 >= sh ? sh - 1 : y1);
        const uint8_t* r0 = src + y0 * sw * 3;
        const uint8_t* r1 = src + y1 * sw * 3;

        for (int ox = pad_x; ox < pad_x + new_w; ++ox) {
            float fx = (ox - pad_x + 0.5f) * sx - 0.5f;
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
                dst[c * OUT_H * OUT_W + oy * OUT_W + ox] = v / 255.0f;
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
            int64_t out_shape[] = {1, 3, OUT_H, OUT_W};
            proc_err = TRITONBACKEND_ResponseOutput(
                response, &output, "images",
                TRITONSERVER_TYPE_FP32, out_shape, 4);

            if (!proc_err) {
                uint64_t out_bytes = 1 * 3 * OUT_H * OUT_W * sizeof(float);
                void* out_buf;
                TRITONSERVER_MemoryType out_mtype = TRITONSERVER_MEMORY_CPU;
                int64_t out_mid = 0;
                proc_err = TRITONBACKEND_OutputBuffer(
                    output, &out_buf, out_bytes, &out_mtype, &out_mid);

                if (!proc_err) {
                    try {
                        letterbox_normalize_chw(
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

#pragma once
// backends/preprocess_cpp/decode_jpeg.h
// Shared JPEG decode utility used by all preprocess backends.

#include <cstdio>
#include <jpeglib.h>
#include <setjmp.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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

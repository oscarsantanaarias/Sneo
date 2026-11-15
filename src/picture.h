#pragma once
#include <windows.h>
#include <wincodec.h>
#include <string>
#include <vector>
#include <cstdint>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "propsys.lib")

template<typename T> struct Held {
    T* p = nullptr;
    ~Held() { if(p) p->Release(); }
    T** operator&() { return &p; }
    T* operator->() { return p; }
};

struct Picture {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
};

inline bool readPicture(const std::string& bytes, Picture& out, std::string& error) {
    HRESULT started = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool owned = started == S_OK || started == S_FALSE;
    auto finish = [&](bool ok) {
        if(owned) CoUninitialize();
        return ok;
    };
    Held<IWICImagingFactory> factory;
    if(FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))){
        error = "windows could not open its picture decoder";
        return finish(false);
    }
    Held<IWICStream> stream;
    Held<IWICBitmapDecoder> decoder;
    Held<IWICBitmapFrameDecode> frame;
    Held<IWICFormatConverter> converter;
    Held<IWICBitmapFlipRotator> turned;
    bool ok = SUCCEEDED(factory->CreateStream(&stream))
        && SUCCEEDED(stream->InitializeFromMemory((BYTE*)bytes.data(), (DWORD)bytes.size()))
        && SUCCEEDED(factory->CreateDecoderFromStream(stream.p, nullptr, WICDecodeMetadataCacheOnDemand, &decoder))
        && SUCCEEDED(decoder->GetFrame(0, &frame))
        && SUCCEEDED(factory->CreateFormatConverter(&converter))
        && SUCCEEDED(converter->Initialize(frame.p, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom));
    if(!ok){
        error = "the picture could not be read, use png, jpg, bmp, gif or webp";
        return finish(false);
    }
    int turn = 1;
    Held<IWICMetadataQueryReader> facts;
    if(SUCCEEDED(frame->GetMetadataQueryReader(&facts))){
        PROPVARIANT value;
        PropVariantInit(&value);
        if(SUCCEEDED(facts->GetMetadataByName(L"System.Photo.Orientation", &value)) && value.vt == VT_UI2) turn = value.uiVal;
        PropVariantClear(&value);
    }
    WICBitmapTransformOptions how = turn == 2
        ? WICBitmapTransformFlipHorizontal
        : turn == 3
            ? WICBitmapTransformRotate180
            : turn == 4
                ? WICBitmapTransformFlipVertical
                : turn == 5
                    ? (WICBitmapTransformOptions)(WICBitmapTransformRotate90 | WICBitmapTransformFlipHorizontal)
                    : turn == 6
                        ? WICBitmapTransformRotate90
                        : turn == 7
                            ? (WICBitmapTransformOptions)(WICBitmapTransformRotate270 | WICBitmapTransformFlipHorizontal)
                            : turn == 8
                                ? WICBitmapTransformRotate270
                                : WICBitmapTransformRotate0;
    IWICBitmapSource* source = converter.p;
    if(how != WICBitmapTransformRotate0 && SUCCEEDED(factory->CreateBitmapFlipRotator(&turned)) && SUCCEEDED(turned->Initialize(converter.p, how))) source = turned.p;
    UINT w = 0, h = 0;
    if(FAILED(source->GetSize(&w, &h)) || w == 0 || h == 0){
        error = "the picture has no pixels";
        return finish(false);
    }
    std::vector<uint8_t> rgba((size_t)w * h * 4);
    if(FAILED(source->CopyPixels(nullptr, w * 4, (UINT)rgba.size(), rgba.data()))){
        error = "the picture pixels could not be read";
        return finish(false);
    }
    out.width = (int)w;
    out.height = (int)h;
    out.rgb.assign((size_t)w * h * 3, 0);
    for(size_t i = 0; i < (size_t)w * h; i++){
        int alpha = rgba[i * 4 + 3];
        for(int c = 0; c < 3; c++) out.rgb[i * 3 + c] = (uint8_t)((rgba[i * 4 + c] * alpha + 255 * (255 - alpha) + 127) / 255);
    }
    return finish(true);
}

inline bool decodeBase64(const std::string& text, std::string& out) {
    static int8_t table[256];
    static bool built = false;
    if(!built){
        for(int i = 0; i < 256; i++) table[i] = -1;
        const char* letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for(int i = 0; i < 64; i++) table[(uint8_t)letters[i]] = (int8_t)i;
        table['-'] = 62;
        table['_'] = 63;
        built = true;
    }
    out.clear();
    out.reserve(text.size() * 3 / 4);
    uint32_t bits = 0;
    int count = 0;
    for(char c : text){
        if(c == '=' || c == '\r' || c == '\n' || c == ' ') continue;
        int8_t v = table[(uint8_t)c];
        if(v < 0) return false;
        bits = bits << 6 | (uint32_t)v;
        count += 6;
        if(count < 8) continue;
        count -= 8;
        out.push_back((char)(bits >> count & 255));
    }
    return !out.empty();
}

inline bool pictureFromUrl(const std::string& url, std::string& bytes, std::string& error) {
    size_t comma = url.find(',');
    if(url.rfind("data:", 0) != 0 || comma == std::string::npos || url.find(";base64") > comma){
        error = "pictures must come inline as base64 data urls";
        return false;
    }
    if(!decodeBase64(url.substr(comma + 1), bytes)){
        error = "the picture data is not valid base64";
        return false;
    }
    return true;
}

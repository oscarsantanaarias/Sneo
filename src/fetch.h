#pragma once
#include <string>
#include <vector>
#include <functional>
#include <cstdint>

#ifndef _WIN32
#include <cstdio>
#include "platform.h"
// en linux las descargas van por curl: el rango se pide igual que en windows y lo que llega se entrega de a pedazos
struct WebLink {
    bool get(const std::string& url, uint64_t begin, uint64_t end, const std::function<bool(const char*, size_t)>& sink, std::string& error) {
        if(url.find('\'') != std::string::npos){
            error = "the address has a quote in it";
            return false;
        }
        bool ranged = end > begin;
        std::string command = "curl -sSfL --retry 3 --connect-timeout 15 -A Sneo/1";
        if(ranged) command += " -r " + std::to_string(begin) + "-" + std::to_string(end - 1);
        command += " '" + url + "' 2>/dev/null";
        FILE* pipe = popen(command.c_str(), "r");
        if(!pipe){
            error = "curl did not start, it is needed for downloads on linux";
            return false;
        }
        std::vector<char> buffer(1 << 20);
        uint64_t received = 0;
        bool fine = true;
        size_t got = 0;
        while(fine && (got = fread(buffer.data(), 1, buffer.size(), pipe)) > 0){
            received += got;
            if(!sink(buffer.data(), got)){
                error = "stopped";
                fine = false;
            }
        }
        int status = pclose(pipe);
        if(fine && status != 0){
            error = "download failed for " + url + " (curl " + std::to_string(status) + ")";
            return false;
        }
        if(fine && ranged && received != end - begin){
            error = "short read: " + std::to_string(received) + " of " + std::to_string(end - begin) + " bytes";
            return false;
        }
        return fine;
    }

    bool text(const std::string& url, std::string& out, std::string& error) {
        out.clear();
        return get(url, 0, 0, [&](const char* data, size_t length){
            out.append(data, length);
            return out.size() < (64u << 20);
        }, error);
    }
};
#else
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>

#pragma comment(lib, "winhttp.lib")

struct WebLink {
    HINTERNET session = nullptr;

    WebLink() {
        session = WinHttpOpen(L"Sneo/1", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        if(!session) session = WinHttpOpen(L"Sneo/1", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        if(!session) return;
        DWORD protocols = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2;
        WinHttpSetOption(session, WINHTTP_OPTION_SECURE_PROTOCOLS, &protocols, sizeof(protocols));
        WinHttpSetTimeouts(session, 15000, 15000, 30000, 60000);
    }

    ~WebLink() {
        if(session) WinHttpCloseHandle(session);
    }

    bool get(const std::string& url, uint64_t begin, uint64_t end, const std::function<bool(const char*, size_t)>& sink, std::string& error) {
        if(!session){
            error = "no http session";
            return false;
        }

        std::wstring wide(url.begin(), url.end());
        wchar_t host[256] = {};
        wchar_t path[4096] = {};
        wchar_t extra[4096] = {};
        URL_COMPONENTS parts = {};
        parts.dwStructSize = sizeof(parts);
        parts.lpszHostName = host;
        parts.dwHostNameLength = 256;
        parts.lpszUrlPath = path;
        parts.dwUrlPathLength = 4096;
        parts.lpszExtraInfo = extra;
        parts.dwExtraInfoLength = 4096;
        if(!WinHttpCrackUrl(wide.c_str(), 0, 0, &parts)){
            error = "bad url " + url;
            return false;
        }

        HINTERNET connection = WinHttpConnect(session, host, parts.nPort, 0);
        if(!connection){
            error = "cannot reach " + url;
            return false;
        }

        std::wstring target = std::wstring(path) + extra;
        DWORD flags = parts.nScheme == INTERNET_SCHEME_HTTPS
            ? WINHTTP_FLAG_SECURE
            : 0;
        HINTERNET request = WinHttpOpenRequest(connection, L"GET", target.c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
        if(!request){
            WinHttpCloseHandle(connection);
            error = "cannot open request";
            return false;
        }

        bool ranged = end > begin;
        std::wstring range = L"Range: bytes=" + std::to_wstring(begin) + L"-" + std::to_wstring(end - 1);
        bool sent = WinHttpSendRequest(request, ranged ? range.c_str() : WINHTTP_NO_ADDITIONAL_HEADERS, ranged ? (DWORD)-1L : 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0)
            && WinHttpReceiveResponse(request, nullptr);

        DWORD status = 0;
        DWORD size = sizeof(status);
        if(sent) WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &size, WINHTTP_NO_HEADER_INDEX);

        bool good = sent && (ranged ? status == 206 : status == 200);
        if(good && ranged){
            wchar_t answered[256] = {};
            DWORD length = sizeof(answered);
            std::wstring wanted = L"bytes " + std::to_wstring(begin) + L"-" + std::to_wstring(end - 1) + L"/";
            good = WinHttpQueryHeaders(request, WINHTTP_QUERY_CONTENT_RANGE, WINHTTP_HEADER_NAME_BY_INDEX, answered, &length, WINHTTP_NO_HEADER_INDEX)
                && std::wstring(answered).rfind(wanted, 0) == 0;
        }
        if(!good){
            error = sent
                ? "server answered " + std::to_string(status) + " for " + url
                : "request failed for " + url + " (" + std::to_string(GetLastError()) + ")";
            WinHttpCloseHandle(request);
            WinHttpCloseHandle(connection);
            return false;
        }

        std::vector<char> buffer(1 << 20);
        uint64_t received = 0;
        bool fine = true;
        while(fine){
            DWORD ready = 0;
            if(!WinHttpQueryDataAvailable(request, &ready)){
                error = "connection dropped after " + std::to_string(received) + " bytes";
                fine = false;
                break;
            }
            if(ready == 0) break;
            DWORD got = 0;
            if(!WinHttpReadData(request, buffer.data(), std::min<DWORD>(ready, (DWORD)buffer.size()), &got)){
                error = "read failed after " + std::to_string(received) + " bytes";
                fine = false;
                break;
            }
            if(got == 0) break;
            received += got;
            if(!sink(buffer.data(), got)){
                error = "stopped";
                fine = false;
            }
        }

        WinHttpCloseHandle(request);
        WinHttpCloseHandle(connection);
        if(fine && ranged && received != end - begin){
            error = "short read: " + std::to_string(received) + " of " + std::to_string(end - begin) + " bytes";
            return false;
        }
        return fine;
    }

    bool text(const std::string& url, std::string& out, std::string& error) {
        out.clear();
        return get(url, 0, 0, [&](const char* data, size_t length){
            out.append(data, length);
            return out.size() < (64u << 20);
        }, error);
    }
};
#endif

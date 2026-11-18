#pragma once
#include "hardware.h"
#include <vector>
#include <thread>
#include <chrono>
#ifdef _WIN32
#include <iphlpapi.h>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#endif

// tiempo de espera de un socket en ms, 0 es sin limite: windows lo toma en ms y linux en segundos y microsegundos
inline void socketWait(SOCKET s, int option, DWORD ms) {
#ifdef _WIN32
    setsockopt(s, SOL_SOCKET, option, (char*)&ms, sizeof(ms));
#else
    timeval wait = {(time_t)(ms / 1000), (suseconds_t)((ms % 1000) * 1000)};
    setsockopt(s, SOL_SOCKET, option, (char*)&wait, sizeof(wait));
#endif
}

enum PacketKind : uint32_t {
    PACKET_HELLO = 1,
    PACKET_NODE_INFO = 2,
    PACKET_STATS_ASK = 3,
    PACKET_STATS = 4,
    PACKET_PLAN = 5,
    PACKET_PLAN_ACK = 6,
    PACKET_WEIGHTS_ASK = 7,
    PACKET_WEIGHTS_CHUNK = 8,
    PACKET_WEIGHTS_DONE = 9,
    PACKET_FORWARD = 10,
    PACKET_FORWARD_DONE = 11,
    PACKET_RESET = 12,
    PACKET_REFUSED = 13,
    PACKET_BYE = 14,
    PACKET_ROWS = 15,
    PACKET_ROWS_DONE = 16,
    PACKET_STATE = 17,
    PACKET_READY = 18
};

struct PacketHead {
    uint32_t kind;
    uint32_t flags;
    uint64_t size;
};

struct HelloBody {
    char key[64];
    uint32_t version;
    uint32_t pad;
};

const uint32_t protocolVersion = 4;

inline bool startSockets() {
    static bool ready = false;
    if(ready) return true;
    WSADATA wsa;
    if(WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
    ready = true;
    return true;
}

inline bool sendAll(SOCKET s, const void* data, uint64_t size) {
    const char* ptr = (const char*)data;
    while(size > 0){
        int chunk = size > (1 << 30) ? (1 << 30) : (int)size;
        int sent = send(s, ptr, chunk, 0);
        if(sent <= 0) return false;
        ptr += sent;
        size -= sent;
    }
    return true;
}

inline bool recvAll(SOCKET s, void* data, uint64_t size) {
    char* ptr = (char*)data;
    while(size > 0){
        int chunk = size > (1 << 30) ? (1 << 30) : (int)size;
        int got = recv(s, ptr, chunk, 0);
        if(got <= 0) return false;
        ptr += got;
        size -= got;
    }
    return true;
}

inline bool sendPacket(SOCKET s, uint32_t kind, const void* body, uint64_t size) {
    PacketHead head = {kind, 0, size};
    if(!sendAll(s, &head, sizeof(head))) return false;
    if(size == 0) return true;
    return sendAll(s, body, size);
}

inline bool readPacket(SOCKET s, PacketHead& head, std::vector<char>& body, uint64_t most = 16ULL << 30) {
    if(!recvAll(s, &head, sizeof(head))) return false;
    if(head.size > most) return false;
    body.resize((size_t)head.size);
    if(head.size == 0) return true;
    return recvAll(s, body.data(), head.size);
}

inline void tuneSocket(SOCKET s) {
    int flag = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(flag));
    int buffer = 8 << 20;
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, (char*)&buffer, sizeof(buffer));
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, (char*)&buffer, sizeof(buffer));
}

inline SOCKET listenTcp(int port, const std::string& host = "0.0.0.0") {
    if(!startSockets()) return INVALID_SOCKET;

    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if(s == INVALID_SOCKET) return INVALID_SOCKET;

    int reuse = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (char*)&reuse, sizeof(reuse));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)port);
    if(inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1){
        closesocket(s);
        return INVALID_SOCKET;
    }

    if(bind(s, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR || listen(s, SOMAXCONN) == SOCKET_ERROR){
        closesocket(s);
        return INVALID_SOCKET;
    }
    return s;
}

inline SOCKET connectTcp(const std::string& ip, int port, int timeoutMs) {
    if(!startSockets()) return INVALID_SOCKET;

    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if(s == INVALID_SOCKET) return INVALID_SOCKET;

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)port);
    if(inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1){
        closesocket(s);
        return INVALID_SOCKET;
    }

    u_long nonBlocking = 1;
    ioctlsocket(s, FIONBIO, &nonBlocking);
    connect(s, (sockaddr*)&addr, sizeof(addr));

    fd_set writable;
    FD_ZERO(&writable);
    FD_SET(s, &writable);
    timeval wait = {timeoutMs / 1000, (timeoutMs % 1000) * 1000};
    if(select((int)s + 1, nullptr, &writable, nullptr, &wait) != 1){
        closesocket(s);
        return INVALID_SOCKET;
    }

    int error = 0;
    socklen_t length = sizeof(error);
    getsockopt(s, SOL_SOCKET, SO_ERROR, (char*)&error, &length);
    if(error != 0){
        closesocket(s);
        return INVALID_SOCKET;
    }

    nonBlocking = 0;
    ioctlsocket(s, FIONBIO, &nonBlocking);
    tuneSocket(s);
    return s;
}

#ifndef _WIN32
// las direcciones ipv4 de las placas de red prendidas, sin la propia ni las automaticas 169.254
inline std::vector<std::string> localAddresses() {
    std::vector<std::string> found;
    ifaddrs* list = nullptr;
    if(getifaddrs(&list) != 0) return found;
    for(ifaddrs* item = list; item; item = item->ifa_next){
        if(!item->ifa_addr || item->ifa_addr->sa_family != AF_INET) continue;
        if(!(item->ifa_flags & IFF_UP) || (item->ifa_flags & IFF_LOOPBACK)) continue;
        char ip[64] = {};
        inet_ntop(AF_INET, &((sockaddr_in*)item->ifa_addr)->sin_addr, ip, sizeof(ip));
        if(strncmp(ip, "169.254.", 8) == 0 || strncmp(ip, "127.", 4) == 0) continue;
        found.push_back(ip);
    }
    freeifaddrs(list);
    return found;
}
#else
inline std::vector<std::string> localAddresses() {
    std::vector<std::string> found;
    if(!startSockets()) return found;

    ULONG size = 16384;
    std::vector<uint8_t> buffer(size);
    ULONG flags = GAA_FLAG_INCLUDE_GATEWAYS | GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER;
    ULONG result = GetAdaptersAddresses(AF_INET, flags, nullptr, (IP_ADAPTER_ADDRESSES*)buffer.data(), &size);
    if(result == ERROR_BUFFER_OVERFLOW){
        buffer.resize(size);
        result = GetAdaptersAddresses(AF_INET, flags, nullptr, (IP_ADAPTER_ADDRESSES*)buffer.data(), &size);
    }
    for(IP_ADAPTER_ADDRESSES* adapter = result == NO_ERROR ? (IP_ADAPTER_ADDRESSES*)buffer.data() : nullptr; adapter; adapter = adapter->Next){
        if(adapter->OperStatus != IfOperStatusUp || adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK || !adapter->FirstGatewayAddress) continue;
        for(IP_ADAPTER_UNICAST_ADDRESS* unicast = adapter->FirstUnicastAddress; unicast; unicast = unicast->Next){
            char ip[64] = {};
            inet_ntop(AF_INET, &((sockaddr_in*)unicast->Address.lpSockaddr)->sin_addr, ip, sizeof(ip));
            if(strncmp(ip, "169.254.", 8) == 0) continue;
            found.push_back(ip);
        }
    }
    if(!found.empty()) return found;

    SOCKET probe = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if(probe == INVALID_SOCKET) return found;

    INTERFACE_INFO list[32];
    DWORD bytes = 0;
    if(WSAIoctl(probe, SIO_GET_INTERFACE_LIST, nullptr, 0, list, sizeof(list), &bytes, nullptr, nullptr) == 0){
        int count = bytes / sizeof(INTERFACE_INFO);
        for(int i = 0; i < count; i++){
            if(!(list[i].iiFlags & IFF_UP) || (list[i].iiFlags & IFF_LOOPBACK)) continue;
            char ip[64] = {};
            inet_ntop(AF_INET, &list[i].iiAddress.AddressIn.sin_addr, ip, sizeof(ip));
            if(strncmp(ip, "169.254.", 8) == 0 || strncmp(ip, "127.", 4) == 0) continue;
            found.push_back(ip);
        }
    }
    closesocket(probe);
    return found;
}
#endif

inline uint64_t nowMs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline uint64_t nowMicros() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

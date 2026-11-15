#pragma once
#include "net.h"
#include "quant.h"
#include <string>
#include <vector>
#include <future>

enum StateOp { STATE_SAVE = 1, STATE_RESTORE = 2, STATE_COMMIT = 3, STATE_RESET = 4 };

// lo que el nodo sabe hacer ademas de lo de siempre lo dice en su READY y el master lo pide en las banderas de cada paquete
// empaquetadas: las filas llegan como los bloques cuantizados de la tabla y el nodo las abre con el mismo codigo que el master, salen iguales
// medias: el nodo puede devolver la pila en fp16 cuando todo cabe, la mitad de bytes por la red
// dejar: en la pasada de verificacion el nodo no toca su estado y lo avanza cuando el master confirma, sin guardar ni restaurar copias
// por partes: en un prompt largo el nodo manda cada parte apenas la termina y la red trabaja mientras calcula la siguiente
enum RowsFlag : uint32_t { ROWS_PACKED = 1, ROWS_HALF = 2, ROWS_LEAVE = 4, ROWS_PIECES = 8 };
const uint32_t rowsFeatures = ROWS_PACKED | ROWS_HALF | ROWS_LEAVE | ROWS_PIECES;
// desde cuantas filas conviene pedir la respuesta por partes
const int piecesFrom = 256;

struct PackedHead {
    int32_t rows;
    int32_t record;
    int32_t embedType;
    int32_t embedBytes;
    int32_t pleType;
    int32_t pleBytes;
};

inline uint64_t packedSize(int rows, int embedBytes, int pleBytes, int pleParts) {
    return sizeof(PackedHead) + 4ULL * rows + (uint64_t)rows * embedBytes + (uint64_t)rows * pleParts * pleBytes;
}

// abre las filas empaquetadas en floats, revisando que cada tamano sea el que dicen los tipos
inline bool unpackRows(const char* body, uint64_t size, int width, int pleParts, int pleWidth, std::vector<int>& positions, std::vector<float>& embeds, std::vector<float>& ples, int& rows, bool& record, std::string& why) {
    PackedHead head = {};
    if(size < sizeof(head)){
        why = "packed rows without their head";
        return false;
    }
    memcpy(&head, body, sizeof(head));
    bool plain = head.pleType < 0 && head.pleBytes == 0;
    if(head.rows <= 0 || head.rows > 1 << 16 || !knownType((uint32_t)head.embedType) || rowBytes((uint32_t)head.embedType, width) != (uint64_t)head.embedBytes || head.embedBytes <= 0){
        why = "packed rows with an embedding type this node does not read";
        return false;
    }
    if(!plain && (!knownType((uint32_t)head.pleType) || rowBytes((uint32_t)head.pleType, pleWidth) != (uint64_t)head.pleBytes || head.pleBytes <= 0)){
        why = "packed rows with a per layer type this node does not read";
        return false;
    }
    if(size != packedSize(head.rows, head.embedBytes, plain ? 0 : head.pleBytes, pleParts)){
        why = "packed rows of the wrong size";
        return false;
    }
    rows = head.rows;
    record = head.record != 0;
    positions.resize((size_t)rows);
    embeds.resize((size_t)rows * width);
    ples.assign((size_t)rows * width, 0.0f);
    const char* at = body + sizeof(head);
    memcpy(positions.data(), at, 4ULL * rows);
    at += 4ULL * rows;
    for(int t = 0; t < rows; t++, at += head.embedBytes) dequantRow((uint32_t)head.embedType, (const uint8_t*)at, embeds.data() + (size_t)t * width, width);
    for(int t = 0; t < rows && !plain; t++){
        for(int p = 0; p < pleParts; p++, at += head.pleBytes) dequantRow((uint32_t)head.pleType, (const uint8_t*)at, ples.data() + (size_t)t * width + (size_t)p * pleWidth, pleWidth);
    }
    return true;
}

// fp16 solo si todo es finito y cabe con margen, si no la pila va entera en fp32
inline bool fitsHalf(const float* v, uint64_t count) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 most = _mm256_setzero_ps(), odd = _mm256_setzero_ps();
    uint64_t i = 0;
    for(; i + 8 <= count; i += 8){
        __m256 x = _mm256_loadu_ps(v + i);
        odd = _mm256_or_ps(odd, _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
        most = _mm256_max_ps(most, _mm256_andnot_ps(sign, x));
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, most);
    float top = 0;
    for(float x : lanes) top = std::max(top, x);
    for(; i < count; i++){
        if(v[i] != v[i]) return false;
        top = std::max(top, fabsf(v[i]));
    }
    return !_mm256_movemask_ps(odd) && top <= 65000.0f;
}

inline void toHalf(const float* v, uint16_t* out, uint64_t count) {
    uint64_t i = 0;
    for(; i + 8 <= count; i += 8) _mm_storeu_si128((__m128i*)(out + i), _mm256_cvtps_ph(_mm256_loadu_ps(v + i), _MM_FROUND_TO_NEAREST_INT));
    for(; i < count; i++) out[i] = (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(v[i]), _MM_FROUND_TO_NEAREST_INT), 0);
}

inline void fromHalf(const uint16_t* v, float* out, uint64_t count) {
    uint64_t i = 0;
    for(; i + 8 <= count; i += 8) _mm256_storeu_ps(out + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(v + i))));
    for(; i < count; i++) out[i] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(v[i])));
}

struct FrontLink {
    SOCKET wire = INVALID_SOCKET;
    std::string node;
    std::string problem;
    std::vector<char> outgoing;
    std::vector<uint16_t> halves;
    uint64_t postedAt = 0;
    double roundMs = 0;
    uint64_t rounds = 0;
    uint64_t sentBytes = 0;
    uint64_t gotBytes = 0;
    int relayLimit = 0;
    uint32_t features = 0;

    bool broken(const std::string& why) {
        if(problem.empty()) problem = why;
        return false;
    }

    // cabecera y cuerpo salen juntos en un solo envio, el cuerpo se escribe en lo que devuelve
    char* begin(uint32_t kind, uint32_t flags, uint64_t size) {
        outgoing.resize(sizeof(PacketHead) + (size_t)size);
        PacketHead head = {kind, flags, size};
        memcpy(outgoing.data(), &head, sizeof(head));
        return outgoing.data() + sizeof(head);
    }

    // mandar y recibir van por separado: en un prompt largo el master manda la parte que sigue y la recibe mientras hace sus capas
    bool post() {
        if(wire == INVALID_SOCKET) return broken("the pool node " + node + " is not connected");
        postedAt = nowMicros();
        if(!sendAll(wire, outgoing.data(), outgoing.size())) return broken("lost the pool node " + node + " while sending rows (socket error " + std::to_string(WSAGetLastError()) + ")");
        sentBytes += outgoing.size();
        return true;
    }

    bool exchange(int n, float* out, int outWidth) {
        return post() && collect(n, out, outWidth);
    }

    bool collect(int n, float* out, int outWidth) {
        PacketHead reply;
        int32_t status[2] = {};
        if(!recvAll(wire, &reply, sizeof(reply)) || reply.kind != PACKET_ROWS_DONE || reply.size < 8 || !recvAll(wire, status, 8)) return broken("lost the pool node " + node + " while waiting for its layers (socket error " + std::to_string(WSAGetLastError()) + ")");
        uint64_t rest = reply.size - 8;
        if(!status[0]){
            std::string why((size_t)std::min<uint64_t>(rest, 4096), ' ');
            if(rest > 0 && !recvAll(wire, &why[0], why.size())) return broken("lost the pool node " + node);
            return broken("the pool node " + node + " failed: " + why);
        }
        if(reply.flags & ROWS_PIECES) return pieces(n, status[1], out, outWidth, sizeof(reply) + 8);
        bool half = (reply.flags & ROWS_HALF) != 0;
        uint64_t cells = (uint64_t)n * outWidth;
        if(status[1] != n || rest != cells * (half ? 2 : 4)) return broken("the pool node " + node + " answered with the wrong number of rows");
        if(half){
            halves.resize((size_t)cells);
            if(!recvAll(wire, halves.data(), rest)) return broken("lost the pool node " + node + " while reading its layers (socket error " + std::to_string(WSAGetLastError()) + ")");
            fromHalf(halves.data(), out, cells);
        } else if(!recvAll(wire, out, rest)) return broken("lost the pool node " + node + " while reading its layers (socket error " + std::to_string(WSAGetLastError()) + ")");
        roundMs += (nowMicros() - postedAt) / 1000.0;
        rounds++;
        gotBytes += sizeof(reply) + reply.size;
        return true;
    }

    // la respuesta por partes: cada una dice cuantas filas trae y si vienen en fp16, una con -1 trae el error del nodo
    bool pieces(int n, int told, float* out, int outWidth, uint64_t got) {
        if(told != n) return broken("the pool node " + node + " answered with the wrong number of rows");
        int done = 0;
        while(done < n){
            int32_t piece[2] = {};
            if(!recvAll(wire, piece, 8)) return broken("lost the pool node " + node + " while reading its layers (socket error " + std::to_string(WSAGetLastError()) + ")");
            got += 8;
            if(piece[0] < 0){
                std::string why((size_t)std::min(std::max(piece[1], 0), 4096), ' ');
                if(!why.empty() && !recvAll(wire, &why[0], why.size())) return broken("lost the pool node " + node);
                return broken("the pool node " + node + " failed: " + why);
            }
            if(piece[0] == 0 || done + piece[0] > n) return broken("the pool node " + node + " answered with the wrong number of rows");
            uint64_t cells = (uint64_t)piece[0] * outWidth;
            float* into = out + (size_t)done * outWidth;
            if(piece[1]){
                halves.resize((size_t)cells);
                if(!recvAll(wire, halves.data(), cells * 2)) return broken("lost the pool node " + node + " while reading its layers (socket error " + std::to_string(WSAGetLastError()) + ")");
                fromHalf(halves.data(), into, cells);
            } else if(!recvAll(wire, into, cells * 4)) return broken("lost the pool node " + node + " while reading its layers (socket error " + std::to_string(WSAGetLastError()) + ")");
            got += cells * (piece[1] ? 2 : 4);
            done += piece[0];
        }
        roundMs += (nowMicros() - postedAt) / 1000.0;
        rounds++;
        gotBytes += got;
        return true;
    }

    bool rows(int n, bool record, const int* positions, const float* embeds, const float* ples, int width, float* out, int outWidth, uint32_t flags = 0) {
        uint64_t rowBytes = (uint64_t)width * 4;
        char* at = begin(PACKET_ROWS, flags, 8 + (uint64_t)n * 4 + 2 * n * rowBytes);
        int32_t head[2] = {n, record ? 1 : 0};
        memcpy(at, head, 8);
        memcpy(at + 8, positions, (size_t)n * 4);
        memcpy(at + 8 + (size_t)n * 4, embeds, (size_t)(n * rowBytes));
        memcpy(at + 8 + (size_t)n * 4 + (size_t)(n * rowBytes), ples, (size_t)(n * rowBytes));
        return exchange(n, out, outWidth);
    }

    bool state(int op, int arg) {
        if(wire == INVALID_SOCKET) return broken("the pool node " + node + " is not connected");
        char packet[sizeof(PacketHead) + 8];
        PacketHead head = {PACKET_STATE, 0, 8};
        int32_t body[2] = {op, arg};
        memcpy(packet, &head, sizeof(head));
        memcpy(packet + sizeof(head), body, 8);
        if(!sendAll(wire, packet, sizeof(packet))) return broken("lost the pool node " + node + " while sending state");
        sentBytes += sizeof(packet);
        return true;
    }
};

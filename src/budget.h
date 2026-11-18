#pragma once

#include "platform.h"
#include <cstdint>
#include <cstring>
#include <cwchar>
#include <vector>

#ifndef _WIN32
// en linux no hay presupuesto de wddm: la tarjeta no le quita memoria al proceso, asi que no hay nada que vigilar
struct ProcessVram {
    bool open() { return false; }
    bool read(uint64_t&) { return false; }
    void close() {}
};

struct VramBudget {
    bool open(const char*) { return false; }
    bool read(uint64_t&, uint64_t&) { return false; }
    bool readShared(uint64_t&, uint64_t&) { return false; }
    void close() {}
};
#else
#include <dxgi1_4.h>
#include <pdh.h>

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "pdh.lib")

// la memoria de video dedicada de este proceso segun windows, la misma que muestra el administrador de tareas
// despues de cargar solo sube, si baja sin que soltemos nada es que windows movio parte del modelo a la ram compartida
struct ProcessVram {
    PDH_HQUERY query = nullptr;
    PDH_HCOUNTER counter = nullptr;
    std::vector<unsigned char> items;

    bool open() {
        if(query) return true;
        if(PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS) return false;
        if(PdhAddEnglishCounterW(query, L"\\GPU Process Memory(*)\\Dedicated Usage", 0, &counter) != ERROR_SUCCESS){
            close();
            return false;
        }
        return true;
    }

    bool read(uint64_t& bytes) {
        if(!query || PdhCollectQueryData(query) != ERROR_SUCCESS) return false;
        DWORD size = 0, count = 0;
        PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &size, &count, nullptr);
        if(size == 0) return false;
        items.resize(size);
        auto* list = (PDH_FMT_COUNTERVALUE_ITEM_W*)items.data();
        if(PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &size, &count, list) != ERROR_SUCCESS) return false;
        wchar_t own[32];
        swprintf_s(own, L"pid_%lu_", GetCurrentProcessId());
        size_t ownLength = wcslen(own);
        uint64_t total = 0;
        bool found = false;
        for(DWORD i = 0; i < count; i++){
            if(wcsncmp(list[i].szName, own, ownLength) != 0) continue;
            total += (uint64_t)list[i].FmtValue.largeValue;
            found = true;
        }
        bytes = total;
        return found;
    }

    void close() {
        if(query) PdhCloseQuery(query);
        query = nullptr;
        counter = nullptr;
    }
};

// windows le da a cada programa un presupuesto de memoria de video y lo baja cuando otro programa se activa
// si usamos mas que ese presupuesto mueve memoria nuestra a la ram compartida y cada paso queda lento
struct VramBudget {
    IDXGIAdapter3* adapter = nullptr;

    bool open(const char luid[8]) {
        if(adapter) return true;
        IDXGIFactory4* factory = nullptr;
        if(FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&factory))) return false;
        IDXGIAdapter1* candidate = nullptr;
        for(UINT i = 0; !adapter && factory->EnumAdapters1(i, &candidate) != DXGI_ERROR_NOT_FOUND; i++){
            DXGI_ADAPTER_DESC1 desc = {};
            if(SUCCEEDED(candidate->GetDesc1(&desc)) && memcmp(&desc.AdapterLuid, luid, 8) == 0) candidate->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&adapter);
            candidate->Release();
        }
        factory->Release();
        return adapter != nullptr;
    }

    bool read(uint64_t& budget, uint64_t& usage) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
        if(!adapter || FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) return false;
        budget = info.Budget;
        usage = info.CurrentUsage;
        return true;
    }

    // la ram que la tarjeta puede ver (la memoria fijada y mapeada cuenta aca), windows la limita a la mitad de la ram
    bool readShared(uint64_t& budget, uint64_t& usage) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
        if(!adapter || FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) return false;
        budget = info.Budget;
        usage = info.CurrentUsage;
        return true;
    }

    void close() {
        if(adapter) adapter->Release();
        adapter = nullptr;
    }
};
#endif

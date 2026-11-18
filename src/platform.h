#pragma once
// lo que cambia entre sistemas: en windows sus encabezados de siempre y en linux las mismas llamadas hechas con lo que trae linux
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <psapi.h>
#include <intrin.h>
#else
#include "linuxshim.h"
#endif

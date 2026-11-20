#include "src/net.h"
#include "src/json.h"
#include "src/gguf.h"
#include "src/pool.h"
#include "src/residency.h"
#include "src/brain.h"
#include "src/tokenizer.h"
#include "src/chat.h"
#include "src/serve.h"
#include "src/messages.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <deque>
#include <set>
#include <random>
#include <ctime>
#include <psapi.h>

#pragma comment(lib, "psapi.lib")

namespace fs = std::filesystem;

struct RemoteNode {
    std::string name;
    std::string ip;
    int port = 9090;
    SOCKET wire = INVALID_SOCKET;
    std::mutex wireLock;
    NodeStats stats = {};
    bool connected = false;
    bool accepted = false;
    uint32_t pingMs = 0;
    std::string problem;
    std::string transfer = "idle";
    uint64_t transferTotal = 0;
    uint64_t transferDone = 0;
};

struct TrustedNode {
    std::string name;
    std::string ip;
    int port = 9090;
    std::string key;
};

struct JoinRequest {
    std::string id;
    std::string ip;
    std::string name;
    int tcpPort = 9090;
    std::string token;
    std::string gpu;
    uint64_t vramTotal = 0;
    uint64_t ramTotal = 0;
    std::string cpu;
    std::string status = "pending";
    std::string problem;
    uint64_t when = 0;
};

struct DoorVisitor {
    std::deque<uint64_t> hits;
    std::deque<uint64_t> joins;
};

struct RequestRecord {
    int64_t when = 0;
    int status = 0;
    uint64_t promptTokens = 0;
    uint64_t outputTokens = 0;
    uint64_t cachedTokens = 0;
    double promptSpeed = 0;
    double outputSpeed = 0;
    uint64_t durationMs = 0;
};

struct Engine {
    std::mutex lock;
    std::string selfName;
    std::string modelFolder = "modelos";
    std::string modelPath;
    std::string savedModel;
    std::string savedFolder;
    int selfScreen = -1;
    uint64_t diskExperts = 0;
    uint64_t ramExperts = 0;
    uint64_t shareBytes = 0;
    uint64_t vramHeld = 0;
    std::string modelId;
    ModelFile model;
    ModelShape shape;
    bool loaded = false;
    std::string loadError;
    uint32_t context = 131072;
    std::string backend = "auto";
    bool speculate = true;
    bool fastRouting = true;
    bool foresee = false;
    int mainGpu = -1;
    bool secondGpu = true;
    int helperGpu = -1;
    double helperGb = 0;
    bool helperFirst = false;
    int frontCap = 0;
    std::vector<GpuCard> gpus;
    std::string folderArg = "modelos";
    int draftLength = 3;
    int copyReach = 8;
    std::string cpuPlan = "all";
    int warmShare = 2;
    bool holdRam = true;
    bool ramWhole = false;
    int pinShare = 50;
    double pinGb = 0;
    bool vision = false;
    int visionGpu = -1;
    double draftSure = 0.6;
    std::string running = "off";
    std::vector<PoolSlot> slots;
    PoolPlan plan;
    std::string frontNode;
    int frontEnd = 0;
    std::string notice;
    uint64_t healedAt = 0;
    uint64_t healedMb = 0;
    uint64_t oomKeep = 0;
    std::map<std::string, std::shared_ptr<RemoteNode>> nodes;
    std::map<std::string, std::string> refused;
    std::vector<TrustedNode> trusted;
    std::vector<JoinRequest> joins;
    std::map<std::string, DoorVisitor> visitors;
    bool acceptRequests = true;
    int doorPort = 9097;
    bool doorOpen = false;
    std::string doorProblem;
    std::string apiListen = "local";
    std::string apiBound = "local";
    int apiPort = 8000;
    NodeStats selfStats = {};
    std::vector<std::pair<void*, GpuStats>> moreGpus;
    std::deque<RequestRecord> requests;
    int activeRequests = 0;
    int backendPort = 0;
    std::string poolFile = "pool.json";
};

Engine engine;
Residency local;
Brain brain;
FrontLink frontLink;
std::atomic<bool> splitDirty{false};
std::string splitApplied;
std::vector<int> boundary;
std::vector<int> turnBoundary;
std::unique_ptr<Tokenizer> words;
std::mutex thinking;
std::atomic<int> brainState{0};
std::atomic<bool> membershipDirty{false};
std::atomic<bool> holdSplit{false};
std::string brainProblem;
std::atomic<int> doorGuests{0};

const char* brainName() {
    int state = brainState;
    return state == 2
        ? "ready"
        : state == 1
            ? "loading"
            : state < 0
                ? "failed"
                : "off";
}

uint64_t selfRamLimit() {
    std::lock_guard<std::mutex> guard(engine.lock);
    uint64_t limit = 0;
    for(const PoolSlot& s : engine.plan.slots){
        if(s.node == engine.selfName && s.kind == SLOT_RAM) limit += s.limit;
    }
    return limit;
}

void applyMemoryCap() {
    static uint64_t applied = 0;
    uint64_t limit = std::max<uint64_t>(selfRamLimit(), 2ULL << 30);
    if(brainState != 2 || !brain.cpu) return;
    uint64_t change = limit > applied ? limit - applied : applied - limit;
    if(change < (512ULL << 20)) return;
    SetProcessWorkingSetSizeEx(GetCurrentProcess(), 256ULL << 20, (SIZE_T)limit, QUOTA_LIMITS_HARDWS_MAX_ENABLE | QUOTA_LIMITS_HARDWS_MIN_DISABLE);
    brain.cpu->memoryCap = limit;
    applied = limit;
    std::cout << "generation keeps at most " << (double)limit / gib << " GB of the model in ram, the rest is read from ssd" << std::endl;
}

uint64_t selfGpuLimit() {
    std::lock_guard<std::mutex> guard(engine.lock);
    uint64_t limit = 0;
    for(const PoolSlot& s : engine.plan.slots){
        if(s.node == engine.selfName && s.kind == SLOT_GPU) limit += s.limit;
    }
    return limit;
}

void closeFront() {
    if(brain.gpu) brain.gpu->remote = nullptr;
    if(frontLink.wire == INVALID_SOCKET) return;
    sendPacket(frontLink.wire, PACKET_BYE, nullptr, 0);
    closesocket(frontLink.wire);
    frontLink.wire = INVALID_SOCKET;
}

bool openFront(const std::string& name, const std::string& ip, int port, const std::string& key, int end, std::string& why) {
    SOCKET s = connectTcp(ip, port, 5000);
    if(s == INVALID_SOCKET){
        why = "nothing answers on " + ip + ":" + std::to_string(port);
        return false;
    }
    DWORD timeout = 15 * 60 * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
    HelloBody hello = {};
    strncpy_s(hello.key, key.c_str(), _TRUNCATE);
    hello.version = protocolVersion;
    hello.pad = 1;
    PacketHead head;
    std::vector<char> body;
    std::cout << "waiting for " << name << " to load layers 0.." << end - 1 << std::endl;
    if(!sendPacket(s, PACKET_HELLO, &hello, sizeof(hello)) || !readPacket(s, head, body, std::max<uint64_t>(sizeof(NodeStats), 4096))){
        closesocket(s);
        why = name + " closed the connection";
        return false;
    }
    if(head.kind == PACKET_REFUSED){
        why.assign(body.begin(), body.end());
        closesocket(s);
        return false;
    }
    int32_t range[2] = {-1, -1};
    if(head.kind != PACKET_READY || body.size() < 8){
        closesocket(s);
        why = name + " answered with something else";
        return false;
    }
    memcpy(range, body.data(), 8);
    if(range[0] != 0 || range[1] != end){
        closesocket(s);
        why = name + " holds layers " + std::to_string(range[0]) + ".." + std::to_string(range[1] - 1) + " instead of 0.." + std::to_string(end - 1);
        return false;
    }
    DWORD forever = 0;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (char*)&forever, sizeof(forever));
    // una tarjeta anterior a las RTX 30 verifica a lo sumo 5 filas por pasada y no podia guardar pasadas mas largas,
    // asi que el nodo dice su limite (un nodo viejo no lo dice y se usa 5)
    int32_t nodeLimit = 5;
    if(body.size() >= 12) memcpy(&nodeLimit, body.data() + 8, 4);
    frontLink.relayLimit = std::max(2, std::min(33, (int)nodeLimit));
    // un worker nuevo dice que formatos de filas entiende, uno viejo no dice nada y recibe las filas como siempre
    uint32_t features = 0;
    if(body.size() >= 16) memcpy(&features, body.data() + 12, 4);
    char maskText[16] = {};
    uint32_t mask = GetEnvironmentVariableA("SNEO_ROWS_FEATURES", maskText, sizeof(maskText)) > 0 ? (uint32_t)strtoul(maskText, nullptr, 10) : rowsFeatures;
    frontLink.features = features & rowsFeatures & mask;
    frontLink.wire = s;
    frontLink.node = name;
    frontLink.problem.clear();
    frontLink.roundMs = 0;
    frontLink.rounds = 0;
    frontLink.sentBytes = 0;
    frontLink.gotBytes = 0;
    return true;
}

struct Preparing {
    std::mutex lock;
    std::deque<std::string> queue;
    std::set<std::string> tried;
    std::string task;
    std::string problem;
    std::string note;
    double done = 0;
    double total = 0;
    bool running = false;
} preparing;

std::string modelsHome() {
    return (fs::path(runtimeFolder()).parent_path() / "modelos").string();
}

void preparingSay(const std::string& task, double done, double total) {
    std::lock_guard<std::mutex> guard(preparing.lock);
    preparing.task = task;
    preparing.done = done;
    preparing.total = total;
}

void preparingEnd(const std::string& problem, const std::string& note) {
    std::lock_guard<std::mutex> guard(preparing.lock);
    preparing.problem = problem;
    preparing.note = note;
    preparing.task.clear();
    std::cout << (problem.empty() ? note : problem) << std::endl;
}

bool runTool(const std::string& tool, const std::string& arg, const std::function<void(const std::string&)>& each) {
    std::string exe = (fs::path(runtimeFolder()).parent_path() / "tools" / tool).string();
    std::string command = "\"\"" + exe + "\" \"" + arg + "\" 2>&1\"";
    FILE* pipe = _popen(command.c_str(), "r");
    if(!pipe) return false;
    char line[1024];
    while(fgets(line, sizeof(line), pipe)) each(line);
    return _pclose(pipe) == 0;
}

void buildDraftLayer() {
    std::string source = (fs::path(modelsHome()) / "Qwen3.8-Flash-Next.draft.gguf").string();
    std::error_code e;
    fs::create_directories(modelsHome(), e);
    fs::space_info space = fs::space(modelsHome(), e);
    if(e){
        preparingEnd("cannot check the free disk in " + modelsHome() + ": " + e.message(), "");
        return;
    }
    if(space.available < (8ULL << 30)){
        preparingEnd("not enough free disk in " + modelsHome() + " to download the draft layer: it needs about 8 GB and " + std::to_string(space.available >> 30) + " GB are free", "");
        return;
    }
    std::cout << "downloading the draft layer for speculative decoding" << std::endl;
    preparingSay("draft", 0, 0);
    bool fetched = runTool("fetchdraft.exe", source, [](const std::string& line){
        double got = 0, want = 0;
        if(sscanf_s(line.c_str(), "%lf of %lf GB", &got, &want) == 2) preparingSay("draft", got, want);
    });
    if(!fetched){
        preparingEnd("downloading the draft layer failed, check the internet connection", "");
        return;
    }
    preparingSay("draft build", 0, 0);
    if(!runTool("draftbuild.exe", modelsHome(), [](const std::string&){})){
        preparingEnd("building the draft layer failed", "");
        return;
    }
    fs::remove(source, e);
    preparingEnd("", "Draft layer ready, it is used the next time the model loads");
}

void buildExpertPack(const std::string& modelPath) {
    ModelFile model;
    std::string error;
    if(!openModel(modelPath, model, error)){
        preparingEnd("cannot read the model to pack its experts: " + error, "");
        return;
    }
    uint64_t bytes = 0;
    for(const TensorInfo& t : model.tensors){
        if(t.expert) bytes += t.bytes;
    }
    std::string target = (fs::path(modelsHome()) / fs::path(packPathFor(modelPath)).filename()).string();
    std::error_code e;
    fs::create_directories(modelsHome(), e);
    fs::space_info space = fs::space(modelsHome(), e);
    if(e){
        preparingEnd("cannot check the free disk in " + modelsHome() + ": " + e.message(), "");
        return;
    }
    if(space.available < bytes + (10ULL << 30)){
        preparingEnd("not enough free disk in " + modelsHome() + " for the expert pack: it needs " + std::to_string((bytes >> 30) + 10) + " GB and " + std::to_string(space.available >> 30) + " GB are free", "");
        return;
    }
    std::cout << "packing the experts of " << fs::path(modelPath).filename().string() << " for faster reads from disk" << std::endl;
    preparingSay("pack", 0, model.layers);
    bool ok = buildPack(model, target, error, [](int done, int total){
        preparingSay("pack", done, total);
    });
    if(!ok){
        preparingEnd("packing the experts failed: " + error, "");
        return;
    }
    preparingEnd("", "Expert pack ready, it is used the next time the model loads");
}

void loadBrain(std::string path, uint64_t ticket);
uint64_t nextLoad();

// lo preparado se usa al cargar asi que si el modelo que lo pidio sigue cargado se carga otra vez cuando termina de responder
void reloadPrepared(const std::string& job) {
    std::unique_lock<std::mutex> busy(thinking);
    std::string path, front;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        path = engine.modelPath;
        front = engine.frontNode;
    }
    if(path.empty() || brainState != 2 || !brain.gpu || !front.empty() || brain.firstLayer > 0) return;
    if(job == "draft" ? brain.gpu->file.arch != "qwen4exp" : job.substr(5) != path) return;
    std::cout << (job == "draft" ? "draft layer ready, loading the model again to use it" : "expert pack ready, loading the model again to use it") << std::endl;
    busy.unlock();
    std::thread(loadBrain, path, nextLoad()).detach();
}

void prepareLoop() {
    SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
    while(true){
        std::string job;
        {
            std::lock_guard<std::mutex> guard(preparing.lock);
            if(preparing.queue.empty()){
                preparing.running = false;
                return;
            }
            job = preparing.queue.front();
            preparing.queue.pop_front();
        }
        if(job == "draft") buildDraftLayer();
        else buildExpertPack(job.substr(5));
        bool failed;
        {
            std::lock_guard<std::mutex> guard(preparing.lock);
            failed = !preparing.problem.empty();
        }
        if(!failed) reloadPrepared(job);
    }
}

void queuePrepare(const std::string& job) {
    std::lock_guard<std::mutex> guard(preparing.lock);
    if(preparing.tried.count(job)) return;
    preparing.tried.insert(job);
    preparing.queue.push_back(job);
    if(preparing.running) return;
    preparing.running = true;
    std::thread(prepareLoop).detach();
}

void pushWeights();

std::atomic<uint64_t> loadTicket{0};

uint64_t nextLoad() {
    if(brainState == 1) stopLoading = true;
    return ++loadTicket;
}

void settleMemory() {
    NvmlLink& nv = nvmlLink();
    if(!nv.ready || !nv.getMemory) return;
    uint64_t start = nowMs();
    unsigned long long last = 0;
    int steady = 0;
    while(nowMs() - start < 15000 && steady < 4){
        unsigned long long memory[3] = {0, 0, 0};
        if(nv.getMemory(nv.device, memory) != 0) return;
        steady = memory[1] < last + (64ULL << 20) ? steady + 1 : 0;
        last = std::max(last, memory[1]);
        Sleep(100);
    }
    if(nowMs() - start > 1000) std::cout << "the gpu took " << (nowMs() - start) / 1000.0 << " s to free its memory" << std::endl;
}

void releaseNodes() {
    std::vector<std::shared_ptr<RemoteNode>> list;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        for(auto& entry : engine.nodes){
            if(entry.second->connected && entry.second->accepted) list.push_back(entry.second);
        }
    }
    for(auto& node : list){
        std::lock_guard<std::mutex> wire(node->wireLock);
        sendPacket(node->wire, PACKET_RESET, nullptr, 0);
    }
}

void restartEngine(const std::string& path) {
    NvmlLink& nv = nvmlLink();
    unsigned long long memory[3] = {0, 0, 0};
    if(nv.ready && nv.getMemory) nv.getMemory(nv.device, memory);
    std::cout << "the gpu could not be opened again in this process, " << (double)memory[1] / gib << " GB free of " << (double)memory[0] / gib << " GB" << std::endl;
    char already[8] = {};
    if(GetEnvironmentVariableA("SNEO_RESTARTED", already, sizeof(already)) > 0){
        std::cout << "the engine already restarted itself for this, close and open Sneo" << std::endl;
        return;
    }
    char exe[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, exe, MAX_PATH);
    // el segundo motor vuelve con su propia configuracion y los dos con el margen que ya aprendieron
    std::string command = "\"" + std::string(exe) + "\" " + std::to_string(engine.apiPort) + " - - \"" + engine.folderArg + "\" \"" + path + "\""
        + (engine.poolFile != "pool.json" ? " --pool \"" + engine.poolFile + "\"" : std::string());
    SetEnvironmentVariableA("SNEO_RESTARTED", "1");
    SetEnvironmentVariableA("SNEO_OOM_KEEP", std::to_string(engine.oomKeep).c_str());
    // el motor nuevo espera a que este salga y suelte su ram fijada antes de repartir el modelo
    SetEnvironmentVariableA("SNEO_PARENT", std::to_string(GetCurrentProcessId()).c_str());

    std::vector<HANDLE> pass;
    for(HANDLE h : {GetStdHandle(STD_OUTPUT_HANDLE), GetStdHandle(STD_ERROR_HANDLE)}){
        if(!h || h == INVALID_HANDLE_VALUE || std::find(pass.begin(), pass.end(), h) != pass.end()) continue;
        if(SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT)) pass.push_back(h);
    }
    SIZE_T size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    std::vector<char> room(size);
    auto list = (LPPROC_THREAD_ATTRIBUTE_LIST)room.data();
    bool listed = !pass.empty() && InitializeProcThreadAttributeList(list, 1, 0, &size)
        && UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, pass.data(), pass.size() * sizeof(HANDLE), nullptr, nullptr);

    STARTUPINFOEXA start = {};
    start.StartupInfo.cb = listed ? sizeof(start) : sizeof(STARTUPINFOA);
    if(listed){
        start.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        start.StartupInfo.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        start.StartupInfo.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        start.lpAttributeList = list;
    }
    PROCESS_INFORMATION info = {};
    BOOL made = CreateProcessA(nullptr, command.data(), nullptr, nullptr, listed, CREATE_NO_WINDOW | (listed ? EXTENDED_STARTUPINFO_PRESENT : 0), nullptr, nullptr, &start.StartupInfo, &info);
    if(listed) DeleteProcThreadAttributeList(list);
    if(!made){
        std::cout << "the engine could not restart itself (error " << GetLastError() << "), close and open Sneo" << std::endl;
        return;
    }
    CloseHandle(info.hThread);
    CloseHandle(info.hProcess);
    std::cout << "the engine restarts itself and loads " << fs::path(path).filename().string() << std::endl;
    std::cout.flush();
    ExitProcess(0);
}

void releaseBrain() {
    brain.cpu.reset();
    brain.gpu.reset();
    brain.kind = "off";
    boundary.clear();
    turnBoundary.clear();
    brainState = 0;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        engine.vramHeld = 0;
        engine.running = "off";
    }
    settleMemory();
}

void loadBrain(std::string path, uint64_t ticket) {
    std::unique_lock<std::mutex> busy(thinking);
    if(ticket != loadTicket) return;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        if(!engine.modelPath.empty() && engine.modelPath != path){
            std::cout << "skipping " << fs::path(path).filename().string() << ", another model was picked meanwhile" << std::endl;
            return;
        }
    }
    stopLoading = false;
    loadSay("Opening the model");
    if(brain.ready()) releaseBrain();
    brainState = 1;
    uint64_t start = nowMs();
    std::cout << "preparing generation with " << path << std::endl;

    ModelFile file;
    std::string error;
    auto nextWords = std::make_unique<Tokenizer>();
    if(!openModel(path, file, error) || !nextWords->load(file, error)){
        brainProblem = error;
        brainState = -1;
        std::cout << "tokenizer failed: " << error << std::endl;
        return;
    }

    std::string mode;
    int context = 0;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        mode = engine.backend;
        context = (int)engine.context;
    }
    std::string frontNode, frontIp, frontKey;
    int frontEnd = 0, frontPort = 0;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        frontNode = engine.frontNode;
        frontEnd = engine.frontEnd;
        splitApplied = frontNode.empty() ? "" : frontNode + ":" + std::to_string(frontEnd);
        if(!frontNode.empty()) std::cout << "the pool node " << frontNode << " takes layers 0.." << frontEnd - 1 << ", sending what it does not have yet" << std::endl;
        auto found = engine.nodes.find(frontNode);
        if(found != engine.nodes.end()){
            frontIp = found->second->ip;
            frontPort = found->second->port;
        }
        for(const TrustedNode& t : engine.trusted){
            if(t.name == frontNode) frontKey = t.key;
        }
        if(frontIp.empty() || frontKey.empty() || mode == "cpu") frontNode.clear();
    }
    closeFront();
    if(!frontNode.empty()) pushWeights();
    else releaseNodes();
    if(stopLoading){
        brainState = 0;
        std::cout << "loading stopped" << std::endl;
        return;
    }

    int threads = (int)std::max<uint32_t>(4, engine.selfStats.host.cores);
    boundary.clear();
    turnBoundary.clear();
    brain.wantDraft = engine.speculate;
    brain.draftLength = engine.draftLength;
    brain.copyReach = engine.copyReach;
    brain.cpuPlan = engine.cpuPlan;
    brain.warmShare = engine.warmShare;
    brain.foresee = engine.foresee;
    brain.holdRam = engine.holdRam;
    brain.ramWhole = engine.ramWhole;
    brain.pinShare = engine.pinShare;
    brain.pinGb = engine.pinGb;
    brain.vision = engine.vision;
    brain.visionGpu = engine.visionGpu;
    brain.firstLayer = frontNode.empty() ? 0 : frontEnd;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        brain.mainGpu = engine.mainGpu;
        brain.useHelper = engine.secondGpu;
        brain.helperGpu = engine.helperGpu;
        brain.helperGb = engine.helperGb;
        brain.helperFirst = engine.helperFirst;
        brain.extraKeep = engine.oomKeep;
    }
    {
        std::lock_guard<std::mutex> held(local.lock);
        brain.freshGpu = local.gpuBytes == 0;
    }
    // un nodo en esta misma pc comparte la tarjeta: carga primero sus capas, si cargaran juntos contaria como suyo
    // lo que el master va tomando y se quedaria sin expertos en la vram (le paso al probar el pool en una sola pc)
    bool sameCard = !frontNode.empty() && frontIp.rfind("127.", 0) == 0;
    std::string frontProblem;
    bool frontOpen = sameCard && openFront(frontNode, frontIp, frontPort, frontKey, frontEnd, frontProblem);
    bool loaded = (!sameCard || frontOpen) && brain.load(path, mode, threads, selfGpuLimit(), selfRamLimit(), context, "");
    if(!loaded && frontProblem.empty()) frontProblem = brain.problem;
    if(loaded && brain.firstLayer > 0){
        loaded = brain.gpu && (frontOpen || openFront(frontNode, frontIp, frontPort, frontKey, frontEnd, frontProblem));
        if(loaded){
            brain.gpu->remote = &frontLink;
            brain.gpu->reset();
        }
    }
    if(stopLoading){
        brain.cpu.reset();
        brain.gpu.reset();
        brain.kind = "off";
        brainState = 0;
        std::cout << "loading stopped" << std::endl;
        return;
    }
    if(!loaded && brain.firstLayer > 0 && !brain.gpuLost){
        if(brain.gpu || (sameCard && !frontOpen)) std::cout << "pool node " << frontNode << " cannot take layers 0.." << frontEnd - 1 << ": " << frontProblem << ", running the whole model on this pc" << std::endl;
        else std::cout << "this pc could not load layers " << frontEnd << ".." << engine.model.layers - 1 << " for the pool: " << frontProblem << ", running the whole model on this pc" << std::endl;
        closeFront();
        releaseNodes();
        brain.firstLayer = 0;
        loaded = brain.load(path, mode, threads, selfGpuLimit(), selfRamLimit(), context, "");
    }
    if(!loaded && brain.gpuLost){
        std::cout << "loading on the gpu failed: " << brain.problem << std::endl;
        closeFront();
        releaseNodes();
        brainState = -1;
        brainProblem = brain.problem;
        restartEngine(path);
        return;
    }
    if(loaded) SetEnvironmentVariableA("SNEO_RESTARTED", nullptr);
    if(!loaded){
        brainProblem = brain.problem;
        brainState = -1;
        std::cout << "model failed: " << brainProblem << std::endl;
        return;
    }

    words = std::move(nextWords);
    brainProblem.clear();
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        engine.running = brain.kind;
        engine.vramHeld = brain.gpu ? brain.gpu->vramHeld : 0;
    }
    brainState = 2;
    std::cout << "generation ready on the " << brain.kind << " in " << (nowMs() - start) / 1000.0 << " s, ram budget " << (double)brain.ramBudget / gib << " GB" << std::endl;
    if(!brain.gpuProblem.empty() && !brain.gpu) std::cout << "gpu load failed, running on the cpu instead: " << brain.gpuProblem << std::endl;
    if(!brain.gpuProblem.empty() && brain.gpu) std::cout << "gpu: " << brain.gpuProblem << std::endl;
    if(brain.gpu){
        brain.gpu->tailSkip = engine.fastRouting ? 7 : 0;
        brain.gpu->foresee = engine.foresee;
        brain.gpu->draftSure = (float)engine.draftSure;
        std::cout << "routing: " << (!engine.fastRouting ? "exact" : brain.gpu->hy4 ? "fast, missed experts ranked 9 to 16 are skipped" : "fast, missed experts ranked 8 to 10 are skipped") << std::endl;
        std::cout << "draft layer: " << (brain.gpu->drafting ? (brain.gpu->tiedDraft ? "built into the model" : "loaded") : brain.gpu->draftProblem.empty() ? "not found" : brain.gpu->draftProblem) << std::endl;
        if(brain.gpu->legacy) std::cout << "gpu: no Ampere tensor cores, prompts run on the dp4a tile kernels" << std::endl;
    }
    if(brain.gpu && brain.gpu->dense) std::cout << (brain.gpu->denseHostBytes
        ? "experts: none, the dense model runs on the gpu and reads " + std::to_string(brain.gpu->denseHostBytes >> 20) + " MB of its layers from pinned ram"
        : std::string("experts: none, the dense model runs whole on the gpu")) << std::endl;
    if(brain.gpu && brain.gpu->seeing) std::cout << "vision: the picture reader runs on " << brain.gpu->visionCard << std::endl;
    if(brain.gpu && !brain.gpu->seeing && !brain.gpu->visionProblem.empty()) std::cout << "vision: " << brain.gpu->visionProblem << std::endl;
    if(brain.gpu && !brain.gpu->dense) std::cout << "experts: " << brain.gpu->residentExperts << " per layer in vram, " << brain.gpu->warmPerLayer << " per layer pinned in ram (" << (double)brain.gpu->warmPinnedBytes / gib << " GB), placed by "
        << (brain.gpu->heatSource.empty() ? std::string("no heat map yet, it learns while you use it") : fs::path(brain.gpu->heatSource).filename().string()) << std::endl;
    if(brain.gpu && brain.gpu->ramWhole) std::cout << "whole model in ram: " << (double)brain.gpu->wholeBytes / gib << " GB more of the model file is read and held in ram in the background" << std::endl;
    if(brain.gpu && brain.gpu->helper.ready) std::cout << "second gpu: " << brain.gpu->helper.name << " holds " << brain.gpu->helper.perLayer << " more experts per layer (" << (double)brain.gpu->helper.bytes / gib << " GB) and computes them beside the cpu" << std::endl;
    if(brain.gpu && !brain.gpu->helper.ready && !brain.gpu->helper.problem.empty()) std::cout << "second gpu: " << brain.gpu->helper.problem << std::endl;
    uint64_t expertBytes = 0;
    uint64_t shareBytes = 0;
    uint32_t expertCount = 0;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        for(const TensorInfo& t : engine.model.tensors){
            if(t.layer >= 0 && t.layer < brain.firstLayer) continue;
            shareBytes += t.bytes;
            if(t.expert) expertBytes += t.bytes;
        }
        expertCount = engine.model.experts;
    }
    uint64_t held = brain.gpu ? brain.gpu->residentBytes + brain.gpu->warmPinnedBytes + brain.gpu->warmBudget + (brain.gpu->helper.ready ? brain.gpu->helper.bytes : 0) : 0;
    bool fits = expertBytes <= held;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        engine.diskExperts = brain.gpu && !fits ? expertBytes - held : 0;
        engine.shareBytes = brain.gpu ? shareBytes : 0;
        // hy4 lee de la ram los expertos que no estan en la vram, por la cache de windows del archivo mapeado
        engine.ramExperts = brain.gpu && brain.gpu->hy4 && !fits ? expertBytes - held : 0;
        if(brain.gpu && brain.gpu->hy4) engine.diskExperts = 0;
    }
    if(brain.gpu && brain.firstLayer == 0){
        std::error_code e;
        bool draftHere = fs::exists(fs::path(modelsHome()) / "Qwen3.8-Flash-Next.draft-q2.gguf", e) || fs::exists(fs::path(path).parent_path() / "Qwen3.8-Flash-Next.draft-q2.gguf", e);
        if(!brain.gpu->packed && !fits && !brain.gpu->hy4) queuePrepare("pack:" + path);
        if(!draftHere && expertCount == 512 && engine.speculate && !brain.gpu->hy4) queuePrepare("draft");
    }
    if(brain.gpu && brain.gpu->remote) std::cout << "layers 0.." << brain.firstLayer - 1 << " run on " << frontNode << ", layers " << brain.firstLayer << ".." << brain.gpu->layers.size() - 1 << " on this pc" << std::endl;
    busy.unlock();
    applyMemoryCap();
}

const char* kindName(int kind) {
    return kind == SLOT_GPU ? "gpu" : kind == SLOT_RAM ? "ram" : "ssd";
}

int kindValue(const std::string& name) {
    return name == "gpu" ? SLOT_GPU : name == "ram" ? SLOT_RAM : SLOT_SSD;
}

std::string makeKey() {
    const char* letters = "abcdefghijkmnpqrstuvwxyz23456789";
    std::random_device seed;
    std::string key;
    for(int i = 0; i < 16; i++) key.push_back(letters[seed() % 32]);
    return key;
}

std::string hashText(const std::string& text) {
    uint64_t h = 1469598103934665603ULL;
    for(unsigned char c : text){
        h ^= c;
        h *= 1099511628211ULL;
    }
    char out[24];
    snprintf(out, sizeof(out), "%016llx", (unsigned long long)h);
    return out;
}

void savePool() {
    std::ostringstream out;
    out << "{\"backend\":" << quoteJson(engine.backend) << ",\"speculate\":" << (engine.speculate ? "true" : "false") << ",\"fastRouting\":" << (engine.fastRouting ? "true" : "false") << ",\"foresee\":" << (engine.foresee ? "true" : "false") << ",\"mainGpu\":" << engine.mainGpu << ",\"secondGpu\":" << (engine.secondGpu ? "true" : "false") << ",\"helperGpu\":" << engine.helperGpu << ",\"helperGb\":" << engine.helperGb << ",\"helperFirst\":" << (engine.helperFirst ? "true" : "false") << ",\"frontLayers\":" << engine.frontCap << ",\"draftLength\":" << engine.draftLength << ",\"copyReach\":" << engine.copyReach << ",\"cpuPlan\":" << quoteJson(engine.cpuPlan) << ",\"warmShare\":" << engine.warmShare << ",\"tuned\":1,\"holdRam\":" << (engine.holdRam ? "true" : "false") << ",\"ramWhole\":" << (engine.ramWhole ? "true" : "false") << ",\"pinShare\":" << engine.pinShare << ",\"pinGb\":" << engine.pinGb << ",\"vision\":" << (engine.vision ? "true" : "false") << ",\"visionGpu\":" << engine.visionGpu << ",\"draftSure\":" << engine.draftSure << ",\"context\":" << engine.context << ",\"acceptRequests\":" << (engine.acceptRequests ? "true" : "false") << ",\"doorPort\":" << engine.doorPort
        << ",\"apiListen\":" << quoteJson(engine.apiListen) << ",\"model\":" << quoteJson(engine.modelPath.empty() ? engine.savedModel : engine.modelPath) << ",\"folder\":" << quoteJson(engine.modelFolder) << ",\"slots\":[";
    for(size_t i = 0; i < engine.slots.size(); i++){
        const PoolSlot& s = engine.slots[i];
        out << (i ? "," : "") << "{\"node\":" << quoteJson(s.node) << ",\"kind\":\"" << kindName(s.kind) << "\",\"gb\":" << s.gb << ",\"percent\":" << s.percent << ",\"keep\":" << s.keep << "}";
    }
    out << "],\"trusted\":[";
    for(size_t i = 0; i < engine.trusted.size(); i++){
        const TrustedNode& t = engine.trusted[i];
        out << (i ? "," : "") << "{\"name\":" << quoteJson(t.name) << ",\"ip\":" << quoteJson(t.ip) << ",\"port\":" << t.port << ",\"key\":" << quoteJson(t.key) << "}";
    }
    out << "]}";
    std::ofstream(engine.poolFile, std::ios::trunc) << out.str();
}

void loadPool() {
    std::ifstream file(engine.poolFile);
    std::stringstream text;
    text << file.rdbuf();
    Json saved = parseJson(text.str());

    engine.context = (uint32_t)saved["context"].num(131072);
    engine.backend = saved["backend"].str("auto");
    if(engine.backend != "gpu" && engine.backend != "cpu") engine.backend = "auto";
    engine.acceptRequests = saved["acceptRequests"].truthy(true);
    engine.speculate = saved["speculate"].truthy(true);
    engine.mainGpu = (int)saved["mainGpu"].num(-1);
    engine.secondGpu = saved["secondGpu"].truthy(true);
    engine.helperGpu = (int)saved["helperGpu"].num(-1);
    engine.helperGb = std::max(0.0, saved["helperGb"].num(0));
    engine.helperFirst = saved["helperFirst"].truthy(false);
    engine.frontCap = std::max(0, (int)saved["frontLayers"].num(0));
    engine.fastRouting = saved["fastRouting"].truthy(true);
    engine.foresee = saved["foresee"].truthy(false);
    engine.draftLength = std::max(1, std::min(16, (int)saved["draftLength"].num(3)));
    engine.copyReach = std::max(0, std::min(32, (int)saved["copyReach"].num(8)));
    engine.cpuPlan = saved["cpuPlan"].str("all");
    if(engine.cpuPlan != "performance" && engine.cpuPlan != "performance-ht" && engine.cpuPlan != "auto") engine.cpuPlan = "all";
    engine.warmShare = std::max(0, std::min(8, (int)saved["warmShare"].num(2)));
    // el reparto por defecto era 4: medido de nuevo rinde mas con 2 y lo guardado con el valor viejo pasa al nuevo una sola vez
    if(saved["tuned"].num(0) < 1 && engine.warmShare == 4) engine.warmShare = 2;
    engine.holdRam = saved["holdRam"].truthy(true);
    engine.ramWhole = saved["ramWhole"].truthy(false);
    engine.pinShare = std::max(10, std::min(95, (int)saved["pinShare"].num(50)));
    engine.pinGb = std::max(0.0, std::min(4096.0, saved["pinGb"].num(0)));
    engine.vision = saved["vision"].truthy(false);
    engine.visionGpu = (int)saved["visionGpu"].num(-1);
    engine.draftSure = std::max(0.0, std::min(0.99, saved["draftSure"].num(0.6)));
    engine.doorPort = (int)saved["doorPort"].num(9097);
    engine.apiListen = saved["apiListen"].str() == "network" ? "network" : "local";
    engine.savedModel = saved["model"].str();
    engine.savedFolder = saved["folder"].str();
    std::error_code absent;
    if(!engine.savedFolder.empty() && fs::is_directory(engine.savedFolder, absent)) engine.modelFolder = engine.savedFolder;
    if(engine.doorPort < 1 || engine.doorPort > 65535) engine.doorPort = 9097;

    for(const Json& s : saved["slots"].items){
        PoolSlot slot;
        slot.node = s["node"].str();
        slot.kind = kindValue(s["kind"].str());
        slot.gb = s["gb"].num();
        slot.percent = s["percent"].num();
        slot.keep = s["keep"].num();
        engine.slots.push_back(slot);
    }

    for(const Json& t : saved["trusted"].items){
        engine.trusted.push_back({t["name"].str(), t["ip"].str(), (int)t["port"].num(9090), t["key"].str()});
    }

    if(engine.slots.empty()){
        PoolSlot gpu = {engine.selfName, SLOT_GPU, 0, 0};
        gpu.keep = engine.selfScreen == 0 ? 0 : 1;
        PoolSlot ram = {engine.selfName, SLOT_RAM, 0, 0};
        ram.keep = 6;
        engine.slots.push_back(gpu);
        engine.slots.push_back(ram);
        engine.slots.push_back({engine.selfName, SLOT_SSD, 64, 0});
    }

    for(PoolSlot& slot : engine.slots){
        if(slot.node == "this-pc") slot.node = engine.selfName;
    }

    savePool();
}

bool isTrusted(const std::string& name) {
    for(const TrustedNode& t : engine.trusted){
        if(t.name == name) return true;
    }
    return false;
}

void splitPlan();

void replan() {
    if(!engine.loaded){
        engine.plan = PoolPlan();
        engine.notice.clear();
        return;
    }

    std::map<std::string, NodeStats> stats;
    stats[engine.selfName] = engine.selfStats;
    std::vector<PoolSlot> usable;
    std::string skipped;

    for(const PoolSlot& slot : engine.slots){
        if(slot.node == engine.selfName){
            usable.push_back(slot);
            continue;
        }
        auto found = engine.nodes.find(slot.node);
        if(found != engine.nodes.end() && found->second->connected && found->second->accepted){
            stats[slot.node] = found->second->stats;
            usable.push_back(slot);
            continue;
        }
        if(skipped.find(slot.node) == std::string::npos) skipped += (skipped.empty() ? "" : ", ") + slot.node;
    }

    engine.plan = planPool(engine.shape, usable, stats, engine.selfName, engine.context);
    engine.notice = skipped.empty() ? "" : skipped + " is offline, its share moved to the other slots";
    splitPlan();
}

void splitPlan() {
    engine.frontNode.clear();
    engine.frontEnd = 0;
    if(!engine.plan.ok || engine.backend == "cpu" || !engine.selfStats.gpu.present) return;

    std::string node;
    for(const PoolSlot& s : engine.plan.slots){
        if(s.node == engine.selfName || s.kind != SLOT_GPU || s.limit == 0) continue;
        auto found = engine.nodes.find(s.node);
        if(found == engine.nodes.end() || !found->second->connected || !found->second->accepted || !found->second->stats.gpu.present) continue;
        node = s.node;
        break;
    }
    if(node.empty()) return;

    uint64_t gpuLimit = 0, gpuShare = 0, ramShare = 0, ramLimit = 0, selfGpu = 0;
    for(const PoolSlot& s : engine.plan.slots){
        if(s.node == node && s.kind == SLOT_GPU){
            gpuLimit += s.limit;
            gpuShare += s.used;
        }
        if(s.node == node && s.kind == SLOT_RAM){
            ramShare += s.used;
            ramLimit += s.limit;
        }
        if(s.node == engine.selfName && s.kind == SLOT_GPU) selfGpu += s.limit;
    }

    const ModelShape& shape = engine.shape;
    int count = (int)shape.dense.size();
    if(count < 4) return;
    // si el orden de prioridad no llega a sus slots el modelo entra en esta pc: pedirle capas solo agregaba la red a cada token
    if(gpuShare + ramShare == 0){
        std::cout << "the model fits before the slots of " << node << " in the priority, it runs on this pc" << std::endl;
        return;
    }
    uint64_t scratch = 5ULL << 28;
    uint64_t gpuRoom = gpuLimit > scratch ? gpuLimit - scratch : 0;
    uint64_t room = std::max(gpuRoom, gpuShare + ramShare);
    uint64_t used = 0, denseUsed = 0;
    int end = 0;
    while(end < count - 2){
        uint64_t dense = shape.dense[end] + shape.kvPerToken[end] * engine.context;
        uint64_t cost = dense + shape.experts[end];
        if(used + cost > room || denseUsed + dense > gpuRoom) break;
        used += cost;
        denseUsed += dense;
        end++;
    }
    if(engine.frontCap > 0) end = std::min(count - 2, std::max(2, engine.frontCap));
    // un nodo tiene que llevar al menos dos capas: si su tarjeta no las tiene libres se le pedian igual y se negaba a cargar
    if(end < 2){
        std::cout << "the pool node " << node << " has " << (double)gpuLimit / gib << " GB free on its card, not enough for two layers, the model runs on this pc" << std::endl;
        return;
    }
    engine.frontNode = node;
    engine.frontEnd = end;

    PoolPlan& plan = engine.plan;
    NodeShare front;
    front.node = node;
    front.firstLayer = 0;
    front.lastLayer = end - 1;
    NodeShare self;
    self.node = engine.selfName;
    self.firstLayer = end;
    self.lastLayer = count - 1;
    self.holdsEdges = true;
    self.headTier = SLOT_GPU;
    self.tableTier = SLOT_SSD;

    uint64_t selfDense = shape.headBytes;
    for(int l = end; l < count; l++) selfDense += shape.dense[l] + shape.kvPerToken[l] * engine.context;
    uint64_t selfExperts = selfGpu > selfDense + scratch ? selfGpu - selfDense - scratch : 0;
    uint64_t frontExperts = gpuRoom;
    for(int l = 0; l < end; l++) frontExperts = frontExperts > shape.dense[l] + shape.kvPerToken[l] * engine.context ? frontExperts - shape.dense[l] - shape.kvPerToken[l] * engine.context : 0;

    plan.layers.clear();
    for(int l = 0; l < count; l++){
        bool mine = l >= end;
        LayerShare item;
        item.layer = l;
        item.node = mine ? engine.selfName : node;
        item.denseBytes = shape.dense[l] + shape.kvPerToken[l] * engine.context;
        item.expertBytes = shape.experts[l];
        item.denseTier = SLOT_GPU;
        uint64_t each = shape.expertCount ? shape.experts[l] / shape.expertCount : 0;
        uint64_t pool = mine ? selfExperts / std::max(1, count - end) : frontExperts / std::max(1, end);
        int fit = each ? (int)std::min<uint64_t>(shape.expertCount, pool / each) : 0;
        item.gpuExperts = fit;
        item.ramExperts = (int)shape.expertCount - fit;
        item.ssdExperts = 0;
        (mine ? self.gpuBytes : front.gpuBytes) += item.denseBytes + (uint64_t)fit * each;
        (mine ? self.ramBytes : front.ramBytes) += (uint64_t)item.ramExperts * each;
        plan.layers.push_back(item);
    }
    front.ramBytes = std::min(front.ramBytes, ramLimit);
    plan.nodes = {front, self};
}

void dropNode(const std::shared_ptr<RemoteNode>& node, const std::string& why) {
    if(node->wire != INVALID_SOCKET) closesocket(node->wire);
    node->wire = INVALID_SOCKET;
    node->connected = false;
    node->problem = why;
    if(node->name == engine.frontNode) membershipDirty = true;
    if(node->accepted) replan();
}

std::string connectNode(std::string ip, int port, std::string key, std::string& why) {
    std::string address = ip + ":" + std::to_string(port);
    SOCKET s = connectTcp(ip, port, 3000);
    if(s == INVALID_SOCKET){
        why = "nothing answers on " + address + ", check that worker.exe is running and the firewall lets port " + std::to_string(port) + " in";
        return "";
    }

    DWORD timeout = 20000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));

    HelloBody hello = {};
    strncpy_s(hello.key, key.c_str(), _TRUNCATE);
    hello.version = protocolVersion;

    uint64_t start = nowMs();
    PacketHead head;
    std::vector<char> body;
    if(!sendPacket(s, PACKET_HELLO, &hello, sizeof(hello)) || !readPacket(s, head, body, std::max<uint64_t>(sizeof(NodeStats), 4096))){
        closesocket(s);
        why = address + " closed the connection";
        return "";
    }

    if(head.kind == PACKET_REFUSED){
        why.assign(body.begin(), body.end());
        std::lock_guard<std::mutex> guard(engine.lock);
        engine.refused[address] = why;
        closesocket(s);
        return "";
    }

    if(head.kind != PACKET_NODE_INFO || body.size() < sizeof(NodeStats)){
        closesocket(s);
        why = address + " is not a Sneo worker";
        return "";
    }

    NodeStats stats;
    memcpy(&stats, body.data(), sizeof(stats));
    sealStats(stats);
    std::string name = stats.host.nodeName;

    std::lock_guard<std::mutex> guard(engine.lock);
    engine.refused.erase(address);

    if(name == engine.selfName) name += " :" + std::to_string(port);

    std::shared_ptr<RemoteNode>& node = engine.nodes[name];
    if(!node) node = std::make_shared<RemoteNode>();
    if(node->connected){
        closesocket(s);
        why = "a machine named " + name + " is already connected";
        return "";
    }

    node->name = name;
    node->ip = ip;
    node->port = port;
    node->wire = s;
    node->stats = stats;
    node->connected = true;
    node->pingMs = (uint32_t)(nowMs() - start);
    node->problem.clear();
    node->accepted = isTrusted(name);

    std::cout << "node " << name << " at " << address << (node->accepted ? " rejoined the pool" : " connected") << std::endl;
    if(node->accepted) replan();
    return name;
}

void statsLoop() {
    while(true){
        NodeStats self = readNode(engine.modelFolder);
        std::vector<std::pair<void*, GpuStats>> others = readOtherGpus();
        residencyStats(local, self.host);
        PROCESS_MEMORY_COUNTERS memory = {};
        if(GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory))) self.host.heldRam = std::max<uint64_t>(self.host.heldRam, memory.WorkingSetSize);
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            self.host.heldGpu = std::max<uint64_t>(self.host.heldGpu, brainState == 2 ? engine.vramHeld : 0);
            if(brainState == 2) self.host.heldRam = std::max<uint64_t>(self.host.heldRam, engine.ramExperts);
        }
        bool watchRam = false;
        std::vector<std::shared_ptr<RemoteNode>> list;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            engine.selfStats = self;
            engine.moreGpus = others;
            for(auto& entry : engine.nodes){
                if(entry.second->connected) list.push_back(entry.second);
            }
            for(const PoolSlot& slot : engine.slots){
                if(slot.node == engine.selfName && slot.kind == SLOT_RAM && slot.keep > 0) watchRam = true;
            }
            if(watchRam) replan();
        }
        if(watchRam) applyMemoryCap();

        for(auto& node : list){
            std::unique_lock<std::mutex> wire(node->wireLock, std::try_to_lock);
            if(!wire.owns_lock()) continue;

            uint64_t start = nowMs();
            PacketHead head;
            std::vector<char> body;
            bool asked = sendPacket(node->wire, PACKET_STATS_ASK, nullptr, 0) && readPacket(node->wire, head, body, std::max<uint64_t>(sizeof(NodeStats), 4096));
            bool ok = asked && head.kind == PACKET_STATS && body.size() >= sizeof(NodeStats);

            std::lock_guard<std::mutex> guard(engine.lock);
            if(asked && head.kind == PACKET_BYE){
                std::cout << "node " << node->name << " disconnected from the pool, the model is unloaded on both machines" << std::endl;
                std::string name = node->name;
                engine.trusted.erase(std::remove_if(engine.trusted.begin(), engine.trusted.end(), [&](const TrustedNode& t){ return t.name == name; }), engine.trusted.end());
                dropNode(node, "disconnected from the pool");
                engine.notice = name + " disconnected from the pool, the model was unloaded on both PCs";
                savePool();
                holdSplit = true;
                std::thread([]{
                    stopGenerating = true;
                    nextLoad();
                    {
                        std::unique_lock<std::mutex> busy(thinking);
                        releaseBrain();
                    }
                    stopGenerating = false;
                    holdSplit = false;
                }).detach();
                continue;
            }
            if(!ok){
                std::cout << "node " << node->name << " disconnected" << std::endl;
                dropNode(node, "connection lost");
                continue;
            }
            memcpy(&node->stats, body.data(), sizeof(NodeStats));
            sealStats(node->stats);
            node->pingMs = (uint32_t)(nowMs() - start);
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void pushWeights();

void membershipLoop() {
    std::string applied;
    std::string seen;
    uint64_t seenAt = nowMs();
    bool first = true;
    while(true){
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::string current;
        std::string path;
        bool differs = false;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            for(auto& entry : engine.nodes){
                if(entry.second->connected && entry.second->accepted) current += entry.first + "|";
            }
            path = engine.modelPath;
            std::string wanted = engine.frontNode.empty() ? "" : engine.frontNode + ":" + std::to_string(engine.frontEnd);
            differs = brainState == 2 && wanted != splitApplied;
        }
        if(first){
            applied = current;
            seen = current;
            first = false;
            continue;
        }
        if(current != seen){
            seen = current;
            seenAt = nowMs();
        }
        bool urgent = membershipDirty.exchange(false);
        bool split = splitDirty.exchange(false);
        bool settled = nowMs() - seenAt > 30000;
        bool moved = current != applied && (urgent || settled);
        if(brainState == 0 || holdSplit){
            applied = current;
            continue;
        }
        if(path.empty() || (!moved && !split && !(differs && settled))) continue;
        applied = current;
        bool gone = false;
        {
            std::unique_lock<std::mutex> busy(thinking);
            gone = brainState == 0 || holdSplit;
            if(!gone) closeFront();
        }
        if(gone) continue;
        std::cout << "pool membership changed, splitting the model again" << std::endl;
        loadBrain(path, nextLoad());
    }
}

void reconnectLoop() {
    while(true){
        std::vector<TrustedNode> wanted;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            for(const TrustedNode& t : engine.trusted){
                auto found = engine.nodes.find(t.name);
                if(found == engine.nodes.end() || !found->second->connected) wanted.push_back(t);
            }
        }
        for(const TrustedNode& t : wanted){
            std::string why;
            connectNode(t.ip, t.port, t.key, why);
        }
        std::this_thread::sleep_for(std::chrono::seconds(15));
    }
}

std::string frontPlanFields(const std::string& node, int end) {
    const ModelFile& model = engine.model;
    uint64_t vramCap = 0, ramCap = 0;
    for(const PoolSlot& s : engine.plan.slots){
        if(s.node == node && s.kind == SLOT_GPU) vramCap += s.limit;
        if(s.node == node && s.kind == SLOT_RAM) ramCap += s.limit;
    }
    auto base = model.meta.find(model.arch + ".rope.freq_base");
    double rope = base == model.meta.end() ? 1e7 : base->second.real;

    std::vector<uint64_t> counts;
    uint64_t seen = 0;
    uint32_t count = 0, width = 0;
    std::string heatPath = model.paths.empty() ? "" : heatFor(model.paths[0], (size_t)model.layers, (int)model.experts);
    FILE* f = nullptr;
    if(!heatPath.empty()) fopen_s(&f, heatPath.c_str(), "rb");
    if(f){
        char tag[4] = {};
        uint32_t version = 0;
        bool ok = fread(tag, 1, 4, f) == 4 && fread(&version, 4, 1, f) == 1 && fread(&count, 4, 1, f) == 1 && fread(&width, 4, 1, f) == 1 && fread(&seen, 8, 1, f) == 1;
        ok = ok && memcmp(tag, "HEAT", 4) == 0 && version == 1 && count == model.layers && width == model.experts;
        if(ok){
            counts.resize((size_t)count * width);
            ok = fread(counts.data(), 8, counts.size(), f) == counts.size();
        }
        if(!ok) counts.clear();
        fclose(f);
    }

    std::string shape;
    for(const auto& m : model.meta){
        const MetaValue& v = m.second;
        if(m.first.rfind(model.arch + ".", 0) != 0 || !v.wholes.empty() || !v.reals.empty() || !v.texts.empty() || !v.text.empty()) continue;
        char real[48] = {};
        snprintf(real, sizeof(real), "%.17g", v.real);
        shape += (shape.empty() ? "" : ",") + quoteJson(m.first) + ":[" + std::to_string(v.whole) + "," + real + "]";
    }

    std::ostringstream out;
    out << ",\"front\":{\"layers\":" << model.layers << ",\"experts\":" << model.experts << ",\"expertsUsed\":" << model.expertsUsed << ",\"hidden\":" << model.hidden
        << ",\"arch\":" << quoteJson(model.arch) << ",\"meta\":{" << shape << "}"
        << ",\"interval\":" << metaWhole(model, model.arch + ".full_attention_interval", 4) << ",\"topK\":" << metaWhole(model, model.arch + ".attention.indexer.top_k", 2048)
        << ",\"ropeBase\":" << std::to_string(rope) << ",\"context\":" << engine.context << ",\"relayLimit\":" << std::max(2, std::min(33, std::max(engine.draftLength, engine.copyReach) + 1))
        << ",\"tailSkip\":" << (engine.fastRouting ? 7 : 0) << ",\"foresee\":" << (engine.foresee ? "true" : "false") << ",\"vramCap\":" << vramCap << ",\"ramCap\":" << ramCap << ",\"heatTokens\":" << seen << ",\"heat\":[";
    for(int l = 0; l < end && !counts.empty(); l++){
        out << (l ? "," : "") << "[";
        for(uint32_t e = 0; e < width; e++) out << (e ? "," : "") << counts[(size_t)l * width + e];
        out << "]";
    }
    out << "]}";
    return out.str();
}

void pushWeights() {
    std::vector<NodeShare> shares;
    std::vector<LayerShare> layerShares;
    std::vector<std::string> paths;
    std::vector<TensorInfo> tensors;
    std::string modelId;
    std::string selfName;
    uint32_t expertCount = 0;
    std::string frontNode;
    std::string frontFields;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        if(!engine.loaded || !engine.plan.ok) return;
        shares = engine.plan.nodes;
        layerShares = engine.plan.layers;
        paths = engine.model.paths;
        tensors = engine.model.tensors;
        modelId = engine.modelId;
        selfName = engine.selfName;
        expertCount = engine.model.experts;
        frontNode = engine.frontNode;
        if(!frontNode.empty()) frontFields = frontPlanFields(engine.frontNode, engine.frontEnd);
    }

    for(const NodeShare& share : shares){
        if(share.node == selfName || share.firstLayer < 0) continue;

        std::shared_ptr<RemoteNode> node;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            auto found = engine.nodes.find(share.node);
            if(found == engine.nodes.end() || !found->second->connected) continue;
            node = found->second;
            node->transfer = "planning";
        }

        std::vector<const TensorInfo*> mine;
        std::ostringstream plan;
        bool front = share.node == frontNode;
        plan << "{\"modelId\":" << quoteJson(modelId) << ",\"firstLayer\":" << share.firstLayer << ",\"lastLayer\":" << share.lastLayer << ",\"expertCount\":" << expertCount << ",\"role\":\"" << (front ? "front" : "store") << "\"" << (front ? frontFields : "") << ",\"layers\":[";
        bool firstLayer = true;
        for(const LayerShare& l : layerShares){
            if(l.node != share.node) continue;
            plan << (firstLayer ? "" : ",") << "{\"layer\":" << l.layer << ",\"dense\":" << l.denseTier << ",\"gpuExperts\":" << l.gpuExperts << ",\"ramExperts\":" << l.ramExperts << ",\"ssdExperts\":" << l.ssdExperts << "}";
            firstLayer = false;
        }
        plan << "],\"tensors\":[";
        for(const TensorInfo& t : tensors){
            if(t.layer < share.firstLayer || t.layer > share.lastLayer) continue;
            plan << (mine.empty() ? "" : ",") << "{\"name\":" << quoteJson(t.name) << ",\"bytes\":" << t.bytes << ",\"type\":" << t.type << ",\"dims\":[";
            for(uint32_t d = 0; d < t.dimCount; d++) plan << (d ? "," : "") << t.dims[d];
            plan << "]}";
            mine.push_back(&t);
        }
        plan << "]}";
        std::string planText = plan.str();

        std::lock_guard<std::mutex> wire(node->wireLock);
        PacketHead head;
        std::vector<char> body;
        if(!sendPacket(node->wire, PACKET_PLAN, planText.data(), planText.size()) || !readPacket(node->wire, head, body) || head.kind != PACKET_PLAN_ACK){
            std::lock_guard<std::mutex> guard(engine.lock);
            dropNode(node, "lost while sending the plan");
            continue;
        }

        Json missing = parseJson(std::string(body.begin(), body.end()));
        std::set<std::string> wanted;
        uint64_t total = 0;
        for(const Json& name : missing.items) wanted.insert(name.str());
        for(const TensorInfo* t : mine){
            if(wanted.count(t->name)) total += t->bytes;
        }

        {
            std::lock_guard<std::mutex> guard(engine.lock);
            node->transfer = total ? "sending" : "ready";
            node->transferTotal = total;
            node->transferDone = 0;
        }

        std::vector<char> chunk;
        const uint64_t step = 8ULL << 20;
        bool failed = false;

        for(const TensorInfo* t : mine){
            if(failed || !wanted.count(t->name)) continue;

            FILE* f = nullptr;
            fopen_s(&f, paths[t->file].c_str(), "rb");
            if(!f){
                failed = true;
                break;
            }
            _fseeki64(f, (long long)t->offset, SEEK_SET);

            uint32_t nameLength = (uint32_t)t->name.size();
            for(uint64_t done = 0; done < t->bytes && !failed; done += step){
                uint64_t size = t->bytes - done < step ? t->bytes - done : step;
                chunk.resize(20 + nameLength + size);
                memcpy(chunk.data(), &nameLength, 4);
                memcpy(chunk.data() + 4, t->name.data(), nameLength);
                memcpy(chunk.data() + 4 + nameLength, &done, 8);
                memcpy(chunk.data() + 12 + nameLength, &t->bytes, 8);
                if(stopLoading || fread(chunk.data() + 20 + nameLength, 1, (size_t)size, f) != size || !sendPacket(node->wire, PACKET_WEIGHTS_CHUNK, chunk.data(), chunk.size())){
                    failed = true;
                    break;
                }
                std::lock_guard<std::mutex> guard(engine.lock);
                node->transferDone += size;
            }
            fclose(f);
        }

        bool closed = !failed && sendPacket(node->wire, PACKET_WEIGHTS_DONE, nullptr, 0) && readPacket(node->wire, head, body) && head.kind == PACKET_WEIGHTS_DONE;

        std::lock_guard<std::mutex> guard(engine.lock);
        if(!closed){
            dropNode(node, "lost while sending weights");
            node->transfer = "failed";
            continue;
        }
        node->transfer = "ready";
        std::cout << "node " << node->name << " holds layers " << share.firstLayer << ".." << share.lastLayer << std::endl;
    }
}

void loadLocal() {
    std::vector<TensorSource> sources;
    std::vector<LayerShare> layers;
    NodeShare mine;
    uint32_t expertCount = 0;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        if(!engine.loaded || !engine.plan.ok) return;
        for(const TensorInfo& t : engine.model.tensors){
            sources.push_back({t.name, engine.model.paths[t.file], t.offset, t.bytes, t.layer, t.expert});
        }
        for(const LayerShare& l : engine.plan.layers){
            if(l.node == engine.selfName) layers.push_back(l);
        }
        for(const NodeShare& n : engine.plan.nodes){
            if(n.node == engine.selfName) mine = n;
        }
        expertCount = engine.model.experts;
    }

    uint64_t start = nowMs();
    std::cout << "loading this pc's share into vram and ram..." << std::endl;
    loadShare(local, sources, layers, mine.holdsEdges, mine.headTier, mine.tableTier, expertCount);
    std::lock_guard<std::mutex> guard(local.lock);
    std::cout << "local share " << local.state << ": " << (double)local.gpuBytes / gib << " GB vram, " << (double)local.ramBytes / gib << " GB ram in " << (nowMs() - start) / 1000.0 << " s" << (local.problem.empty() ? "" : " | " + local.problem) << std::endl;
}

struct FoundModel {
    std::string path;
    uint64_t bytes = 0;
    std::string name;
    bool shared = false;
};

std::vector<FoundModel> scanModels(const std::string& folder);

std::string diskJson() {
    static std::mutex counting;
    static uint64_t checked = 0, modelsBytes = 0, ownBytes = 0;
    static std::string checkedFolder;
    std::string folder;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        folder = engine.modelFolder;
    }
    uint64_t models = 0, own = 0;
    {
        std::lock_guard<std::mutex> once(counting);
        if(nowMs() - checked > 30000 || folder != checkedFolder){
            modelsBytes = 0;
            std::set<std::pair<uint64_t, uint64_t>> seen;
            std::error_code walk;
            for(auto& entry : fs::recursive_directory_iterator(folder, walk)){
                std::string name = entry.path().filename().string();
                std::transform(name.begin(), name.end(), name.begin(), ::tolower);
                if(!entry.is_regular_file(walk) || name.size() < 5 || name.substr(name.size() - 5) != ".gguf" || name.rfind("mmproj", 0) == 0 || name.rfind("mtp", 0) == 0) continue;
                HANDLE file = CreateFileW(entry.path().wstring().c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
                if(file == INVALID_HANDLE_VALUE) continue;
                BY_HANDLE_FILE_INFORMATION info = {};
                bool known = GetFileInformationByHandle(file, &info) != 0;
                CloseHandle(file);
                if(!known) continue;
                uint64_t index = ((uint64_t)info.nFileIndexHigh << 32) | info.nFileIndexLow;
                if(!seen.insert({(uint64_t)info.dwVolumeSerialNumber, index}).second) continue;
                modelsBytes += ((uint64_t)info.nFileSizeHigh << 32) | info.nFileSizeLow;
            }
            ownBytes = 0;
            std::error_code e;
            for(auto& entry : fs::directory_iterator(modelsHome(), e)){
                std::string name = entry.path().filename().string();
                if(entry.is_regular_file(e) && (name.find(".mxpack") != std::string::npos || name.find(".draft") != std::string::npos)) ownBytes += entry.file_size(e);
            }
            checked = nowMs();
            checkedFolder = folder;
        }
        models = modelsBytes;
        own = ownBytes;
    }
    IO_COUNTERS io = {};
    GetProcessIoCounters(GetCurrentProcess(), &io);
    std::error_code e;
    std::lock_guard<std::mutex> guard(engine.lock);
    std::string drive = fs::absolute(engine.loaded && !engine.model.paths.empty() ? engine.model.paths[0] : folder, e).root_name().string();
    std::ostringstream out;
    out << "{\"folder\":" << quoteJson(folder) << ",\"drive\":" << quoteJson(drive) << ",\"kind\":\"" << driveName(engine.selfStats.host.driveKind) << "\""
        << ",\"free\":" << engine.selfStats.host.diskFree << ",\"total\":" << engine.selfStats.host.diskTotal
        << ",\"models\":" << models << ",\"own\":" << own << ",\"current\":" << (engine.loaded ? engine.model.totalBytes : 0)
        << ",\"fromDisk\":" << (brainState == 2 ? engine.diskExperts : 0) << ",\"share\":" << (brainState == 2 ? engine.shareBytes : 0)
        << ",\"firstLayer\":" << (brainState == 2 ? brain.firstLayer : 0) << ",\"ready\":" << (brainState == 2 ? "true" : "false") << ",\"engineRead\":" << io.ReadTransferCount << "}";
    return out.str();
}

std::vector<FoundModel> scanModels(const std::string& folder) {
    std::vector<FoundModel> found;
    std::error_code code;

    for(auto& entry : fs::recursive_directory_iterator(folder, code)){
        if(!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if(lower.size() < 5 || lower.substr(lower.size() - 5) != ".gguf") continue;
        if(lower.find(".draft") != std::string::npos || lower.rfind("mmproj", 0) == 0 || lower.rfind("mtp", 0) == 0) continue;

        size_t part = lower.find("-of-");
        if(part != std::string::npos && part >= 6 && lower.substr(part - 6, 6) != "-00001") continue;

        uint64_t bytes = entry.file_size();
        if(part != std::string::npos){
            for(int i = 2; i < 100; i++){
                std::string next = splitPath(entry.path().string(), i, atoi(lower.c_str() + part + 4));
                if(next.empty() || !fs::exists(next, code)) break;
                bytes += fs::file_size(next, code);
            }
        }
        found.push_back({entry.path().string(), bytes, name, false});
    }

    // modelos que otro programa local ya bajo: su carpeta y sus nombres son los de ese programa
    char where[MAX_PATH] = {};
    fs::path store = GetEnvironmentVariableA("OLLAMA_MODELS", where, sizeof(where)) > 0
        ? fs::path(where)
        : GetEnvironmentVariableA("USERPROFILE", where, sizeof(where)) > 0
            ? fs::path(where) / ".ollama" / "models"
            : fs::path();
    fs::path manifests = store / "manifests";
    std::error_code walk;
    if(store.empty() || !fs::exists(manifests, walk)) return found;

    for(auto& entry : fs::recursive_directory_iterator(manifests, walk)){
        if(!entry.is_regular_file(walk)) continue;
        std::vector<std::string> parts;
        for(auto& part : fs::relative(entry.path(), manifests, walk)) parts.push_back(part.string());
        if(parts.size() < 4) continue;
        std::string owner = parts[parts.size() - 3];
        std::string title = (owner == "library" ? "" : owner + "/") + parts[parts.size() - 2] + ":" + parts.back();
        if(parts[0] != "registry.ollama.ai") title = parts[0] + "/" + title;

        std::ifstream file(entry.path(), std::ios::binary);
        std::stringstream text;
        text << file.rdbuf();
        Json manifest = parseJson(text.str());
        for(const Json& layer : manifest["layers"].items){
            std::string digest = layer["digest"].str();
            if(layer["mediaType"].str() != "application/vnd.ollama.image.model" || digest.rfind("sha256:", 0) != 0) continue;
            fs::path blob = store / "blobs" / ("sha256-" + digest.substr(7));
            std::error_code missing;
            uint64_t bytes = fs::file_size(blob, missing);
            if(missing) continue;
            found.push_back({blob.string(), bytes, title, true});
        }
    }
    return found;
}

bool loadModel(const std::string& path) {
    ModelFile model;
    std::string error;
    if(!openModel(path, model, error)){
        std::lock_guard<std::mutex> guard(engine.lock);
        engine.loadError = error;
        std::cout << "model load failed: " << error << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> guard(engine.lock);
    engine.model = std::move(model);
    engine.modelPath = path;
    engine.shape = shapeOf(engine.model);
    engine.modelId = modelIdOf(engine.model);
    engine.loaded = true;
    engine.loadError.clear();
    replan();

    std::cout << "model: " << engine.model.name << " | " << engine.model.arch << " | " << engine.model.layers << " layers | " << engine.model.experts << " experts | " << (double)engine.model.totalBytes / gib << " GB in " << engine.model.paths.size() << " files" << std::endl;
    uint32_t drive = readDriveKind(path);
    std::cout << "model disk: " << driveName(drive) << std::endl;
    if(drive == DriveHdd || drive == DriveUsb) std::cout << "warning: the model sits on a slow disk, experts read from it will hold generation back; an SSD or NVMe drive is much faster" << std::endl;
    return true;
}

std::string gpuJson(const GpuStats& g) {
    std::ostringstream out;
    out << "{\"present\":" << (g.present ? "true" : "false") << ",\"name\":" << quoteJson(g.name)
        << ",\"vramTotal\":" << g.vramTotal << ",\"vramFree\":" << g.vramFree
        << ",\"load\":" << g.load << ",\"memoryLoad\":" << g.memoryLoad << ",\"temperature\":" << g.temperature
        << ",\"power\":" << g.powerMilliwatts / 1000.0 << ",\"powerLimit\":" << g.powerLimitMilliwatts / 1000.0
        << ",\"clock\":" << g.clockMhz << ",\"fan\":" << g.fanPercent
        << ",\"pcieGen\":" << g.pcieGen << ",\"pcieWidth\":" << g.pcieWidth
        << ",\"pcieTx\":" << (uint64_t)g.pcieTxKbs * 1024 << ",\"pcieRx\":" << (uint64_t)g.pcieRxKbs * 1024
        << ",\"energy\":" << g.energyMillijoules / 3600000.0 << "}";
    return out.str();
}

std::string statsJson(const NodeStats& s) {
    std::ostringstream out;
    out << "{\"name\":" << quoteJson(s.host.nodeName) << ",\"cpu\":" << quoteJson(s.host.cpuName)
        << ",\"cores\":" << s.host.cores << ",\"threads\":" << s.host.threads << ",\"cpuLoad\":" << s.host.cpuLoad
        << ",\"ramTotal\":" << s.host.ramTotal << ",\"ramFree\":" << s.host.ramFree
        << ",\"diskRead\":" << s.host.diskReadBytes << ",\"diskWrite\":" << s.host.diskWriteBytes
        << ",\"diskFree\":" << s.host.diskFree << ",\"diskTotal\":" << s.host.diskTotal << ",\"drive\":\"" << driveName(s.host.driveKind) << "\""
        << ",\"heldGpu\":" << s.host.heldGpu << ",\"heldRam\":" << s.host.heldRam << ",\"loadTotal\":" << s.host.loadTotal << ",\"loadDone\":" << s.host.loadDone
        << ",\"gpu\":" << gpuJson(s.gpu) << "}";
    return out.str();
}

std::string planJson(const PoolPlan& plan) {
    std::ostringstream out;
    out << "{\"ok\":" << (plan.ok ? "true" : "false") << ",\"problem\":" << quoteJson(plan.problem) << ",\"demand\":" << plan.demandBytes << ",\"nodes\":[";
    for(size_t i = 0; i < plan.nodes.size(); i++){
        const NodeShare& n = plan.nodes[i];
        out << (i ? "," : "") << "{\"node\":" << quoteJson(n.node) << ",\"firstLayer\":" << n.firstLayer << ",\"lastLayer\":" << n.lastLayer
            << ",\"holdsEdges\":" << (n.holdsEdges ? "true" : "false") << ",\"gpu\":" << n.gpuBytes << ",\"ram\":" << n.ramBytes << ",\"ssd\":" << n.ssdBytes << "}";
    }
    out << "],\"layers\":[";
    for(size_t i = 0; i < plan.layers.size(); i++){
        const LayerShare& l = plan.layers[i];
        out << (i ? "," : "") << "{\"layer\":" << l.layer << ",\"node\":" << quoteJson(l.node) << ",\"dense\":\"" << kindName(l.denseTier)
            << "\",\"gpuExperts\":" << l.gpuExperts << ",\"ramExperts\":" << l.ramExperts << ",\"ssdExperts\":" << l.ssdExperts << "}";
    }
    out << "]}";
    return out.str();
}

std::string poolJson() {
    std::lock_guard<std::mutex> guard(engine.lock);
    std::ostringstream out;
    out << "{\"self\":" << quoteJson(engine.selfName) << ",\"selfScreen\":" << engine.selfScreen << ",\"backend\":" << quoteJson(engine.backend) << ",\"running\":" << quoteJson(engine.running) << ",\"generation\":\"" << brainName() << "\",\"context\":" << engine.context
        << ",\"pinned\":" << (brainState == 2 && brain.gpu ? brain.gpu->warmPinnedBytes : 0)
        << ",\"mainGpu\":" << engine.mainGpu << ",\"secondGpu\":" << (engine.secondGpu ? "true" : "false") << ",\"helperGpu\":" << engine.helperGpu << ",\"helperGb\":" << engine.helperGb << ",\"helperFirst\":" << (engine.helperFirst ? "true" : "false") << ",\"frontLayers\":" << engine.frontCap << ",\"frontEnd\":" << engine.frontEnd
        << ",\"draftLength\":" << engine.draftLength << ",\"copyReach\":" << engine.copyReach << ",\"cpuPlan\":" << quoteJson(engine.cpuPlan) << ",\"warmShare\":" << engine.warmShare << ",\"holdRam\":" << (engine.holdRam ? "true" : "false") << ",\"ramWhole\":" << (engine.ramWhole ? "true" : "false") << ",\"pinShare\":" << engine.pinShare << ",\"pinGb\":" << engine.pinGb << ",\"vision\":" << (engine.vision ? "true" : "false") << ",\"visionGpu\":" << engine.visionGpu << ",\"visionReady\":" << (brainState == 2 && brain.gpu && brain.gpu->seeing ? "true" : "false") << ",\"visionProblem\":" << quoteJson(brain.gpu ? brain.gpu->visionProblem : std::string()) << ",\"draftSure\":" << engine.draftSure << ",\"speculate\":" << (engine.speculate ? "true" : "false")
        << ",\"fastRouting\":" << (engine.fastRouting ? "true" : "false") << ",\"foresee\":" << (engine.foresee ? "true" : "false") << ",\"gpus\":[";
    for(size_t i = 0; i < engine.gpus.size(); i++){
        const GpuCard& g = engine.gpus[i];
        out << (i ? "," : "") << "{\"ordinal\":" << g.ordinal << ",\"name\":" << quoteJson(g.name) << ",\"vram\":" << g.vram << ",\"compute\":\"" << g.major << "." << g.minor << "\"}";
    }
    out << "]"
        << ",\"loaded\":" << (engine.loaded ? "true" : "false") << ",\"notice\":" << quoteJson(engine.notice)
        << ",\"model\":{\"name\":" << quoteJson(engine.model.name) << ",\"arch\":" << quoteJson(engine.model.arch) << ",\"path\":" << quoteJson(engine.modelPath)
        << ",\"bytes\":" << engine.model.totalBytes << ",\"layers\":" << engine.model.layers << ",\"experts\":" << engine.model.experts
        << ",\"expertsUsed\":" << engine.model.expertsUsed << ",\"contextLength\":" << engine.model.contextLength << ",\"error\":" << quoteJson(engine.loadError) << "}";

    out << ",\"nodes\":[{\"name\":" << quoteJson(engine.selfName) << ",\"self\":true,\"connected\":true,\"accepted\":true,\"pingMs\":0,\"stats\":" << statsJson(engine.selfStats) << "}";
    for(auto& entry : engine.nodes){
        const RemoteNode& n = *entry.second;
        out << ",{\"name\":" << quoteJson(n.name) << ",\"self\":false,\"ip\":" << quoteJson(n.ip) << ",\"port\":" << n.port
            << ",\"connected\":" << (n.connected ? "true" : "false") << ",\"accepted\":" << (n.accepted ? "true" : "false")
            << ",\"pingMs\":" << n.pingMs << ",\"problem\":" << quoteJson(n.problem) << ",\"transfer\":" << quoteJson(n.transfer)
            << ",\"transferTotal\":" << n.transferTotal << ",\"transferDone\":" << n.transferDone << ",\"stats\":" << statsJson(n.stats) << "}";
    }

    out << "],\"refused\":[";
    bool first = true;
    for(auto& entry : engine.refused){
        std::string name;
        for(const TrustedNode& t : engine.trusted){
            if(t.ip + ":" + std::to_string(t.port) == entry.first) name = t.name;
        }
        out << (first ? "" : ",") << "{\"address\":" << quoteJson(entry.first) << ",\"name\":" << quoteJson(name) << ",\"why\":" << quoteJson(entry.second) << "}";
        first = false;
    }

    out << "],\"requests\":[";
    first = true;
    for(const JoinRequest& r : engine.joins){
        if(r.status != "pending") continue;
        out << (first ? "" : ",") << "{\"id\":" << quoteJson(r.id) << ",\"ip\":" << quoteJson(r.ip) << ",\"tcpPort\":" << r.tcpPort << ",\"name\":" << quoteJson(r.name)
            << ",\"gpu\":" << quoteJson(r.gpu) << ",\"vramTotal\":" << r.vramTotal << ",\"ramTotal\":" << r.ramTotal << ",\"cpu\":" << quoteJson(r.cpu)
            << ",\"age\":" << (nowMs() - r.when) / 1000 << ",\"problem\":" << quoteJson(r.problem) << "}";
        first = false;
    }

    out << "],\"network\":{\"acceptRequests\":" << (engine.acceptRequests ? "true" : "false") << ",\"doorPort\":" << engine.doorPort
        << ",\"doorOpen\":" << (engine.doorOpen ? "true" : "false") << ",\"doorProblem\":" << quoteJson(engine.doorProblem)
        << ",\"apiPort\":" << engine.apiPort << ",\"apiListen\":" << quoteJson(engine.apiListen) << ",\"apiBound\":" << quoteJson(engine.apiBound) << ",\"addresses\":[";
    std::vector<std::string> addresses = localAddresses();
    for(size_t i = 0; i < addresses.size(); i++) out << (i ? "," : "") << quoteJson(addresses[i]);

    out << "]},\"slots\":[";
    for(size_t i = 0; i < engine.slots.size(); i++){
        const PoolSlot& s = engine.slots[i];
        uint64_t limit = 0, used = 0;
        for(const PoolSlot& p : engine.plan.slots){
            if(p.node == s.node && p.kind == s.kind){
                limit = p.limit;
                used = p.used;
            }
        }
        out << (i ? "," : "") << "{\"node\":" << quoteJson(s.node) << ",\"kind\":\"" << kindName(s.kind) << "\",\"gb\":" << s.gb << ",\"percent\":" << s.percent << ",\"keep\":" << s.keep
            << ",\"limit\":" << limit << ",\"used\":" << used << "}";
    }
    out << "],\"plan\":" << planJson(engine.plan);
    {
        std::lock_guard<std::mutex> held(local.lock);
        out << ",\"local\":{\"state\":" << quoteJson(local.state) << ",\"problem\":" << quoteJson(local.problem) << ",\"total\":" << local.total
            << ",\"done\":" << local.done << ",\"gpu\":" << local.gpuBytes << ",\"ram\":" << local.ramBytes << "}";
    }
    out << "}";
    return out.str();
}

std::string diskJson();

std::string monitorJson() {
    std::string disk = diskJson();
    std::lock_guard<std::mutex> guard(engine.lock);
    std::ostringstream out;
    const char* state = engine.activeRequests > 0
        ? "busy"
        : brainState == 1
            ? "loading"
            : engine.loaded
                ? "idle"
                : "empty";
    out << "{\"state\":\"" << state << "\",\"generation\":\"" << brainName() << "\",\"generationProblem\":" << quoteJson(brainProblem) << ",\"active\":" << engine.activeRequests
        << ",\"model\":" << quoteJson(engine.model.name) << ",\"context\":" << engine.context << ",\"nodes\":[";
    out << "{\"name\":" << quoteJson(engine.selfName) << ",\"self\":true,\"connected\":true,\"pingMs\":0,\"stats\":" << statsJson(engine.selfStats) << ",\"moreGpus\":[";
    for(size_t i = 0; i < engine.moreGpus.size(); i++){
        bool helping = brainState == 2 && brain.gpu && brain.gpu->helper.ready && brain.gpu->helper.card == engine.moreGpus[i].first;
        out << (i ? "," : "") << "{\"stats\":" << gpuJson(engine.moreGpus[i].second) << ",\"role\":\"" << (helping ? "helps with experts" : engine.secondGpu ? "not used" : "turned off in Settings") << "\""
            << ",\"held\":" << (helping ? brain.gpu->helper.bytes : 0) << ",\"experts\":" << (helping ? brain.gpu->helper.perLayer : 0) << "}";
    }
    out << "]}";
    for(auto& entry : engine.nodes){
        const RemoteNode& n = *entry.second;
        if(!n.accepted) continue;
        out << ",{\"name\":" << quoteJson(n.name) << ",\"self\":false,\"connected\":" << (n.connected ? "true" : "false") << ",\"pingMs\":" << n.pingMs << ",\"stats\":" << statsJson(n.stats) << "}";
    }
    out << "],\"requests\":[";
    for(size_t i = 0; i < engine.requests.size(); i++){
        const RequestRecord& r = engine.requests[i];
        out << (i ? "," : "") << "{\"when\":" << r.when << ",\"status\":" << r.status << ",\"prompt\":" << r.promptTokens << ",\"output\":" << r.outputTokens
            << ",\"cached\":" << r.cachedTokens << ",\"promptSpeed\":" << r.promptSpeed << ",\"outputSpeed\":" << r.outputSpeed << ",\"durationMs\":" << r.durationMs << "}";
    }
    uint64_t experts = 0, gpuExperts = 0;
    for(const LayerShare& l : engine.plan.layers){
        experts += l.gpuExperts + l.ramExperts + l.ssdExperts;
        gpuExperts += l.gpuExperts;
    }
    out << "],\"expertsInVram\":" << (experts ? (double)gpuExperts / experts : 0) << ",\"disk\":" << disk << "}";
    return out.str();
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::string raw;
    std::string origin;
    std::string host;
};

bool readHttp(SOCKET s, HttpRequest& req, size_t limit = 1ULL << 31) {
    char buffer[16384];
    size_t headerEnd = std::string::npos;
    size_t wanted = 0;

    while(true){
        int got = recv(s, buffer, sizeof(buffer), 0);
        if(got <= 0) break;
        req.raw.append(buffer, got);
        if(req.raw.size() > limit) return false;

        if(headerEnd == std::string::npos){
            headerEnd = req.raw.find("\r\n\r\n");
            if(headerEnd == std::string::npos) continue;
            std::string head = req.raw.substr(0, headerEnd);
            std::transform(head.begin(), head.end(), head.begin(), ::tolower);
            size_t at = head.find("content-length:");
            wanted = at == std::string::npos ? 0 : (size_t)atoll(head.c_str() + at + 15);
            if(headerEnd + 4 + wanted > limit) return false;
            size_t origin = head.find("\r\norigin:");
            if(origin != std::string::npos){
                size_t end = head.find("\r\n", origin + 2);
                req.origin = head.substr(origin + 9, end == std::string::npos ? std::string::npos : end - origin - 9);
                req.origin.erase(0, req.origin.find_first_not_of(' '));
            }
            size_t host = head.find("\r\nhost:");
            if(host != std::string::npos){
                size_t end = head.find("\r\n", host + 2);
                req.host = head.substr(host + 7, end == std::string::npos ? std::string::npos : end - host - 7);
                req.host.erase(0, req.host.find_first_not_of(' '));
                req.host.erase(req.host.find_last_not_of(' ') + 1);
            }
        }

        if(req.raw.size() >= headerEnd + 4 + wanted) break;
    }

    if(headerEnd == std::string::npos) return false;

    size_t space = req.raw.find(' ');
    size_t second = req.raw.find(' ', space + 1);
    if(space == std::string::npos || second == std::string::npos) return false;
    req.method = req.raw.substr(0, space);
    req.path = req.raw.substr(space + 1, second - space - 1);
    size_t query = req.path.find('?');
    if(query != std::string::npos) req.path = req.path.substr(0, query);
    req.body = req.raw.substr(headerEnd + 4);
    return true;
}

void replyHttp(SOCKET s, int status, const std::string& body, const char* type = "application/json") {
    const char* label = status == 200
        ? "OK"
        : status == 204
            ? "No Content"
            : status == 400
                ? "Bad Request"
                : status == 403
                    ? "Forbidden"
                    : status == 404
                        ? "Not Found"
                        : status == 429
                            ? "Too Many Requests"
                            : "Error";
    std::string out = "HTTP/1.1 " + std::to_string(status) + " " + label + "\r\n"
                      "Content-Type: " + type + "\r\n"
                      "Access-Control-Allow-Origin: *\r\n"
                      "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                      "Access-Control-Allow-Headers: *\r\n"
                      "Content-Length: " + std::to_string(body.size()) + "\r\n"
                      "Connection: close\r\n\r\n" + body;
    sendAll(s, out.data(), out.size());
    shutdown(s, SD_SEND);
}

std::string dechunk(const std::string& body) {
    std::string out;
    size_t at = 0;
    while(at < body.size()){
        size_t lineEnd = body.find("\r\n", at);
        if(lineEnd == std::string::npos) break;
        size_t size = strtoul(body.substr(at, lineEnd - at).c_str(), nullptr, 16);
        if(size == 0) break;
        out.append(body, lineEnd + 2, size);
        at = lineEnd + 2 + size + 2;
    }
    return out;
}

void readUsage(const Json& item, RequestRecord& record) {
    const Json& usage = item["usage"];
    if(usage.kind == JSON_OBJECT){
        record.promptTokens = (uint64_t)usage["prompt_tokens"].num((double)record.promptTokens);
        record.outputTokens = (uint64_t)usage["completion_tokens"].num((double)record.outputTokens);
        record.cachedTokens = (uint64_t)usage["prompt_tokens_details"]["cached_tokens"].num((double)record.cachedTokens);
    }
    const Json& timings = item["timings"];
    if(timings.kind == JSON_OBJECT){
        record.promptSpeed = timings["prompt_per_second"].num(record.promptSpeed);
        record.outputSpeed = timings["predicted_per_second"].num(record.outputSpeed);
    }
}

void forwardChat(SOCKET client, const HttpRequest& req) {
    int port = 0;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        port = engine.backendPort;
        engine.activeRequests++;
    }

    RequestRecord record;
    record.when = (int64_t)time(nullptr) * 1000;
    uint64_t start = nowMs();

    SOCKET backend = connectTcp("127.0.0.1", port, 3000);
    std::string response;

    if(backend == INVALID_SOCKET){
        record.status = 503;
        replyHttp(client, 503, "{\"error\":{\"message\":\"the engine cannot generate yet and no backend answers on port " + std::to_string(port) + "\"}}");
    } else {
        std::string outgoing = req.raw;
        size_t keep = outgoing.find("Connection: keep-alive");
        if(keep == std::string::npos) keep = outgoing.find("Connection: Keep-Alive");
        if(keep != std::string::npos) outgoing.replace(keep, 22, "Connection: close     ");
        sendAll(backend, outgoing.data(), outgoing.size());

        char buffer[16384];
        bool clientGone = false;
        while(true){
            int got = recv(backend, buffer, sizeof(buffer), 0);
            if(got <= 0) break;
            response.append(buffer, got);
            if(send(client, buffer, got, 0) <= 0){
                clientGone = true;
                break;
            }
        }
        closesocket(backend);
        shutdown(client, SD_SEND);
        record.status = clientGone ? 499 : atoi(response.substr(9, 3).c_str());
    }

    record.durationMs = nowMs() - start;

    size_t headerEnd = response.find("\r\n\r\n");
    if(headerEnd != std::string::npos){
        std::string head = response.substr(0, headerEnd);
        std::transform(head.begin(), head.end(), head.begin(), ::tolower);
        std::string body = response.substr(headerEnd + 4);
        if(head.find("transfer-encoding: chunked") != std::string::npos) body = dechunk(body);

        size_t firstChar = body.find_first_not_of(" \r\n\t");
        if(firstChar != std::string::npos && body[firstChar] == '{'){
            readUsage(parseJson(body), record);
        } else {
            std::istringstream lines(body);
            std::string line;
            while(std::getline(lines, line)){
                if(line.rfind("data: {", 0) == 0) readUsage(parseJson(line.substr(6)), record);
            }
        }
    }

    if(record.outputSpeed == 0 && record.outputTokens > 0 && record.durationMs > 0){
        record.outputSpeed = record.outputTokens * 1000.0 / record.durationMs;
    }

    std::lock_guard<std::mutex> guard(engine.lock);
    engine.activeRequests--;
    engine.requests.push_front(record);
    while(engine.requests.size() > 50) engine.requests.pop_back();
}

void wakeBrain() {
    static std::mutex waking;
    std::lock_guard<std::mutex> once(waking);
    if(brainState == 1 || brainState == 2) return;
    std::string path;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        path = engine.modelPath;
    }
    if(path.empty()) return;
    std::cout << "loading " << fs::path(path).filename().string() << " for the first request" << std::endl;
    loadBrain(path, nextLoad());
}

bool collectPictures(const Json& messages, std::vector<std::string>& found, std::string& error) {
    for(const Json& message : messages.items){
        const Json& content = message["content"];
        if(content.kind != JSON_LIST) continue;
        for(const Json& item : content.items){
            std::string type = item["type"].str();
            if(item.has("video") || type == "video"){
                error = "videos are not supported, send pictures";
                return false;
            }
            if(!item.has("image") && !item.has("image_url") && type != "image") continue;
            const Json& link = item["image_url"];
            std::string url = link.kind == JSON_TEXT ? link.str() : link["url"].str();
            if(url.empty()) url = item["image"].str();
            if(url.empty() && item["source"]["type"].str() == "base64") url = "data:" + item["source"]["media_type"].str("image/png") + ";base64," + item["source"]["data"].str();
            if(!url.empty() && url.rfind("data:", 0) != 0 && url.find("://") == std::string::npos) url = "data:image/png;base64," + url;
            std::string bytes;
            if(!pictureFromUrl(url, bytes, error)) return false;
            found.push_back(bytes);
        }
    }
    return true;
}

void generateLocal(SOCKET client, const Json& body, bool messagesApi = false) {
    auto problemJson = [&](const std::string& why) {
        return messagesApi
            ? "{\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":" + quoteJson(why) + "}}"
            : "{\"error\":{\"message\":" + quoteJson(why) + "}}";
    };
    wakeBrain();
    std::unique_lock<std::mutex> busy(thinking);
    if(brainState == 0){
        busy.unlock();
        wakeBrain();
        busy.lock();
    }

    if(brainState != 2 || !brain.ready() || !words){
        std::string why = brainState == 1
            ? "the engine is still loading the model, try again in a moment"
            : brainProblem.empty()
                ? "no model is loaded for generation"
                : brainProblem;
        replyHttp(client, 503, problemJson(why));
        return;
    }

    GenerateRequest request = readGenerateRequest(body);
    request.messagesApi = messagesApi;
    ChatOptions options = chatOptions(body);
    if(brain.gpu && brain.gpu->file.arch != "qwen4exp"){
        auto found = brain.gpu->file.meta.find("tokenizer.chat_template");
        bool effort = found != brain.gpu->file.meta.end() && found->second.text.find("reasoning_effort") != std::string::npos;
        options.effortNotes = effort;
        options.keepByDefault = effort;
    }
    bool gemmaModel = brain.gpu && brain.gpu->gemma;
    options.gemma = gemmaModel;
    bool hyModel = brain.gpu && brain.gpu->hy4;
    options.hy = hyModel;
    std::string prompt, error;
    if(!renderChat(request.messages, request.tools, options, prompt, error)){
        replyHttp(client, 400, problemJson(error));
        return;
    }
    std::vector<std::string> shown;
    if(!collectPictures(request.messages, shown, error)){
        replyHttp(client, 400, problemJson(error));
        return;
    }
    std::vector<int> seen;
    if(!shown.empty() && (!brain.gpu || !brain.gpu->seeing)){
        // hy4 no tiene lector de imagenes: pedir que se prenda vision no sirve si ya esta prendido
        std::string why = brain.gpu && brain.gpu->hy4
            ? "Hy4-flash reads text only in Sneo, describe the picture in words or load a model that reads pictures (Qwen3.8-Flash-Next, Qwen3.6 or Gemma 4)"
            : brain.gpu && !brain.gpu->visionProblem.empty()
                ? brain.gpu->visionProblem
                : "this model is loaded without Vision, turn on Vision in Settings to send pictures";
        replyHttp(client, 400, problemJson(why));
        return;
    }
    for(const std::string& bytes : shown){
        auto looked = std::chrono::steady_clock::now();
        int slot = brain.gpu->seePicture(bytes, error);
        if(slot < 0){
            replyHttp(client, 400, problemJson(error));
            return;
        }
        seen.push_back(slot);
        const auto& picture = brain.gpu->pictures[slot];
        std::cout << "picture " << seen.size() << ": " << picture.wide * 32 << "x" << picture.high * 32 << ", " << picture.wide * picture.high << " cells, read in " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - looked).count() << " ms" << std::endl;
    }

    {
        std::lock_guard<std::mutex> guard(engine.lock);
        engine.activeRequests++;
    }

    Tokenizer& tokenizer = *words;
    int boundaryMark = tokenizer.find(gemmaModel ? "<turn|>" : hyModel ? hyEnd() : "<|im_end|>");
    bool side = body["sneo_side"].truthy();
    TextStream stream;
    bool first = true;
    uint64_t reused = 0;

    GenerateHooks hooks;
    std::string failure;
    hooks.problem = [&]{
        return !failure.empty()
            ? failure
            : brain.gpu && !brain.gpu->problem.empty()
                ? brain.gpu->problem
                : brain.problem.empty()
                    ? std::string("generation failed")
                    : brain.problem;
    };
    hooks.render = [&](const GenerateRequest&){ return prompt; };
    int picturePad = seen.empty() ? -1 : tokenizer.find("<|image_pad|>");
    hooks.encode = [&](const std::string& text){
        std::vector<int> ids = tokenizer.encode(text, true);
        if(picturePad < 0) return ids;
        std::vector<int> cells;
        size_t next = 0;
        for(int id : ids){
            if(id != picturePad || next >= seen.size()){
                cells.push_back(id);
                continue;
            }
            int slot = seen[next++];
            const auto& picture = brain.gpu->pictures[slot];
            for(int k = 0; k < picture.high * picture.wide; k++) cells.push_back(-1 - (slot * 32768 + k));
        }
        return cells;
    };
    hooks.piece = [&](int id){ return stream.push(tokenizer.tokenText(id)); };
    hooks.tokenText = [&](int id){ return tokenizer.tokenText(id); };
    if(hyModel){
        hooks.thinkOpen = hyMark("think");
        hooks.thinkClose = hyMark("think", true);
        hooks.toolOpen = hyMark("tool_calls");
        hooks.toolClose = hyMark("tool_calls", true);
        hooks.closeThinking = hyMark("think", true);
    }
    if(gemmaModel){
        hooks.thinkOpen = "<|channel>thought";
        hooks.thinkClose = "<channel|>";
        hooks.toolOpen = "<|tool_call>";
        hooks.toolClose = "<tool_call|>";
        hooks.closeThinking = "\n<channel|>";
    }
    hooks.isStop = [&](int id){ return tokenizer.isStop(id); };
    hooks.reusedTokens = [&]{ return reused; };
    hooks.reusable = [&](const std::vector<int>& ids) -> uint64_t {
        const std::vector<int>& held = brain.tokens();
        if(!held.empty() && held.size() < ids.size() && std::equal(held.begin(), held.end(), ids.begin())) return held.size();
        for(const std::vector<int>* mark : {&turnBoundary, &boundary}){
            if(!mark->empty() && mark->size() < ids.size() && std::equal(mark->begin(), mark->end(), ids.begin())) return mark->size();
        }
        return 0;
    };
    hooks.forward = [&](const std::vector<int>& ids, size_t, std::vector<float>& logits) {
        std::vector<int> feed = ids;
        auto run = [&](const std::vector<int>& part) {
            std::vector<int> positions(part.size());
            for(size_t i = 0; i < part.size(); i++) positions[i] = (int)(brain.tokens().size() + i);
            logits = brain.forward(part, positions);
            return !logits.empty();
        };

        if(first){
            first = false;
            size_t limit = brain.gpu ? (size_t)brain.gpu->contextLimit : (size_t)engine.context;
            if(ids.size() + 16 > limit){
                failure = "this request has " + std::to_string(ids.size()) + " tokens and the context window holds " + std::to_string(limit) + ", raise the context window in Settings, Engine";
                return false;
            }
            std::vector<int>& held = brain.tokens();
            size_t have = held.size();
            size_t firstCut = 0, lastCut = 0;
            for(size_t i = 0; i < ids.size() && boundaryMark >= 0; i++){
                if(ids[i] != boundaryMark) continue;
                if(!firstCut) firstCut = i + 1;
                lastCut = i + 1;
            }
            auto matches = [&](const std::vector<int>& mark) {
                return !mark.empty() && mark.size() < ids.size() && std::equal(mark.begin(), mark.end(), ids.begin());
            };
            size_t at = 0, computed = 0;
            bool fresh = false;
            if(have > 0 && have < ids.size() && std::equal(held.begin(), held.end(), ids.begin())){
                at = have;
            } else if(brain.gpu && matches(turnBoundary) && brain.gpu->restoreState(2)){
                at = turnBoundary.size();
            } else if(brain.gpu && matches(boundary) && brain.gpu->restoreState()){
                at = boundary.size();
            } else {
                brain.reset();
                fresh = true;
            }
            bool systemPiece = fresh && brain.gpu && !side && firstCut >= 256 && firstCut < ids.size();
            bool turnPiece = brain.gpu && !side && lastCut > at && lastCut >= 256 && lastCut < ids.size();
            bool marking = (systemPiece || turnPiece) && brain.gpu->canMark(ids.size() - at);
            if(systemPiece && !marking){
                if(!run(std::vector<int>(ids.begin(), ids.begin() + firstCut))) return false;
                if(brain.gpu->saveState()) boundary.assign(ids.begin(), ids.begin() + firstCut);
                else std::cout << "prompt snapshot failed: " << brain.gpu->problem << std::endl;
                at = firstCut;
                computed = firstCut;
            }
            reused = at - computed;
            if(turnPiece && !marking && lastCut > at){
                if(!run(std::vector<int>(ids.begin() + at, ids.begin() + lastCut))) return false;
                if(brain.gpu->saveState(2)) turnBoundary.assign(ids.begin(), ids.begin() + lastCut);
                else std::cout << "turn snapshot failed: " << brain.gpu->problem << std::endl;
                at = lastCut;
            }
            feed.assign(ids.begin() + at, ids.end());
            if(marking){
                GpuModel& gpu = *brain.gpu;
                gpu.cutRows[0] = systemPiece ? (int)(firstCut - at) : -1;
                gpu.cutSlots[0] = systemPiece ? 0 : -1;
                gpu.cutRows[1] = turnPiece ? (int)(lastCut - at) : -1;
                gpu.cutSlots[1] = turnPiece ? 2 : -1;
                bool done = run(feed);
                if(done && systemPiece && gpu.cutKept[0]) boundary.assign(ids.begin(), ids.begin() + firstCut);
                if(done && turnPiece && gpu.cutKept[1]) turnBoundary.assign(ids.begin(), ids.begin() + lastCut);
                if(done && ((systemPiece && !gpu.cutKept[0]) || (turnPiece && !gpu.cutKept[1]))) std::cout << "a prompt snapshot was not kept inside the prompt block" << std::endl;
                gpu.cutRows[0] = gpu.cutRows[1] = -1;
                gpu.cutSlots[0] = gpu.cutSlots[1] = -1;
                return done;
            }
        }
        return run(feed);
    };

    double proposeMs = 0, verifyMs = 0, markMs = 0, commitMs = 0, feedBefore = brain.gpu ? brain.gpu->feedMs : 0;
    uint64_t helperRowsBefore = brain.gpu ? brain.gpu->helper.rows : 0;
    double helperWaitBefore = brain.gpu ? brain.gpu->helperWaitMs : 0;
    std::string helperProblem = brain.gpu ? brain.gpu->helper.problem : "";
    if(brain.gpu && engine.speculate && body["speculate"].truthy(brain.gpu->file.arch == "qwen4exp" || brain.gpu->drafting || brain.gpu->hy4)){
        hooks.guessStart = brain.gpu->tiedDraft || brain.gpu->hy4 ? 2 : 0;
        hooks.contextLimit = brain.gpu->contextLimit;
        hooks.verifyLimit = brain.gpu->hy4 ? brain.gpu->hyBuffers.everyCap : brain.gpu->drafting ? brain.gpu->relayLimit : brain.gpu->verifyLimit;
        if(brain.gpu->remote && frontLink.relayLimit > 1) hooks.verifyLimit = std::min(hooks.verifyLimit, frontLink.relayLimit);
        hooks.draftLimit = engine.draftLength;
        hooks.copyLimit = engine.copyReach;
        if(brain.gpu->hy4) hooks.copyLeast = 8;
        if(brain.gpu->drafting) hooks.propose = [&](int token, int most){
            uint64_t asked = nowMicros();
            std::vector<int> drafts = brain.gpu->draftPropose(token, std::min(most, brain.gpu->relayLimit - 1));
            proposeMs += (nowMicros() - asked) / 1000.0;
            return drafts;
        };
        hooks.commit = [&](const std::vector<int>& accepted){
            uint64_t asked = nowMicros();
            bool done = brain.gpu->commitRows(accepted);
            commitMs += (nowMicros() - asked) / 1000.0;
            return done;
        };
        hooks.mark = [&](int rows){
            uint64_t asked = nowMicros();
            bool done = brain.gpu->markRows(rows);
            markMs += (nowMicros() - asked) / 1000.0;
            return done;
        };
        hooks.rewind = [&]{ return brain.gpu->rewindRows(); };
        hooks.settle = [&](const std::vector<int>& rows){
            uint64_t asked = nowMicros();
            bool done = brain.gpu->settleRows(rows);
            commitMs += (nowMicros() - asked) / 1000.0;
            return done;
        };
        hooks.verify = [&](const std::vector<int>& ids, std::vector<float>& rows) {
            uint64_t asked = nowMicros();
            std::vector<int> positions(ids.size());
            for(size_t i = 0; i < ids.size(); i++) positions[i] = (int)(brain.tokens().size() + i);
            rows = brain.gpu->forwardAll(ids, positions);
            verifyMs += (nowMicros() - asked) / 1000.0;
            return !rows.empty();
        };
    }

    RequestRecord record;
    record.when = (int64_t)time(nullptr) * 1000;
    uint64_t start = nowMs();
    uint64_t routedBefore = brain.gpu ? brain.gpu->decodeRouted : 0;
    uint64_t residentBefore = brain.gpu ? brain.gpu->decodeResident : 0;
    uint64_t helpedBefore = brain.gpu ? brain.gpu->helper.rows : 0;
    double expertBefore = brain.gpu ? brain.gpu->expertMs : 0;
    uint64_t coldBefore = brain.gpu ? brain.gpu->coldExperts : 0;
    uint64_t foreseenBefore = brain.gpu ? brain.gpu->foreseen : 0;
    uint64_t liftedBefore = brain.gpu ? brain.gpu->liftCount : 0;
    uint64_t relayRoutesBefore = brain.gpu ? brain.gpu->relayRoutes : 0, relayKeptBefore = brain.gpu ? brain.gpu->relayKept : 0;
    uint64_t relayCpuBefore = brain.gpu ? brain.gpu->relayCpu : 0, relayWarmBefore = brain.gpu ? brain.gpu->relayWarm : 0;
    double serveWaitBefore = brain.gpu ? brain.gpu->serveWait : 0, serveWorkBefore = brain.gpu ? brain.gpu->serveWork : 0;
    double denseWorkBefore = brain.gpu ? brain.gpu->denseWork : 0, denseWaitBefore = brain.gpu ? brain.gpu->denseWait : 0;
    uint64_t usedBefore = brain.gpu ? brain.gpu->foreseenUsed : 0;
    double frontBefore = frontLink.roundMs;
    uint64_t roundsBefore = frontLink.rounds;
    uint64_t sentBefore = frontLink.sentBytes, gotBefore = frontLink.gotBytes;
    double waitedBefore = brain.gpu ? brain.gpu->frontMs : 0;
    bool narrow = brain.gpu && request.repeatPenalty == 1.0f && (request.temperature <= 0.0f || (request.topK > 0 && request.topK <= 64));
    request.shortlist = narrow ? 64 : 0;
    if(brain.gpu) brain.gpu->shortlist = request.shortlist;
    GenerateResult result = runGenerate(client, request, hooks);
    if(brain.gpu) brain.gpu->shortlist = 0;
    record.durationMs = nowMs() - start;
    bool split = brain.gpu && brain.gpu->remote;
    if(split && !frontLink.problem.empty()){
        std::cout << "pool node " << frontLink.node << " failed: " << frontLink.problem << std::endl;
        splitDirty = true;
    }
    uint64_t routed = brain.gpu ? brain.gpu->decodeRouted - routedBefore + brain.gpu->relayRoutes - relayRoutesBefore : 0;
    double hit = routed ? 100.0 * (double)(brain.gpu->decodeResident - residentBefore + brain.gpu->relayKept - relayKeptBefore) / (double)routed : 0.0;
    record.status = result.clientGone ? 499 : result.finish == "error" ? 500 : 200;
    if(result.finish == "error") std::cout << "generation failed: " << hooks.problem() << std::endl;
    record.promptTokens = result.promptTokens;
    record.outputTokens = result.outputTokens;
    record.cachedTokens = result.cachedTokens;
    record.promptSpeed = result.promptMs > 0 ? (result.promptTokens - result.cachedTokens) * 1000.0 / result.promptMs : 0;
    record.outputSpeed = result.outputMs > 0 ? result.outputTokens * 1000.0 / result.outputMs : 0;

    if(result.finish == "error" && !request.stream){
        replyHttp(client, 500, problemJson(hooks.problem()));
    }

    std::cout << "generated " << result.outputTokens << " tokens (" << record.outputSpeed << " tok/s), prompt " << result.promptTokens << " tokens with " << result.cachedTokens << " reused (" << record.promptSpeed << " tok/s), drafts kept " << result.kept << " of " << result.drafted << ", model " << result.modelMs << " ms of " << result.outputMs << " ms decoding, vram hit " << hit << "%";
    if(result.drafted) std::cout << ", propose " << proposeMs << " ms, verify " << verifyMs << " ms (draft feed " << (brain.gpu->feedMs - feedBefore) << " ms), snapshot " << markMs << " ms, commit " << commitMs << " ms";
    if(result.copied) std::cout << ", copied from the prompt " << result.copyKept << " of " << result.copied;
    if(brain.gpu && !brain.gpu->gemma && !brain.gpu->hy4){
        unsigned long long probeMemory[3] = {0, 0, 0};
        NvmlLink& probeNv = nvmlLink();
        if(probeNv.ready) probeNv.getMemory(probeNv.device, probeMemory);
        uint64_t probeFree = probeMemory[1];
        std::cout << ", borrowed rows " << brain.gpu->loanUsed << ", context room " << brain.gpu->kvRoom << ", recent slots given back " << brain.gpu->givenBack << " of " << brain.gpu->recentPerLayer << ", fixed " << brain.gpu->fixedExperts << ", free vram " << (probeFree >> 20) << " MB" << (brain.gpu->budgetBytes ? ", windows budget " + std::to_string(brain.gpu->budgetBytes >> 20) + " MB, used " + std::to_string(brain.gpu->usageBytes >> 20) + " MB" : std::string()) << ", graphs " << brain.gpu->tokenGraphs.size();
    }
    if(brain.gpu && brain.gpu->helper.rows > helperRowsBefore) std::cout << ", second gpu " << brain.gpu->helper.rows - helperRowsBefore << " expert rows, waited " << brain.gpu->helperWaitMs - helperWaitBefore << " ms";
    if(brain.gpu && brain.gpu->helper.problem != helperProblem) std::cout << ", second gpu: " << brain.gpu->helper.problem;
    if(!result.thinkingCut.empty()) std::cout << ", thinking closed at " << result.thinkingTokens << " tokens (" << result.thinkingCut << ")";
    // ms de las vueltas al nodo y ms que esta pc de verdad espero por el (en un prompt largo parte de la vuelta pasa mientras hace sus capas)
    if(split) std::cout << ", " << frontLink.node << " " << frontLink.roundMs - frontBefore << " ms in " << frontLink.rounds - roundsBefore << " rounds (this pc waited " << brain.gpu->frontMs - waitedBefore << " ms), " << (frontLink.sentBytes - sentBefore) / 1048576.0 << " MB sent and " << (frontLink.gotBytes - gotBefore) / 1048576.0 << " MB back";
    if(brain.gpu) std::cout << ", cpu experts " << brain.gpu->expertMs - expertBefore << " ms, " << brain.gpu->coldExperts - coldBefore << " outside the warm cache, " << brain.gpu->foreseen - foreseenBefore << " read ahead (" << brain.gpu->foreseenUsed - usedBefore << " used)";
    if(brain.gpu && brain.gpu->liftCount > liftedBefore) std::cout << ", " << brain.gpu->liftCount - liftedBefore << " recent experts brought to video memory";
    if(brain.gpu && brain.gpu->denseHostBytes) std::cout << ", layers in ram: cpu worked " << brain.gpu->denseWork - denseWorkBefore << " ms and waited " << brain.gpu->denseWait - denseWaitBefore << " ms for the gpu";
    if(brain.gpu && brain.gpu->relayRoutes > relayRoutesBefore) std::cout << ", verify passes: " << 100.0 * (double)(brain.gpu->relayKept - relayKeptBefore) / (double)(brain.gpu->relayRoutes - relayRoutesBefore) << "% of routes in vram, " << brain.gpu->relayWarm - relayWarmBefore << " rows from pinned ram, " << brain.gpu->relayCpu - relayCpuBefore << " on the cpu, host waited " << brain.gpu->serveWait - serveWaitBefore << " ms for the gpu and worked " << brain.gpu->serveWork - serveWorkBefore << " ms";
    if(brain.gpu && brain.gpu->helper.rows > helpedBefore) std::cout << ", " << brain.gpu->helper.name << " computed " << brain.gpu->helper.rows - helpedBefore << " expert rows";
    if(brain.gpu && !brain.gpu->helper.ready && !brain.gpu->helper.problem.empty() && helpedBefore > 0) std::cout << ", " << brain.gpu->helper.problem;
    if(brain.gpu && !brain.gpu->useGraphs) std::cout << ", cuda graphs off (" << brain.gpu->graphProblem << ")";
    if(brain.gpu && brain.gpu->useGraphs && !brain.gpu->graphless.empty()) std::cout << ", " << brain.gpu->graphless.size() << " window sizes without cuda graphs (" << brain.gpu->graphProblem << ")";
    if(brain.gpu && brain.gpu->dedicatedBytes) std::cout << ", dedicated vram " << (brain.gpu->dedicatedBytes >> 20) << " MB (most " << (brain.gpu->dedicatedPeak >> 20) << " MB)";
    std::cout << std::endl;

    // windows movio parte del modelo a la ram compartida y no la devuelve: se carga de nuevo apenas termina esta respuesta
    // con lo que de verdad queda libre, que fue lo que medimos que devuelve la velocidad (48 a 82 tok/s)
    // la tarjeta se lleno porque otro programa (un segundo motor) se quedo con lo que esta carga pedia despues:
    // cada respuesta fallaba con cuda error 2 hasta reiniciar, ahora carga de nuevo dejando 1 GB mas libre cada vez
    static uint64_t healedAt = 0;
    std::string problemText = result.finish == "error" ? hooks.problem() : std::string();
    size_t memoryAt = problemText.find("cuda error 2");
    bool outOfMemory = memoryAt != std::string::npos && (memoryAt + 12 >= problemText.size() || !isdigit((unsigned char)problemText[memoryAt + 12]));
    bool heal = brain.gpu && (brain.gpu->movedBytes > 0 || outOfMemory) && (healedAt == 0 || nowMs() - healedAt > (outOfMemory ? 15000 : 120000));
    std::lock_guard<std::mutex> guard(engine.lock);
    engine.activeRequests--;
    engine.requests.push_front(record);
    while(engine.requests.size() > 50) engine.requests.pop_back();
    if(heal && !engine.modelPath.empty()){
        healedAt = nowMs();
        engine.healedAt = healedAt;
        engine.healedMb = brain.gpu->movedBytes >> 20;
        if(outOfMemory){
            engine.oomKeep = std::min<uint64_t>(engine.oomKeep + (1ULL << 30), 6ULL << 30);
            std::cout << "the card ran out of video memory because another program took it, loading the model again and leaving " << (engine.oomKeep >> 30) << " GB more free" << std::endl;
        } else {
            std::cout << "windows moved " << engine.healedMb << " MB of the model from video memory to shared ram and writing stays slow until it loads again, loading it again now" << std::endl;
        }
        std::thread(loadBrain, engine.modelPath, nextLoad()).detach();
    }
}

void handleClient(SOCKET client) {
    HttpRequest req;
    if(!readHttp(client, req)){
        closesocket(client);
        return;
    }

    // una pagina de otro sitio, o un nombre de internet que apunta a esta pc, no puede usar el motor ni leer lo que contesta
    size_t scheme = req.origin.find("://");
    std::string originHost = scheme == std::string::npos ? req.origin : req.origin.substr(scheme + 3);
    originHost = originHost.rfind("[", 0) == 0 ? originHost.substr(0, originHost.find(']') + 1) : originHost.substr(0, originHost.find(':'));
    bool foreign = !req.origin.empty() && originHost != "localhost" && originHost != "127.0.0.1" && originHost != "[::1]";
    std::string hostName = req.host.rfind("[", 0) == 0 ? req.host.substr(0, req.host.find(']') + 1) : req.host.substr(0, req.host.find(':'));
    std::transform(hostName.begin(), hostName.end(), hostName.begin(), ::tolower);
    bool named = !hostName.empty() && hostName != "localhost" && hostName != "127.0.0.1" && hostName != "[::1]";
    bool listenLocal = true;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        listenLocal = engine.apiBound != "network";
    }
    if(foreign || (listenLocal && named)){
        replyHttp(client, 403, "{\"error\":\"requests from other sites are not accepted\"}");
        closesocket(client);
        return;
    }

    if(req.method == "OPTIONS"){
        replyHttp(client, 204, "");
        closesocket(client);
        return;
    }

    Json body = parseJson(req.body);
    std::string route = req.method + " " + req.path;

    if(route == "POST /v1/chat/completions" || route == "POST /api/chat" || route == "POST /v1/messages"){
        int port = 0;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            port = engine.backendPort;
        }
        if(port > 0) forwardChat(client, req);
        else if(route == "POST /v1/messages") generateLocal(client, chatFromMessages(body), true);
        else generateLocal(client, body);
        closesocket(client);
        return;
    }

    if(route == "POST /v1/messages/count_tokens"){
        Json chat = chatFromMessages(body);
        std::string prompt, error;
        ChatOptions counted = chatOptions(chat);
        counted.gemma = brain.gpu && brain.gpu->gemma;
        counted.hy = brain.gpu && brain.gpu->hy4;
        if(!renderChat(chat["messages"], chat["tools"], counted, prompt, error)){
            replyHttp(client, 400, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":" + quoteJson(error) + "}}");
            closesocket(client);
            return;
        }
        size_t count = brainState == 2 && words ? words->encode(prompt, true).size() : prompt.size() / 4;
        replyHttp(client, 200, "{\"input_tokens\":" + std::to_string(count) + "}");
        closesocket(client);
        return;
    }

    if(route == "GET /api/pool"){
        replyHttp(client, 200, poolJson());
        closesocket(client);
        return;
    }

    if(route == "GET /api/monitor"){
        replyHttp(client, 200, monitorJson());
        closesocket(client);
        return;
    }

    if(route == "POST /api/pool"){
        bool switchBackend = false;
        bool resized = false;
        bool split = false;
        bool limitsMoved = false;
        bool gpusMoved = false;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            uint32_t before = engine.context;
            auto selfSlots = [&]{
                std::string key;
                for(const PoolSlot& s : engine.slots){
                    if(s.node == engine.selfName && s.kind != SLOT_SSD) key += std::to_string(s.kind) + ":" + std::to_string(s.gb) + ":" + std::to_string(s.percent) + ":" + std::to_string(s.keep) + ";";
                }
                return key;
            };
            std::string slotsBefore = selfSlots();
            if(body["slots"].kind == JSON_LIST){
                engine.slots.clear();
                for(const Json& s : body["slots"].items){
                    PoolSlot slot;
                    slot.node = s["node"].str();
                    slot.kind = kindValue(s["kind"].str());
                    slot.gb = s["gb"].num();
                    slot.percent = s["percent"].num();
                    slot.keep = s["keep"].num();
                    const NodeStats* stats = slot.node == engine.selfName ? &engine.selfStats : nullptr;
                    auto found = engine.nodes.find(slot.node);
                    if(!stats && found != engine.nodes.end()) stats = &found->second->stats;
                    uint64_t whole = !stats
                        ? 0
                        : slot.kind == SLOT_GPU
                            ? stats->gpu.vramTotal
                            : slot.kind == SLOT_RAM
                                ? stats->host.ramTotal
                                : 0;
                    if(whole > 0 && slot.gb * gib >= (double)whole - 0.05 * gib) slot.gb = 0;
                    if(!slot.node.empty()) engine.slots.push_back(slot);
                }
            }
            if(body.has("context")){
                double asked = body["context"].num(engine.context);
                engine.context = asked >= 4096 && asked <= 1048576 ? (uint32_t)asked : engine.context;
            }
            resized = engine.context != before;
            limitsMoved = selfSlots() != slotsBefore;
            if(limitsMoved) engine.oomKeep = 0;
            std::string wanted = body["backend"].str(engine.backend);
            if(wanted == "auto" || wanted == "gpu" || wanted == "cpu"){
                switchBackend = wanted != engine.backend;
                engine.backend = wanted;
            }
            int pickedGpu = body.has("mainGpu") ? (int)body["mainGpu"].num(-1) : engine.mainGpu;
            if(pickedGpu < -1 || pickedGpu >= (int)engine.gpus.size()) pickedGpu = -1;
            bool helping = body.has("secondGpu") ? body["secondGpu"].truthy(true) : engine.secondGpu;
            int helperPick = body.has("helperGpu") ? (int)body["helperGpu"].num(-1) : engine.helperGpu;
            if(helperPick < -1 || helperPick >= (int)engine.gpus.size()) helperPick = -1;
            double helperRoom = body.has("helperGb") ? std::max(0.0, body["helperGb"].num(0)) : engine.helperGb;
            bool helperAhead = body.has("helperFirst") ? body["helperFirst"].truthy(false) : engine.helperFirst;
            int frontCap = body.has("frontLayers") ? std::max(0, (int)body["frontLayers"].num(0)) : engine.frontCap;
            int drafts = body.has("draftLength") ? std::max(1, std::min(16, (int)body["draftLength"].num(3))) : engine.draftLength;
            int reach = body.has("copyReach") ? std::max(0, std::min(32, (int)body["copyReach"].num(8))) : engine.copyReach;
            std::string plan = body.has("cpuPlan") ? body["cpuPlan"].str("all") : engine.cpuPlan;
            if(plan != "performance" && plan != "performance-ht" && plan != "auto") plan = "all";
            int share = body.has("warmShare") ? std::max(0, std::min(8, (int)body["warmShare"].num(2))) : engine.warmShare;
            double sure = body.has("draftSure") ? std::max(0.0, std::min(0.99, body["draftSure"].num(0.6))) : engine.draftSure;
            bool guessing = body.has("speculate") ? body["speculate"].truthy(true) : engine.speculate;
            bool skipping = body.has("fastRouting") ? body["fastRouting"].truthy(true) : engine.fastRouting;
            bool reading = body.has("foresee") ? body["foresee"].truthy(false) : engine.foresee;
            engine.holdRam = body.has("holdRam") ? body["holdRam"].truthy(true) : engine.holdRam;
            bool seeing = body.has("vision") ? body["vision"].truthy(false) : engine.vision;
            int pinning = body.has("pinShare") ? std::max(10, std::min(95, (int)body["pinShare"].num(50))) : engine.pinShare;
            double pinSize = body.has("pinGb") ? std::max(0.0, std::min(4096.0, body["pinGb"].num(0))) : engine.pinGb;
            bool whole = body.has("ramWhole") ? body["ramWhole"].truthy(false) : engine.ramWhole;
            int seeingGpu = body.has("visionGpu") ? (int)body["visionGpu"].num(-1) : engine.visionGpu;
            if(seeingGpu < -1 || seeingGpu >= (int)engine.gpus.size()) seeingGpu = -1;
            gpusMoved = pickedGpu != engine.mainGpu || helping != engine.secondGpu || helperPick != engine.helperGpu || helperRoom != engine.helperGb || helperAhead != engine.helperFirst || frontCap != engine.frontCap || drafts != engine.draftLength || reach != engine.copyReach || plan != engine.cpuPlan || share != engine.warmShare
                || sure != engine.draftSure || guessing != engine.speculate || skipping != engine.fastRouting || reading != engine.foresee || seeing != engine.vision || seeingGpu != engine.visionGpu || pinning != engine.pinShare || pinSize != engine.pinGb || whole != engine.ramWhole;
            engine.ramWhole = whole;
            engine.pinShare = pinning;
            engine.pinGb = pinSize;
            engine.vision = seeing;
            engine.visionGpu = seeingGpu;
            engine.mainGpu = pickedGpu;
            engine.secondGpu = helping;
            engine.helperGpu = helperPick;
            engine.helperGb = helperRoom;
            engine.helperFirst = helperAhead;
            engine.frontCap = frontCap;
            engine.draftLength = drafts;
            engine.copyReach = reach;
            engine.cpuPlan = plan;
            engine.warmShare = share;
            engine.draftSure = sure;
            engine.speculate = guessing;
            engine.fastRouting = skipping;
            engine.foresee = reading;
            savePool();
            replan();
            split = !engine.frontNode.empty() || brain.firstLayer > 0;
        }
        applyMemoryCap();
        if(brainState == 2 && brain.gpu) brain.gpu->holdRam = engine.holdRam;
        if(resized) std::cout << "context set to " << engine.context << " tokens, reloading the model" << std::endl;
        if((switchBackend || resized || gpusMoved) && split) splitDirty = true;
        if(limitsMoved && brainState != 0) std::cout << "memory limits changed, reloading the model" << std::endl;
        if(gpusMoved && brainState != 0) std::cout << "gpu roles changed, reloading the model" << std::endl;
        if((switchBackend || resized || limitsMoved || gpusMoved) && !split && !engine.modelPath.empty() && brainState != 0) std::thread(loadBrain, engine.modelPath, nextLoad()).detach();
        replyHttp(client, 200, poolJson());
        closesocket(client);
        return;
    }

    if(route == "POST /api/pool/request/accept"){
        membershipDirty = true;
        std::string id = body["id"].str();
        JoinRequest wanted;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            for(const JoinRequest& r : engine.joins){
                if(r.id == id && r.status == "pending") wanted = r;
            }
        }

        if(wanted.id.empty()){
            replyHttp(client, 404, "{\"error\":\"that join request is gone, ask the other PC to send it again\"}");
            closesocket(client);
            return;
        }

        std::string why;
        std::string name = connectNode(wanted.ip, wanted.tcpPort, wanted.token, why);
        bool joined = false;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            for(JoinRequest& r : engine.joins){
                if(r.id != id) continue;
                r.problem = name.empty() ? why : "";
                if(name.empty()) continue;
                r.status = "accepted";
                r.when = nowMs();
            }

            auto found = name.empty() ? engine.nodes.end() : engine.nodes.find(name);
            if(found != engine.nodes.end()){
                found->second->accepted = true;
                engine.trusted.erase(std::remove_if(engine.trusted.begin(), engine.trusted.end(), [&](const TrustedNode& t){ return t.name == name; }), engine.trusted.end());
                engine.trusted.push_back({name, wanted.ip, wanted.tcpPort, wanted.token});

                bool hasSlots = false;
                for(const PoolSlot& s : engine.slots){
                    if(s.node == name) hasSlots = true;
                }

                if(!hasSlots){
                    size_t gpuAt = 0, ramAt = 0;
                    for(size_t i = 0; i < engine.slots.size(); i++){
                        if(engine.slots[i].kind == SLOT_GPU) gpuAt = i + 1;
                        if(engine.slots[i].kind <= SLOT_RAM) ramAt = i + 1;
                    }
                    if(found->second->stats.gpu.present) {
                        PoolSlot gpu = {name, SLOT_GPU, floor((double)found->second->stats.gpu.vramTotal * 0.9 / (double)gib * 2.0) / 2.0, 0};
                        engine.slots.insert(engine.slots.begin() + gpuAt, gpu);
                        ramAt++;
                    }
                    PoolSlot ram = {name, SLOT_RAM, 0, 0};
                    ram.keep = 8;
                    engine.slots.insert(engine.slots.begin() + ramAt, ram);
                }
                savePool();
                replan();
                joined = true;
                std::cout << "node " << name << " joined the pool from " << wanted.ip << std::endl;
            }
        }

        if(!joined){
            replyHttp(client, 502, "{\"error\":" + quoteJson(why.empty() ? "the worker did not stay connected" : why) + "}");
            closesocket(client);
            return;
        }

        applyMemoryCap();
        replyHttp(client, 200, poolJson());
        closesocket(client);
        return;
    }

    if(route == "POST /api/pool/request/decline"){
        std::string id = body["id"].str();
        bool found = false;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            for(JoinRequest& r : engine.joins){
                if(r.id != id || r.status != "pending") continue;
                r.status = "declined";
                r.when = nowMs();
                found = true;
                std::cout << "join request from " << r.name << " at " << r.ip << " declined" << std::endl;
            }
        }

        if(!found){
            replyHttp(client, 404, "{\"error\":\"that join request is gone\"}");
            closesocket(client);
            return;
        }

        replyHttp(client, 200, poolJson());
        closesocket(client);
        return;
    }

    if(route == "POST /api/pool/network"){
        double door = body["doorPort"].num(-1);
        std::string listen = body["apiListen"].str();
        bool badDoor = body.has("doorPort") && (door < 1 || door > 65535 || door != floor(door));
        bool badListen = body.has("apiListen") && listen != "local" && listen != "network";
        bool badAccept = body.has("acceptRequests") && body["acceptRequests"].kind != JSON_BOOL;

        if(badDoor || badListen || badAccept){
            replyHttp(client, 400, "{\"error\":\"acceptRequests must be true or false, doorPort a port from 1 to 65535 and apiListen local or network\"}");
            closesocket(client);
            return;
        }

        {
            std::lock_guard<std::mutex> guard(engine.lock);
            if(body.has("doorPort") && (int)door == engine.apiPort){
                replyHttp(client, 400, "{\"error\":\"the door port cannot be the engine API port\"}");
                closesocket(client);
                return;
            }
            if(body.has("acceptRequests")) engine.acceptRequests = body["acceptRequests"].truthy();
            if(body.has("doorPort")) engine.doorPort = (int)door;
            if(body.has("apiListen")) engine.apiListen = listen;
            savePool();
        }

        replyHttp(client, 200, poolJson());
        closesocket(client);
        return;
    }

    if(route == "POST /api/pool/remove"){
        membershipDirty = true;
        std::string name = body["name"].str();
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            auto found = engine.nodes.find(name);
            if(found != engine.nodes.end()){
                std::shared_ptr<RemoteNode> node = found->second;
                if(node->connected){
                    std::lock_guard<std::mutex> wire(node->wireLock);
                    sendPacket(node->wire, PACKET_BYE, nullptr, 0);
                }
                if(node->wire != INVALID_SOCKET) closesocket(node->wire);
                engine.nodes.erase(found);
            }
            for(const TrustedNode& t : engine.trusted){
                if(t.name == name) engine.refused.erase(t.ip + ":" + std::to_string(t.port));
            }
            engine.trusted.erase(std::remove_if(engine.trusted.begin(), engine.trusted.end(), [&](const TrustedNode& t){ return t.name == name; }), engine.trusted.end());
            engine.slots.erase(std::remove_if(engine.slots.begin(), engine.slots.end(), [&](const PoolSlot& s){ return s.node == name; }), engine.slots.end());
            savePool();
            replan();
        }
        applyMemoryCap();
        replyHttp(client, 200, poolJson());
        closesocket(client);
        return;
    }

    if(route == "POST /api/pool/apply"){
        bool ready = false;
        bool split = false;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            ready = engine.loaded && engine.plan.ok;
            split = !engine.frontNode.empty() || brain.firstLayer > 0;
        }
        if(ready && split) splitDirty = true;
        if(ready && !split){
            std::thread(pushWeights).detach();
            std::thread(loadLocal).detach();
        }
        replyHttp(client, ready ? 200 : 400, ready ? poolJson() : "{\"error\":\"the plan is not valid yet\"}");
        closesocket(client);
        return;
    }

    if(route == "POST /api/pool/unload"){
        local.generation++;
        releaseAll(local);
        releaseNodes();
        std::cout << "memory released on every node" << std::endl;
        replyHttp(client, 200, poolJson());
        closesocket(client);
        return;
    }

    if(route == "GET /v1/models"){
        std::string folder;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            folder = engine.modelFolder;
        }
        std::ostringstream out;
        out << "{\"object\":\"list\",\"data\":[";
        auto found = scanModels(folder);
        for(size_t i = 0; i < found.size(); i++){
            out << (i ? "," : "") << "{\"id\":" << quoteJson(found[i].name) << ",\"object\":\"model\",\"created\":0,\"owned_by\":\"sneo\"}";
        }
        out << "]}";
        replyHttp(client, 200, out.str());
        closesocket(client);
        return;
    }

    if(route == "GET /api/models" || route == "GET /api/tags" || route == "POST /api/models/scan"){
        std::string folder;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            if(route == "POST /api/models/scan" && !body["dir"].str().empty() && body["dir"].str() != engine.modelFolder){
                engine.modelFolder = body["dir"].str();
                savePool();
            }
            folder = engine.modelFolder;
        }
        auto found = scanModels(folder);
        std::ostringstream out;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            out << "{\"status\":\"ok\",\"currentModel\":" << quoteJson(engine.loaded ? engine.modelPath : "") << ",\"loaded\":" << (engine.loaded ? "true" : "false")
                << ",\"ready\":" << (brainState == 2 ? "true" : "false") << ",\"warming\":" << (brainState == 1 ? "true" : "false");
            if(brainState == -1) out << ",\"problem\":" << quoteJson(brainProblem);
            if(brainState == 1){
                std::lock_guard<std::mutex> step(loadStatus.lock);
                out << ",\"load\":{\"step\":" << quoteJson(stopLoading ? std::string("Stopping") : loadStatus.step) << ",\"done\":" << loadStatus.done.load() << ",\"total\":" << loadStatus.total.load() << "}";
            }
            if(engine.healedAt) out << ",\"healed\":{\"seconds\":" << (nowMs() - engine.healedAt) / 1000 << ",\"mb\":" << engine.healedMb << "}";
        }
        {
            std::lock_guard<std::mutex> held(preparing.lock);
            out << ",\"architectures\":[\"qwen4exp\",\"qwen35moe\",\"qwen35\",\"gemma4\",\"hy_v4_flash\"]";
            out << ",\"prepare\":{\"task\":" << quoteJson(preparing.task) << ",\"done\":" << preparing.done << ",\"total\":" << preparing.total
                << ",\"problem\":" << quoteJson(preparing.problem) << ",\"note\":" << quoteJson(preparing.note) << "}";
        }
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            out
                << ",\"folder\":" << quoteJson(folder) << ",\"total\":" << found.size() << ",\"models\":[";
        }
        for(size_t i = 0; i < found.size(); i++){
            out << (i ? "," : "") << "{\"name\":" << quoteJson(found[i].name) << ",\"path\":" << quoteJson(found[i].path) << ",\"size\":" << found[i].bytes << ",\"shared\":" << (found[i].shared ? "true" : "false") << "}";
        }
        out << "]}";
        replyHttp(client, 200, out.str());
        closesocket(client);
        return;
    }

    if(route == "POST /api/shutdown"){
        sockaddr_in peer = {};
        int length = sizeof(peer);
        bool local = getpeername(client, (sockaddr*)&peer, &length) == 0 && peer.sin_family == AF_INET && (ntohl(peer.sin_addr.s_addr) >> 24) == 127;
        if(!local){
            replyHttp(client, 403, "{\"error\":\"only this PC can stop the engine\"}");
            closesocket(client);
            return;
        }
        replyHttp(client, 200, "{\"stopping\":true}");
        closesocket(client);
        std::thread([]{
            std::cout << "shutting down, releasing the model" << std::endl;
            stopGenerating = true;
            nextLoad();
            std::unique_lock<std::mutex> busy(thinking, std::defer_lock);
            for(int i = 0; i < 50 && !busy.owns_lock(); i++){
                if(!busy.try_lock()) Sleep(100);
            }
            if(busy.owns_lock()){
                closeFront();
                releaseNodes();
                releaseBrain();
                std::cout << "memory released, the engine stops" << std::endl;
            }
            std::cout.flush();
            ExitProcess(0);
        }).detach();
        return;
    }

    if(route == "POST /api/models/stop"){
        bool stopping = brainState == 1;
        nextLoad();
        replyHttp(client, 200, std::string("{\"stopping\":") + (stopping ? "true" : "false") + "}");
        closesocket(client);
        return;
    }

    if(route == "POST /api/models/unload"){
        stopGenerating = true;
        nextLoad();
        {
            std::unique_lock<std::mutex> busy(thinking);
            releaseBrain();
        }
        stopGenerating = false;
        releaseNodes();
        std::cout << "model unloaded, video memory and ram released" << std::endl;
        replyHttp(client, 200, poolJson());
        closesocket(client);
        return;
    }

    if(route == "POST /api/models/load"){
        std::string path = body["path"].str();
        if(path.empty() && body["name"].str().empty()){
            std::lock_guard<std::mutex> guard(engine.lock);
            path = engine.modelPath;
        }
        if(path.empty()){
            std::string folder;
            {
                std::lock_guard<std::mutex> guard(engine.lock);
                folder = engine.modelFolder;
            }
            for(auto& item : scanModels(folder)){
                if(item.name == body["name"].str()) path = item.path;
            }
        }
        bool ok = !path.empty() && loadModel(path);
        if(ok){
            std::lock_guard<std::mutex> guard(engine.lock);
            engine.oomKeep = 0;
            savePool();
        }
        if(ok) std::thread(loadBrain, path, nextLoad()).detach();
        std::lock_guard<std::mutex> guard(engine.lock);
        std::ostringstream out;
        out << "{\"success\":" << (ok ? "true" : "false") << ",\"cargado\":" << (engine.loaded ? "true" : "false") << ",\"ruta\":" << quoteJson(engine.modelPath)
            << ",\"capas\":" << engine.model.layers << ",\"expertos\":" << engine.model.experts << ",\"error\":" << quoteJson(engine.loadError) << "}";
        replyHttp(client, 200, out.str());
        closesocket(client);
        return;
    }

    if(route == "GET /api/status" || route == "POST /api/limits"){
        std::lock_guard<std::mutex> guard(engine.lock);
        std::ostringstream out;
        out << "{\"master\":{\"gpu\":" << quoteJson(engine.selfStats.gpu.name) << ",\"vramLibreGb\":" << (double)engine.selfStats.gpu.vramFree / gib << "}"
            << ",\"cargado\":" << (engine.loaded ? "true" : "false") << ",\"modelo\":" << quoteJson(engine.modelPath)
            << ",\"success\":true,\"plan\":" << planJson(engine.plan) << "}";
        replyHttp(client, 200, out.str());
        closesocket(client);
        return;
    }

    replyHttp(client, 404, "{\"error\":\"unknown route " + quoteJson(route).substr(1));
    closesocket(client);
}

void handleDoor(SOCKET client, std::string ip) {
    DWORD timeout = 4000;
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
    setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, (char*)&timeout, sizeof(timeout));

    uint64_t now = nowMs();
    bool flooded = false;
    bool member = false;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        engine.joins.erase(std::remove_if(engine.joins.begin(), engine.joins.end(), [&](const JoinRequest& r){ return now - r.when > 600000; }), engine.joins.end());

        if(engine.visitors.size() > 256){
            for(auto it = engine.visitors.begin(); it != engine.visitors.end();){
                bool stale = it->second.hits.empty() || now - it->second.hits.back() > 60000;
                it = stale ? engine.visitors.erase(it) : ++it;
            }
        }

        for(auto& n : engine.nodes){
            if(n.second->connected && n.second->accepted && n.second->ip == ip) member = true;
        }
        DoorVisitor& visitor = engine.visitors[ip];
        while(!visitor.hits.empty() && now - visitor.hits.front() > 60000) visitor.hits.pop_front();
        if(!member) visitor.hits.push_back(now);
        flooded = visitor.hits.size() > 60;
    }

    if(flooded){
        replyHttp(client, 429, "{\"error\":\"too many requests, wait a minute\"}");
        closesocket(client);
        return;
    }

    HttpRequest req;
    if(!readHttp(client, req, 4096)){
        replyHttp(client, 400, "{\"error\":\"bad request\"}");
        closesocket(client);
        return;
    }

    if(req.method == "GET" && req.path == "/pool"){
        if(!member){
            replyHttp(client, 403, "{\"error\":\"only machines in this pool can read it\"}");
            closesocket(client);
            return;
        }
        replyHttp(client, 200, poolJson());
        closesocket(client);
        return;
    }

    if(req.method == "GET" && req.path == "/hello"){
        std::string links;
        ULONG size = 16384;
        std::vector<uint8_t> buffer(size);
        ULONG flags = GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER;
        ULONG result = GetAdaptersAddresses(AF_INET, flags, nullptr, (IP_ADAPTER_ADDRESSES*)buffer.data(), &size);
        if(result == ERROR_BUFFER_OVERFLOW){
            buffer.resize(size);
            result = GetAdaptersAddresses(AF_INET, flags, nullptr, (IP_ADAPTER_ADDRESSES*)buffer.data(), &size);
        }
        for(IP_ADAPTER_ADDRESSES* adapter = result == NO_ERROR ? (IP_ADAPTER_ADDRESSES*)buffer.data() : nullptr; adapter; adapter = adapter->Next){
            if(adapter->OperStatus != IfOperStatusUp || adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;
            std::string kind = adapter->IfType == IF_TYPE_IEEE80211
                ? "wifi"
                : adapter->IfType == IF_TYPE_ETHERNET_CSMACD
                    ? "ethernet"
                    : "other";
            for(IP_ADAPTER_UNICAST_ADDRESS* unicast = adapter->FirstUnicastAddress; unicast; unicast = unicast->Next){
                char ip[64] = {};
                inet_ntop(AF_INET, &((sockaddr_in*)unicast->Address.lpSockaddr)->sin_addr, ip, sizeof(ip));
                if(strncmp(ip, "127.", 4) == 0) continue;
                links += std::string(links.empty() ? "" : ",") + "{\"ip\":" + quoteJson(ip) + ",\"kind\":\"" + kind + "\",\"mbps\":" + std::to_string(adapter->TransmitLinkSpeed / 1000000ULL) + "}";
            }
        }
        std::string hello;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            hello = "{\"sneo\":true,\"name\":" + quoteJson(engine.selfName) + ",\"gpu\":" + quoteJson(std::string(engine.selfStats.gpu.name))
                + ",\"vramTotal\":" + std::to_string(engine.selfStats.gpu.vramTotal) + ",\"ramTotal\":" + std::to_string(engine.selfStats.host.ramTotal)
                + ",\"accepting\":" + (engine.acceptRequests ? "true" : "false") + ",\"model\":" + quoteJson(engine.model.name) + ",\"links\":[" + links + "]}";
        }
        replyHttp(client, 200, hello);
        closesocket(client);
        return;
    }

    if(req.method == "GET" && req.path.rfind("/join/", 0) == 0){
        std::string id = req.path.substr(6);
        std::string status;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            for(const JoinRequest& r : engine.joins){
                if(r.id == id && r.ip == ip) status = r.status;
            }
        }

        replyHttp(client, status.empty() ? 404 : 200, status.empty() ? "{\"error\":\"unknown request\"}" : "{\"status\":" + quoteJson(status) + "}");
        closesocket(client);
        return;
    }

    if(req.method != "POST" || req.path != "/join"){
        replyHttp(client, 404, "{\"error\":\"not found\"}");
        closesocket(client);
        return;
    }

    Json body = parseJson(req.body);
    std::string name = body["name"].str();
    std::string token = body["token"].str();
    double tcpPort = body["tcpPort"].num();
    bool clean = !name.empty() && name.size() < 64 && token.size() >= 16 && token.size() < sizeof(HelloBody::key) && tcpPort >= 1 && tcpPort <= 65535 && tcpPort == floor(tcpPort);
    for(char c : name){
        if((unsigned char)c < 32) clean = false;
    }
    for(char c : token){
        if(!isalnum((unsigned char)c)) clean = false;
    }

    if(!clean){
        replyHttp(client, 400, "{\"error\":\"invalid join request\"}");
        closesocket(client);
        return;
    }

    if(body["version"].num() != protocolVersion){
        replyHttp(client, 400, "{\"error\":\"this pool runs another Sneo version, update worker.exe\"}");
        closesocket(client);
        return;
    }

    JoinRequest request;
    request.id = makeKey();
    request.ip = ip;
    request.name = name;
    request.tcpPort = (int)tcpPort;
    request.token = token;
    request.gpu = body["gpu"].str().substr(0, 95);
    request.vramTotal = (uint64_t)std::max(0.0, body["vramTotal"].num());
    request.ramTotal = (uint64_t)std::max(0.0, body["ramTotal"].num());
    request.cpu = body["cpu"].str().substr(0, 95);
    request.when = now;

    int status = 200;
    {
        std::lock_guard<std::mutex> guard(engine.lock);
        DoorVisitor& visitor = engine.visitors[ip];
        while(!visitor.joins.empty() && now - visitor.joins.front() > 60000) visitor.joins.pop_front();

        auto same = [&](const JoinRequest& r){ return r.status == "pending" && r.ip == ip && r.tcpPort == request.tcpPort; };
        size_t pending = 0;
        for(const JoinRequest& r : engine.joins){
            if(r.status == "pending" && !same(r)) pending++;
        }

        status = !engine.acceptRequests
            ? 403
            : visitor.joins.size() >= 5
                ? 429
                : pending >= 8
                    ? 503
                    : 200;

        if(status == 200){
            engine.joins.erase(std::remove_if(engine.joins.begin(), engine.joins.end(), same), engine.joins.end());
            engine.joins.push_back(request);
            visitor.joins.push_back(now);
            std::cout << "join request from " << name << " at " << ip << ":" << request.tcpPort << std::endl;
        }
    }

    std::string reply = status == 200
        ? "{\"id\":" + quoteJson(request.id) + ",\"status\":\"pending\"}"
        : status == 403
            ? "{\"error\":\"this pool is not taking join requests\"}"
            : status == 429
                ? "{\"error\":\"too many join requests from this address, wait a minute\"}"
                : "{\"error\":\"the pool owner has too many pending requests, try again later\"}";
    replyHttp(client, status, reply);
    closesocket(client);
}

void doorLoop() {
    SOCKET door = INVALID_SOCKET;
    int openPort = 0;
    int failedPort = 0;
    uint64_t failedAt = 0;

    while(true){
        bool wanted = false;
        int port = 0;
        {
            std::lock_guard<std::mutex> guard(engine.lock);
            wanted = engine.acceptRequests;
            port = engine.doorPort;
            if(!wanted) engine.doorProblem.clear();
        }

        if(door != INVALID_SOCKET && (!wanted || port != openPort)){
            closesocket(door);
            door = INVALID_SOCKET;
            std::cout << "pool door on port " << openPort << " closed" << std::endl;
            std::lock_guard<std::mutex> guard(engine.lock);
            engine.doorOpen = false;
        }

        if(!wanted) failedPort = 0;

        if(door == INVALID_SOCKET && wanted && (port != failedPort || nowMs() - failedAt > 10000)){
            door = listenTcp(port, "0.0.0.0");
            if(door == INVALID_SOCKET && port != failedPort) std::cout << "cannot open the pool door on port " << port << std::endl;
            if(door != INVALID_SOCKET) std::cout << "pool door open for join requests on port " << port << std::endl;
            openPort = door == INVALID_SOCKET ? 0 : port;
            failedPort = door == INVALID_SOCKET ? port : 0;
            failedAt = nowMs();
            std::lock_guard<std::mutex> guard(engine.lock);
            engine.doorOpen = door != INVALID_SOCKET;
            engine.doorProblem = door == INVALID_SOCKET ? "port " + std::to_string(port) + " is taken by another program" : "";
        }

        if(door == INVALID_SOCKET){
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        fd_set ready;
        FD_ZERO(&ready);
        FD_SET(door, &ready);
        timeval wait = {0, 500000};
        if(select(0, &ready, nullptr, nullptr, &wait) != 1) continue;

        sockaddr_in from = {};
        int length = sizeof(from);
        SOCKET client = accept(door, (sockaddr*)&from, &length);
        if(client == INVALID_SOCKET) continue;

        if(doorGuests >= 32){
            closesocket(client);
            continue;
        }

        char ip[64] = {};
        inet_ntop(AF_INET, &from.sin_addr, ip, sizeof(ip));
        doorGuests++;
        std::thread([client, address = std::string(ip)](){
            handleDoor(client, address);
            doorGuests--;
        }).detach();
    }
}

int main(int argc, char* argv[]) {
    keepFullSpeed();
    std::string missing = missingCpuFeatures();
    if(!missing.empty()){
        printf("this processor lacks%s, which the engine needs (any Intel since Haswell or AMD since Ryzen has them)\n", missing.c_str());
        return 1;
    }
    int httpPort = 8000;
    if(argc > 1) httpPort = std::stoi(argv[1]);
    if(argc > 4) engine.modelFolder = argv[4];
    if(argc > 4) engine.folderArg = argv[4];
    std::string firstModel = argc > 5 && std::string(argv[5]).rfind("--", 0) != 0 ? argv[5] : "";
    bool asked = !firstModel.empty();
    for(int i = 1; i + 1 < argc; i++){
        if(std::string(argv[i]) == "--forward") engine.backendPort = std::stoi(argv[i + 1]);
        // un segundo motor en la misma carpeta con otro modelo lee su propia configuracion
        if(std::string(argv[i]) == "--pool") engine.poolFile = argv[i + 1];
    }
    char keptText[32] = {};
    if(GetEnvironmentVariableA("SNEO_OOM_KEEP", keptText, sizeof(keptText)) > 0) engine.oomKeep = std::min<uint64_t>(std::strtoull(keptText, nullptr, 10), 6ULL << 30);
    // al reiniciarse solo el motor viejo todavia tiene la ram fijada de su carga: si se mide antes el reparto sale con la mitad de la ram
    char parentText[32] = {};
    if(GetEnvironmentVariableA("SNEO_PARENT", parentText, sizeof(parentText)) > 0){
        SetEnvironmentVariableA("SNEO_PARENT", nullptr);
        uint64_t began = nowMs();
        HANDLE parent = OpenProcess(SYNCHRONIZE, FALSE, (DWORD)std::strtoul(parentText, nullptr, 10));
        if(parent){
            WaitForSingleObject(parent, 60000);
            CloseHandle(parent);
        }
        uint64_t most = 0;
        for(int steady = 0; steady < 4 && nowMs() - began < 30000; Sleep(250)){
            MEMORYSTATUSEX memory = {};
            memory.dwLength = sizeof(memory);
            GlobalMemoryStatusEx(&memory);
            steady = memory.ullAvailPhys < most + (256ULL << 20) ? steady + 1 : 0;
            most = std::max<uint64_t>(most, memory.ullAvailPhys);
        }
        std::cout << "waited " << (nowMs() - began) / 1000.0 << " s for the old engine to give back its memory, " << (double)most / gib << " GB of ram free" << std::endl;
    }

    if(!startSockets()){
        std::cout << "winsock failed" << std::endl;
        return 1;
    }

    engine.selfStats = readNode(engine.modelFolder);
    engine.selfName = engine.selfStats.host.nodeName;
    engine.selfScreen = gpuShowsScreen();
    engine.apiPort = httpPort;
    std::string startFolder = engine.modelFolder;
    loadPool();
    if(engine.modelFolder != startFolder) engine.selfStats = readNode(engine.modelFolder);
    engine.gpus = listGpus();
    int mainPick = engine.mainGpu >= 0 && engine.mainGpu < (int)engine.gpus.size() ? engine.mainGpu : 0;
    if(mainPick < (int)engine.gpus.size() && engine.gpus[mainPick].nvml){
        nvmlLink().device = engine.gpus[mainPick].nvml;
        engine.selfScreen = gpuShowsScreen();
        engine.selfStats = readNode(engine.modelFolder);
    }
    std::error_code absent;
    engine.apiBound = engine.apiListen;
    std::string apiHost = engine.apiBound == "network" ? "0.0.0.0" : "127.0.0.1";

    std::cout << "Sneo master on " << engine.selfName << " port " << httpPort << std::endl;
    std::cout << "gpu: " << engine.selfStats.gpu.name << " | vram free " << (double)engine.selfStats.gpu.vramFree / gib << " GB | ram free " << (double)engine.selfStats.host.ramFree / gib << " GB" << std::endl;
    for(const GpuCard& g : engine.gpus){
        if(engine.gpus.size() < 2) break;
        std::cout << "  gpu " << g.ordinal << ": " << g.name << " " << (double)g.vram / gib << " GB, compute " << g.major << "." << g.minor
            << (g.ordinal == mainPick ? ", main" : engine.secondGpu && (engine.helperGpu < 0 || engine.helperGpu == g.ordinal) ? ", helps with experts" : ", not used") << std::endl;
    }

    auto found = scanModels(engine.modelFolder);
    std::cout << "models in " << engine.modelFolder << ": " << found.size() << std::endl;
    for(auto& item : found) std::cout << "  " << item.name << (item.shared ? " from another app" : "") << " (" << (double)item.bytes / gib << " GB)" << std::endl;
    if(firstModel.empty() && !engine.savedModel.empty() && fs::exists(engine.savedModel, absent)) firstModel = engine.savedModel;
    if(firstModel.empty() && !found.empty()) firstModel = found[0].path;
    if(!firstModel.empty() && loadModel(firstModel) && engine.backendPort == 0){
        if(asked) std::thread(loadBrain, firstModel, nextLoad()).detach();
        else std::cout << "the model loads with the first request, nothing is in memory until then" << std::endl;
    }
    if(engine.backendPort > 0) std::cout << "chat requests are forwarded to port " << engine.backendPort << std::endl;

    SOCKET http = listenTcp(httpPort, apiHost);
    for(int i = 0; i < 40 && http == INVALID_SOCKET; i++){
        Sleep(250);
        http = listenTcp(httpPort, apiHost);
    }
    if(http == INVALID_SOCKET){
        std::cout << "cannot listen on " << apiHost << ":" << httpPort << std::endl;
        return 1;
    }

    std::thread(doorLoop).detach();
    std::thread(statsLoop).detach();
    std::thread(reconnectLoop).detach();
    std::thread(membershipLoop).detach();

    std::cout << "openai api on http://" << apiHost << ":" << httpPort << "/v1" << (engine.apiBound == "network" ? " (open to the local network)" : " (this pc only)") << std::endl;

    while(true){
        SOCKET client = accept(http, nullptr, nullptr);
        if(client == INVALID_SOCKET) continue;
        int prompt = 1;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, (char*)&prompt, sizeof(prompt));
        std::thread(handleClient, client).detach();
    }
}

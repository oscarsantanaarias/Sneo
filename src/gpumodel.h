#pragma once
#include "platform.h"
#include "qwen4.h"
#include "gpukernels.h"
#include "volta.h"
#include "hostdot.h"
#include "pack.h"
#include "split.h"
#include "vision.h"
#include "budget.h"
#include <array>
#include <iostream>
#include <set>
#include <unordered_map>
#include <filesystem>

struct GpuMatrix {
    uint64_t data = 0;
    uint32_t type = 0;
    int rows = 0;
    int cols = 0;
    uint64_t stride = 0;
};

#include "gemma4.h"
#include "hy4.h"
#include "safetensors.h"

inline std::atomic<bool> stopLoading{false};

// el paso de la carga y cuanto lleva para que la app no muestre solo un cargando sin avance
struct LoadStatus {
    std::mutex lock;
    std::string step;
    std::atomic<uint64_t> done{0};
    std::atomic<uint64_t> total{0};
};
inline LoadStatus loadStatus;

inline void loadSay(const std::string& step) {
    std::lock_guard<std::mutex> guard(loadStatus.lock);
    loadStatus.step = step;
    loadStatus.done = 0;
    loadStatus.total = 0;
}

inline std::string heatFor(const std::string& modelPath, size_t layerCount, int width) {
    std::filesystem::path folder = std::filesystem::path(runtimeFolder()).parent_path() / "profiles";
    std::string stem = std::filesystem::path(modelPath).stem().string();
    std::string best;
    size_t bestShared = 0;
    uint64_t bestSeen = 0;
    std::error_code error;
    for(const auto& entry : std::filesystem::directory_iterator(folder, error)){
        if(entry.path().extension() != ".heat") continue;
        FILE* f = nullptr;
        fopen_s(&f, entry.path().string().c_str(), "rb");
        if(!f) continue;
        char tag[4] = {};
        uint32_t version = 0, count = 0, columns = 0;
        uint64_t seen = 0;
        bool ok = fread(tag, 1, 4, f) == 4 && fread(&version, 4, 1, f) == 1 && fread(&count, 4, 1, f) == 1 && fread(&columns, 4, 1, f) == 1 && fread(&seen, 8, 1, f) == 1;
        fclose(f);
        if(!ok || memcmp(tag, "HEAT", 4) != 0 || version != 1 || count != layerCount || columns != (uint32_t)width) continue;
        std::string name = entry.path().stem().string();
        size_t shared = 0;
        while(shared < name.size() && shared < stem.size() && tolower((unsigned char)name[shared]) == tolower((unsigned char)stem[shared])) shared++;
        if(shared == stem.size() && shared == name.size()) return entry.path().string();
        if(best.empty() || shared > bestShared || (shared == bestShared && seen > bestSeen)){
            best = entry.path().string();
            bestShared = shared;
            bestSeen = seen;
        }
    }
    return best;
}

struct GpuLayer {
    bool attention = false;
    uint64_t hcNorm[2] = {};
    GpuMatrix hcDown[2], hcUp[2], hcInject[2];

    GpuMatrix qkv, gate, alpha, beta, ssmOut;
    uint64_t convWeight = 0;
    uint64_t decay = 0;
    uint64_t dtBias = 0;
    uint64_t ssmNorm = 0;

    GpuMatrix query, key, value, attnOut, indexQuery, indexKey;
    uint64_t queryNorm = 0;
    uint64_t keyNorm = 0;
    uint64_t indexQueryNorm = 0;
    uint64_t indexKeyNorm = 0;

    GpuMatrix router, sharedScalar, sharedGate, sharedUp, sharedDown;
    bool hostFfn = false;
    Matrix hostGate, hostUp, hostDown;
    Matrix expertGate, expertUp, expertDown;
    const TensorInfo* expertTensor[3] = {};
    uint64_t resident[3] = {};
    std::vector<int> slot;
    std::vector<int> lastRouting;
    uint64_t slots = 0;
    uint64_t slice = 0;
    uint64_t piece[3] = {};
    std::vector<int> warm;
    uint64_t warmSlots = 0;
    uint64_t warmBase[3] = {};
    uint8_t* warmHost[3] = {};
    std::vector<int> help;
    uint64_t helpBase[3] = {};

    uint64_t state = 0;
    uint64_t convHistory = 0;
    uint64_t keys = 0;
    uint64_t values = 0;
    uint64_t indexRaw = 0;
    uint64_t pooled = 0;
};

struct HelperCard {
    bool ready = false;
    void* card = nullptr;
    bool pascalPair = false;
    int device = -1;
    int major = 0;
    int minor = 0;
    void* context = nullptr;
    KernelModule module;
    GridTables tables = {};
    std::vector<uint64_t> allocations;
    std::vector<void*> pinned;
    uint64_t gateOut = 0, upOut = 0, hiddenOut = 0;
    float* inRows = nullptr;
    int* slotList = nullptr;
    float* outRows = nullptr;
    uint64_t inMapped = 0, slotMapped = 0, outMapped = 0;
    uint64_t inRowsOnCard = 0, slotsOnCard = 0;
    PackedInput packedIn, packedHidden;
    void* marks[5] = {};
    void* done = nullptr;
    double partMs[4] = {};
    std::vector<int> where;
    int capacity = 0;
    int perLayer = 0;
    uint64_t bytes = 0;
    uint64_t rows = 0;
    std::string name;
    std::string problem;
};

struct Lift {
    int layer;
    int expert;
    int place;
    const uint8_t* from[3];
    uint64_t at[3];
    int source[3];
    bool pinnedCopy;
    bool landed;
    int room;
    bool down;
};

struct Lifter {
    std::thread thread;
    void* stream = nullptr;
    void* event = nullptr;
    uint8_t* stage = nullptr;
    std::vector<HANDLE> files;
};

struct GpuModel {
    ModelFile file;
    std::vector<MappedFile> maps;
    std::vector<GpuLayer> layers;
    std::vector<uint64_t> allocations;
    std::vector<void*> pinned;
    std::vector<void*> staging;
    ThreadPool pool;
    KernelModule module;
    GridTables tables = {};
    HelperCard helper;
    bool useHelper = true;
    bool helperFirst = false;
    int helperOrdinal = -1;
    uint64_t helperCap = 0;
    Matrix embed, pleTable;
    GpuMatrix head, outDown, outUp, pleKey, pleValue;
    uint64_t outNorm = 0;
    uint64_t pleNormKey = 0;
    uint64_t pleNormQuery = 0;
    uint64_t pleNormConv = 0;
    uint64_t pleConv = 0;
    uint64_t pleHistory = 0;
    uint64_t ropeCos = 0;
    uint64_t ropeSin = 0;
    double ropeTheta = 1e7;
    int ropeSpans[3] = {11, 11, 10};
    struct SeenPicture {
        uint64_t hash = 0;
        int high = 0;
        int wide = 0;
        uint64_t used = 0;
        std::vector<float> rows;
    };
    std::vector<SeenPicture> pictures;
    VisionEncoder seer;
    uint64_t pictureClock = 0;
    std::unordered_map<uint64_t, int> pictureSlots;
    int imageToken = -1;
    uint64_t placesOnGpu = 0;
    std::vector<int> hostPlaces;
    size_t ropeTailFrom = 0;
    int ropeTailShift = 0;
    uint64_t pleMultipliers[3] = {};
    uint64_t pleVocab[pleHeads] = {};
    uint64_t pleOffset[pleHeads] = {};

    uint64_t embedRows = 0, stack = 0, xn = 0, mixGate = 0, lo = 0, inj = 0, x = 0, y = 0;
    uint64_t qkv = 0, z = 0, alphaOut = 0, betaOut = 0, u = 0, o = 0;
    uint64_t q = 0, k = 0, v = 0, iq = 0, ik = 0, scores = 0, cells = 0, positionsOnGpu = 0;
    uint64_t routerOut = 0, scalar = 0, sharedGateOut = 0, sharedUpOut = 0, sharedHidden = 0, shared = 0;
    uint64_t routeIds = 0, routeWeights = 0, lists = 0, expertGateOut = 0, expertUpOut = 0, expertHidden = 0, expertOut = 0;
    uint64_t teamLists = 0;
    uint64_t pleEmbed = 0, pleKeyOut = 0, pleValueOut = 0, gated = 0, normd = 0, fin = 0, vocabOut = 0;
    PackedInput packs[2], fines[2];

    float* hostEmbed = nullptr;
    float* hostPle = nullptr;
    int* hostPositions = nullptr;
    int* hostIds = nullptr;
    float* hostX = nullptr;
    int* hostLists = nullptr;
    float* hostExpertOut = nullptr;
    float* hostLogits = nullptr;
    int* hostNext = nullptr;
    uint64_t mappedNext = 0, nextIds = 0, nextWeights = 0;
    bool foresee = false;
    uint64_t foreseen = 0, foreseenUsed = 0;
    std::vector<uint32_t> guessMark;
    uint64_t mappedEmbed = 0, mappedPle = 0, mappedPositions = 0, mappedIds = 0, mappedX = 0, mappedLists = 0, mappedExpertOut = 0, mappedLogits = 0;
    uint64_t slotTable = 0;
    void* routed = nullptr;
    void* stream = nullptr;
    std::map<int, void*> tokenGraphs;
    std::set<int> graphless;
    std::set<void*> widerShared;
    volatile uint32_t* hostFlags = nullptr;
    bool timing = false;
    uint64_t mappedTimeline = 0;
    volatile unsigned long long* hostTimeline = nullptr;
    std::vector<double> cpuSaw, cpuDone, cpuGather;
    std::vector<int> cpuSeen;
    uint64_t mappedFlags = 0;
    uint64_t spillCounts = 0;
    uint32_t ticket = 0;
    int graphBudget = 0;
    float* spillX = nullptr;
    float* spillOut = nullptr;

    std::vector<int> expertTally, expertBegin, cpuExperts, cpuTokens;
    float* batchX = nullptr;
    int relayLimit = 4;
    int relayRows = 0;
    int promptRelay = 128;
    int teamRows = 128;
    int stageRing = 0;
    uint64_t planWidth = 0;
    int coldRoutes = 256;
    int coldRows = 3072;
    std::vector<std::vector<int>> coldPlan;
    uint64_t hostExpertRows = 0, hostXRows = 0;
    double coldMs = 0;
    uint64_t coldRouted = 0;
    bool packedAttention = true;
    bool feedWaiting = false;
    bool repayWaiting = false;
    bool deferTail = true;
    std::vector<int> feedIds, feedPositions;
    size_t feedAt = 0;
    int feedRows = 0;
    int cutRows[2] = {-1, -1};
    int cutSlots[2] = {-1, -1};
    bool cutKept[2] = {false, false};
    int crestCuts[2] = {-1, -1};
    uint64_t replayStore = 0;
    int replayRows = 0;
    bool recording = false;
    bool settling = false;
    bool leaveState = false;
    int firstLayer = 0;
    int endLayer = 0;
    int expertCount = 512;
    bool dense = false;
    bool gemma = false;
    GemmaShape gemmaShape;
    GemmaBuffers gemmaBuffers;
    std::vector<GemmaLayer> gemmaLayers;
    bool hy4 = false;
    HyShape hyShape;
    HyBuffers hyBuffers;
    std::vector<HyLayer> hyLayers;
    HyCache hyCache;
    HyDraft hyDraft;
    int hyTop = 0;
    void* hyRouted = nullptr;
    FrontLink* remote = nullptr;
    std::vector<float> frontStack;
    // en el nodo del pool: cada parte de filas lista sale apenas se copia de la tarjeta
    std::function<bool(int, int)> rowsReady;
    std::vector<uint64_t> pleRowIds;
    std::vector<uint64_t> aheadPle;
    std::future<bool> frontPending;
    bool frontAhead = false;
    bool pipeFront = false;
    int aheadRows = 0;
    double frontMs = 0;
    std::string draftPath;
    std::string vocabPath;
    GpuMatrix draftHead;
    std::vector<int> draftVocab;
    std::string draftProblem;
    bool drafting = false;
    bool tiedDraft = false;
    int shortlist = 0;
    GpuLayer draft;
    GpuMatrix draftEmbedProj, draftHiddenProj, draftOutDown, draftOutUp;
    uint64_t draftEmbedNorm = 0, draftHiddenNorm = 0, draftOutNorm = 0;
    uint64_t draftStack = 0, draftRows = 0, draftPending = 0, draftOwn = 0;
    int draftPendingAt = -1;
    bool draftConsumed = true;
    float draftSure = 0.6f;
    int draftBatch = 1024;
    // el borrador incluido en el modelo adivina solo entre los primeros 64K ids (98% de los tokens de codigo y de texto):
    // con la cabeza entera cada token adivinado leia ~1 GB en un 27B, asi tarda la mitad (medido 530 a 270 ms por 256 tokens)
    int draftHeadRows = 65536;
    double feedMs = 0;
    int tailSkip = 0;
    uint64_t skippedRoutes = 0;
    std::vector<float> cpuInput, cpuGate, cpuUp;
    std::vector<signed char> sides;
    HostPacked hostInput, hostHidden;

    std::vector<int> tokens;
    std::vector<double> layerMs;
    double embedMs = 0;
    double headMs = 0;
    double pleMs = 0;
    double expertMs = 0;
    double helperLaunchMs = 0, helperCpuMs = 0, helperWaitMs = 0;
    bool timeHelper = false;
    uint64_t helperCalls = 0;
    double helperLimit = 400.0;
    int helperDelay = 0;
    int hostDelay = 0;
    uint64_t coldExperts = 0;
    uint64_t spilledRows = 0;
    uint64_t missedRoutes = 0;
    uint64_t missedUnique = 0;
    double waitMs = 0;
    uint64_t routedCount = 0;
    uint64_t residentCount = 0;
    std::string heatPath;
    std::vector<uint64_t> heat;
    uint64_t heatTokens = 0;
    std::string heatSave;
    std::string heatSource;
    std::vector<uint64_t> learned;
    uint64_t heatClock = 0;
    std::vector<float> recent;
    int adaptEvery = 32;
    int adaptSwaps = 16;
    int sinceAdapt = 0;
    uint64_t swapCount = 0;
    std::vector<uint64_t> expertHits;
    uint64_t expertBudget = 0;
    uint64_t hostBudget = 0;
    uint64_t vramReserve = 3ULL << 30;
    uint64_t vramCap = 0;
    uint64_t extraKeep = 0;
    bool pinLost = false;
    uint64_t startBudget = 0;
    uint64_t startUsage = 0;
    uint64_t freeAtStart = 0;
    uint64_t cardAtStart = 0;
    uint64_t hostCap = 0;
    uint64_t denseBytes = 0;
    uint64_t cacheBytes = 0;
    uint64_t residentBytes = 0;
    uint64_t vramHeld = 0;
    int residentExperts = 0;
    int contextLimit = 32768;
    int chunk = 256;
    int crestRows = 1024;
    int loanCrest = 0;
    int loanFrom = 4096;
    int loanRows = 32768;
    int loanUsed = 0;
    bool lent = false;
    uint64_t loanBase = 0;
    uint64_t loanSize = 0;
    std::vector<std::array<uint64_t, 2>> loanPages;
    std::vector<std::pair<uint64_t*, uint64_t>> loanSaved;
    std::vector<std::array<int, 3>> loanSlots;
    uint8_t* loanStore = nullptr;
    uint64_t loanStoreSize = 0;
    int loanKept[7] = {};
    int budget = 2051;
    int scoreStride = 0;
    bool usePle = true;
    bool exact = false;
    bool fine = true;
    bool everyRow = false;
    int verifyLimit = 16;
    uint64_t snapshots[4] = {};
    void* hostSnapshots[4] = {};
    std::vector<uint32_t> warmClock;
    std::set<std::pair<uint32_t, int>> warmOrder;
    uint64_t warmBytes = 0;
    uint64_t warmBudget = 0;
    uint64_t warmChecked = 0;
    uint64_t warmKeep = 0;
    bool holdRam = true;
    bool ramWhole = false;
    uint64_t wholeBytes = 0;
    bool vision = false;
    int visionGpu = -1;
    bool seeing = false;
    std::string visionProblem;
    std::string visionCard;
    uint32_t warmTick = 0;
    int warmExperts = 0;
    int warmPerLayer = 0;
    int warmRooms = 0;
    bool spareWarm = true;
    uint64_t relayRoutes = 0;
    uint64_t relayKept = 0;
    uint64_t relayCpu = 0;
    uint64_t relayWarm = 0;
    double serveWait = 0;
    double serveWork = 0;
    uint64_t warmHits = 0;
    uint64_t decodeRouted = 0;
    uint64_t decodeResident = 0;
    uint64_t warmPinnedBytes = 0;
    bool pinWarm = true;
    int pinShare = 50;
    double pinGb = 0;
    int warmShare = 2;
    bool tensorPlain = true;
    bool tilePlain = true;
    bool typedRows = true;
    bool fewMma = true;
    int fewMmaFrom = 6;
    bool legacy = false;
    bool residual = true;
    bool batchFine = false;
    bool legacyWide = true;
    bool flushQueue = true;
    bool readAheadCold = true;
    bool directRows = true;
    bool pairRows = true;
    bool groupedRows = true;
    bool wideTiles = false;
    bool deepTiles = true;
    bool deepWide = true;
    bool exactMix = false;
    bool tallTiles = true;
    int wideLeast = 64;
    bool streamPrefill = true;
    bool directStream = true;
    std::string cpuPlan = "all";
    GROUP_AFFINITY hostPlace = {};
    DWORD pinnedThread = 0;
    int shortRead = 64;
    int superLimit = 8192;
    int stageSlots = 32;
    int batchRows = 8192;
    ExpertPack expertPack;
    bool packed = false;
    uint64_t stageStride = 0;
    int stageSides = 6;
    uint64_t stage[8] = {};
    uint8_t* ring[8] = {};
    void* copyStream = nullptr;
    void* moved[8] = {};
    void* stageFree[8] = {};
    struct Piece {
        int layer;
        int first;
        int count;
    };
    std::vector<std::vector<int>> streamPlan;
    std::vector<Piece> streamPieces;
    size_t streamIssued = 0;
    size_t streamDone = 0;
    uint64_t moeOut = 0;
    uint64_t splitScratch = 0;
    float* hostWeights = nullptr;
    uint64_t mappedWeights = 0;
    uint64_t streamedCount = 0;
    double streamMs = 0;
    double readMs = 0;
    double copyMs = 0;
    std::string phase = "other";
    uint64_t residualRows = 0;
    int residualCount = 0;
    int residualStart = 0;
    int residualStride = hcWidth;
    std::thread warmer;
    std::atomic<bool> warmStop{false};
    int recentSlots = 32;
    int recentFloor = 16;
    int recentPerLayer = 0;
    int fixedExperts = 0;
    int liftCrew = 4;
    int liftLimit = 512;
    int diskLifts = 4;
    int diskLimit = 8;
    int diskPending = 0;
    int liftPending = 0;
    uint32_t recentTick = 1;
    uint64_t liftCount = 0;
    uint64_t liftRoom = 0;
    std::vector<int> recentHolds;
    std::vector<uint32_t> recentUsed;
    std::vector<char> recentBusy;
    std::vector<std::pair<int, int>> wanted;
    std::vector<Lift> lifting, lifted;
    std::vector<Lifter> lifters;
    std::mutex liftLock;
    std::condition_variable liftWake;
    bool liftStop = false;
    int* slotStage = nullptr;
    int* warmStage = nullptr;
    int recentSpare = 4;
    std::vector<std::vector<int>> ramFree;
    std::vector<std::vector<int>> spareFree;
    std::vector<int> ramComing;
    std::atomic<uint32_t> touched{0};
    std::atomic<bool> warmDone{false};
    bool locking = true;
    size_t savedTokens[4] = {};
    bool useGraphs = true;
    bool splitAttention = true;
    bool groupAttention = true;
    bool mmaAttention = true;
    bool intAttention = true;
    bool voltaAttention = false;
    bool mixUp = true;
    bool tileScores = true;
    bool useBlas = true;
    bool halfMix = true;
    bool groupScan = true;
    bool fastFeed = true;
    bool fusedNorm = true;
    bool halfGates = true;
    bool joinInject = true;
    bool wideHead = false;
    void* blas = nullptr;
    uint64_t blasWork = 0;
    uint64_t halfWeights = 0;
    uint64_t denseHostBytes = 0;
    int denseCpuRows = 16;
    uint64_t denseStage = 0;
    double denseWork = 0.0, denseWait = 0.0;
    float* hostDenseOut = nullptr;
    uint64_t mappedDenseOut = 0;
    uint64_t halfRoom = 0;
    uint64_t wideRows = 0;
    uint64_t wideRoom = 0;
    std::map<const GpuMatrix*, uint64_t> halved;
    bool fusedMix = true;
    bool xPacked = false;
    bool stagedSets = true;
    bool teams = true;
    bool growKv = true;
    int kvStart = 16384;
    int kvRoom = 0;
    uint64_t pageGrain = 0;
    struct Stretch {
        uint64_t base = 0;
        uint64_t size = 0;
        std::vector<std::array<uint64_t, 2>> parts;
        std::vector<char> mapped;
    };
    std::vector<Stretch> stretches;
    std::vector<std::array<uint64_t, 3>> stretchPages;
    std::vector<unsigned long long> freePages;
    struct PoolSpan {
        uint64_t base = 0;
        uint64_t size = 0;
        uint64_t solid = 0;
        unsigned long long whole = 0;
        std::vector<unsigned long long> pages;
    };
    std::vector<PoolSpan> poolSpans;
    int givenBack = 0;
    bool exchangeHold = false;
    std::vector<uint8_t*> heldCopies;
    uint64_t growMargin = 256ULL << 20;
    bool liftHold = false;
    bool relayAlways = false;
    int graphRows = 5;
    int fewRows = 9;
    std::string graphProblem;
    std::string launchProblem;
    bool profile = false;
    bool profileShapes = false;
    std::map<std::string, double> kernelMs;
    std::map<std::string, int> kernelCalls;
    std::vector<std::pair<std::string, std::pair<void*, void*>>> marks;
    size_t markUsed = 0;
    bool fixed = false;
    bool relay = false;
    bool hostInt8 = true;
    int span = 0;
    int groups = 0;
    int residentRows = 0;
    int cpuRows = 0;
    int cpuBase = 0;
    int listSpan = 0;
    int widest = 1;
    bool ready = false;
    bool failed = false;
    std::string problem;

    ~GpuModel() {
        seer.close();
        vramBudget.close();
        processVram.close();
        if(ready && !hy4) saveHeat();
        if(ready && hy4) hySaveHeat();
        if(hy4) hyStopUploads();
        if(helper.context) closeHelper();
        warmStop = true;
        if(warmer.joinable()) warmer.join();
        {
            std::lock_guard<std::mutex> guard(liftLock);
            liftStop = true;
        }
        liftWake.notify_all();
        for(Lifter& own : lifters){
            if(own.thread.joinable()) own.thread.join();
        }
        if(cudaLink().ready){
            cudaLink().setCurrent(cudaLink().context);
            cudaLink().sync();
            if(blas) blasLink().destroy(blas);
            if(hyBuffers.blas) blasLink().destroy(hyBuffers.blas);
            blas = nullptr;
            for(uint64_t p : allocations) cudaLink().release(p);
            for(PoolSpan& p : poolSpans) dropPool(p);
            poolSpans.clear();
            for(auto& page : stretchPages){
                cudaLink().unmap(page[0], (size_t)page[1]);
                cudaLink().forget(page[2]);
            }
            for(Stretch& s : stretches) cudaLink().unreserve(s.base, (size_t)s.size);
            for(unsigned long long handle : freePages) cudaLink().forget(handle);
            freePages.clear();
            stretchPages.clear();
            stretches.clear();
            heldCopies.clear();
            kvRoom = 0;
            for(void* p : pinned) cudaLink().hostFree(p);
            for(PackedInput* packed : {&packs[0], &packs[1], &fines[0], &fines[1]}){
                for(uint64_t p : {packed->values, packed->scales, packed->sums}){
                    if(p) cudaLink().release(p);
                }
            }
            CudaQueue& queue = cudaQueue();
            for(auto& g : tokenGraphs){
                if(g.second) queue.execDestroy(g.second);
            }
            for(auto& m : marks){
                queue.eventDestroy(m.second.first);
                queue.eventDestroy(m.second.second);
            }
            if(routed) queue.eventDestroy(routed);
            for(int i = 0; i < 8; i++){
                if(moved[i]) queue.eventDestroy(moved[i]);
                if(stageFree[i]) queue.eventDestroy(stageFree[i]);
            }
            if(copyStream) queue.streamDestroy(copyStream);
            for(Lifter& own : lifters){
                if(own.event) queue.eventDestroy(own.event);
                if(own.stream) queue.streamDestroy(own.stream);
            }
            if(stream) queue.streamDestroy(stream);
        }
        for(MappedFile& m : maps){
            if(m.base) UnmapViewOfFile(m.base);
            if(m.mapping) CloseHandle(m.mapping);
            if(m.file != INVALID_HANDLE_VALUE) CloseHandle(m.file);
        }
    }

    int waitStream() {
        CudaQueue& queue = cudaQueue();
        return queue.streamSync
            ? queue.streamSync(stream)
            : cudaLink().sync();
    }

    double since(std::chrono::steady_clock::time_point from) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count();
    }

    template<typename... T>
    void go(const char* name, unsigned int gridX, unsigned int gridY, unsigned int blockX, unsigned int blockY, T... values) {
        void* args[] = {(void*)&values...};
        start(name, gridX, gridY, blockX, blockY, 0, args);
    }

    template<typename... T>
    void goShared(unsigned int shared, const char* name, unsigned int gridX, unsigned int gridY, unsigned int blockX, unsigned int blockY, T... values) {
        void* args[] = {(void*)&values...};
        start(name, gridX, gridY, blockX, blockY, shared, args);
    }

    void start(const char* name, unsigned int gridX, unsigned int gridY, unsigned int blockX, unsigned int blockY, unsigned int shared, void** args) {
        void* fn = kernelFunction(module, name);
        // mas de 48 KB de memoria compartida hay que pedirlo una vez por kernel
        if(fn && shared > 48 * 1024 && !widerShared.count(fn)){
            if(cudaLink().funcAttribute) cudaLink().funcAttribute(fn, 8, (int)shared);
            widerShared.insert(fn);
        }
        if(profile){
            if(markUsed == marks.size()){
                std::pair<void*, void*> pair = {nullptr, nullptr};
                cudaQueue().eventCreate(&pair.first, 0);
                cudaQueue().eventCreate(&pair.second, 0);
                marks.push_back({"", pair});
            }
            marks[markUsed].first = profileShapes
                ? std::string(name) + " " + std::to_string(gridX) + "x" + std::to_string(gridY)
                : phase + " " + name;
            cudaQueue().eventRecord(marks[markUsed].second.first, stream);
        }
        int launched = fn ? cudaLink().launch(fn, gridX, gridY, 1, blockX, blockY, 1, shared, stream, args, nullptr) : -1;
        if(launched != 0 && !failed) launchProblem = std::string(name) + " (" + std::to_string(launched) + ", grid " + std::to_string(gridX) + "x" + std::to_string(gridY) + ", shared " + std::to_string(shared) + ")";
        if(launched != 0) failed = true;
        if(profile) cudaQueue().eventRecord(marks[markUsed++].second.second, stream);
    }

    bool fetchAll(const std::vector<std::array<uint64_t, 4>>& wanted, bool toHost = false, void* context = nullptr) {
        const uint64_t piece = 16ULL << 20;
        std::vector<std::array<uint64_t, 4>> list;
        for(const auto& w : wanted){
            for(uint64_t done = 0; done < w[2]; done += piece) list.push_back({w[0], w[1] + done, std::min(piece, w[2] - done), w[3] + done});
        }
        std::sort(list.begin(), list.end(), [](const std::array<uint64_t, 4>& a, const std::array<uint64_t, 4>& b){
            return a[0] != b[0] ? a[0] < b[0] : a[1] < b[1];
        });
        uint64_t sum = 0;
        for(const auto& f : list) sum += f[2];
        loadStatus.done = 0;
        loadStatus.total = sum;

        while(staging.size() < 8){
            void* room = nullptr;
            if(cudaLink().hostAlloc(&room, (size_t)(piece + 8192), 0) != 0){
                if(problem.empty()) problem = "cannot pin the staging buffers for reading weights";
                return false;
            }
            staging.push_back(room);
            pinned.push_back(room);
        }

        std::atomic<size_t> next{0};
        std::atomic<bool> broken{false};
        std::vector<std::thread> crew;
        for(int w = 0; w < 8; w++){
            crew.emplace_back([&, w]{
                cudaLink().setCurrent(context ? context : cudaLink().context);
                uint8_t* buffer = (uint8_t*)staging[w];
                std::vector<HANDLE> handles(file.paths.size(), INVALID_HANDLE_VALUE);
                for(size_t i = next++; i < list.size() && !broken; i = next++){
                    if(stopLoading){
                        broken = true;
                        continue;
                    }
                    const auto& f = list[i];
                    HANDLE& h = handles[f[0]];
                    if(h == INVALID_HANDLE_VALUE) h = CreateFileA(file.paths[f[0]].c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
                    uint64_t start = f[1] & ~4095ULL;
                    uint64_t skip = f[1] - start;
                    uint64_t length = (skip + f[2] + 4095) & ~4095ULL;
                    OVERLAPPED at = {};
                    at.Offset = (DWORD)start;
                    at.OffsetHigh = (DWORD)(start >> 32);
                    DWORD got = 0;
                    if(h == INVALID_HANDLE_VALUE || !ReadFile(h, buffer, (DWORD)length, &got, &at) || got < skip + f[2]){
                        broken = true;
                        continue;
                    }
                    if(toHost) memcpy((void*)f[3], buffer + skip, (size_t)f[2]);
                    else if(cudaLink().copyUp(f[3], buffer + skip, (size_t)f[2]) != 0) broken = true;
                    loadStatus.done += f[2];
                }
                for(HANDLE h : handles){
                    if(h != INVALID_HANDLE_VALUE) CloseHandle(h);
                }
            });
        }
        for(std::thread& t : crew) t.join();
        if(broken && problem.empty()) problem = "reading weights into video memory failed";
        return !broken;
    }

    int openHelper(const std::string& source, const std::vector<uint8_t>& blob, uint64_t perExpert, int most) {
        CudaLink& cu = cudaLink();
        CudaQueue& queue = cudaQueue();
        if(!useHelper || most <= 0 || perExpert == 0 || cu.count < 2 || !cu.deviceGet || !cu.retain || !cu.attribute || !queue.hostDevice) return 0;
        for(int d = 0; d < cu.count && helper.device < 0; d++){
            int dev = -1, major = 0, minor = 0;
            if(helperOrdinal >= 0 && d != helperOrdinal) continue;
            if(cu.deviceGet(&dev, d) != 0 || dev == cu.device) continue;
            cu.attribute(&major, 75, dev);
            cu.attribute(&minor, 76, dev);
            if(major * 10 + minor < 61) continue;
            helper.device = dev;
            helper.major = major;
            helper.minor = minor;
        }
        if(helper.device < 0) return 0;
        helper.pascalPair = helper.major == 6 && helper.minor == 1 && cu.major * 10 + cu.minor >= 75;
        if(cu.retain(&helper.context, helper.device) != 0){
            helper.context = nullptr;
            helper.device = -1;
            return 0;
        }

        char bus[32] = {};
        NvmlLink& nv = nvmlLink();
        void* card = nullptr;
        if(cu.busId && cu.busId(bus, (int)sizeof(bus), helper.device) == 0 && nv.ready && nv.byBus && nv.byBus(bus, &card) == 0 && card){
            char name[96] = {};
            if(nv.getName(card, name, sizeof(name)) == 0) helper.name = name;
            helper.card = card;
        }
        if(helper.name.empty()) helper.name = "gpu " + std::to_string(helper.device);

        bool ok = buildModule(source, helper.module, helper.context, helper.major, helper.minor, helper.pascalPair);
        if(!ok) helper.problem = "the kernels did not build for " + helper.name + ": " + helper.module.log;
        auto take = [&](uint64_t bytes) -> uint64_t {
            uint64_t ptr = 0;
            if(cu.alloc(&ptr, (size_t)bytes) != 0) return 0;
            helper.allocations.push_back(ptr);
            return ptr;
        };
        auto mapped = [&](void** host, uint64_t* device, uint64_t bytes, unsigned int flags) -> bool {
            if(cu.hostAlloc(host, (size_t)bytes, flags) != 0) return false;
            helper.pinned.push_back(*host);
            return queue.hostDevice(device, *host, 0) == 0;
        };

        for(void*& mark : helper.marks) if(ok && helper.pascalPair && queue.eventCreate(&mark, 0) != 0) mark = nullptr;
        if(ok && queue.eventCreate(&helper.done, 2) != 0) helper.done = nullptr;
        uint64_t table = ok ? take(blob.size()) : 0;
        ok = ok && table && cu.copyUp(table, blob.data(), blob.size()) == 0;
        if(ok) helper.tables = {table, table + 2048, table + 6144, table + 14336, table + 15360, table + 17408, table + 17536, table + 17552};
        helper.capacity = expertsUsed * (std::max(relayLimit, 1) + 1);
        helper.where.assign(helper.capacity, 0);
        uint64_t scratch = 4ULL * helper.capacity * expertWidth;
        helper.gateOut = ok ? take(scratch) : 0;
        helper.upOut = ok ? take(scratch) : 0;
        helper.hiddenOut = ok ? take(scratch) : 0;
        helper.inRowsOnCard = ok && helper.pascalPair ? take(4ULL * helper.capacity * hidden) : 0;
        helper.slotsOnCard = ok && helper.pascalPair ? take(4ULL * helper.capacity) : 0;
        ok = ok && helper.gateOut && helper.upOut && helper.hiddenOut && (!helper.pascalPair || (helper.inRowsOnCard && helper.slotsOnCard));
        ok = ok && mapped((void**)&helper.inRows, &helper.inMapped, 4ULL * helper.capacity * hidden, 6)
            && mapped((void**)&helper.slotList, &helper.slotMapped, 4ULL * helper.capacity, 6)
            && mapped((void**)&helper.outRows, &helper.outMapped, 4ULL * helper.capacity * hidden, 2);
        if(!ok && helper.problem.empty()) helper.problem = "could not set up " + helper.name;

        size_t freeBytes = 0, totalBytes = 0;
        cu.info(&freeBytes, &totalBytes);
        unsigned long long physical[3] = {0, 0, 0};
        if(card && nv.getMemory(card, physical) == 0 && physical[0] > 0) freeBytes = (size_t)physical[1];
        uint64_t keep = 512ULL << 20;
        uint64_t room = freeBytes > keep ? freeBytes - keep : 0;
        if(helperCap > 0) room = std::min<uint64_t>(room, helperCap > keep ? helperCap - keep : 0);
        int perLayer = ok && room > 0
            ? (int)std::min<uint64_t>((uint64_t)most, room / perExpert)
            : 0;
        while(perLayer > 0){
            size_t mark = helper.allocations.size();
            bool fits = true;
            for(int l = firstLayer; l < endLayer && fits; l++){
                GpuLayer& L = layers[l];
                const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
                for(int i = 0; i < 3 && fits; i++){
                    uint64_t ptr = 0;
                    fits = cu.alloc(&ptr, (size_t)((uint64_t)perLayer * parts[i]->rows * parts[i]->stride + 256)) == 0;
                    if(fits) helper.allocations.push_back(ptr);
                    L.helpBase[i] = ptr;
                }
            }
            if(fits) break;
            while(helper.allocations.size() > mark){
                cu.release(helper.allocations.back());
                helper.allocations.pop_back();
            }
            perLayer -= std::max(1, perLayer / 16);
        }
        cu.setCurrent(cu.context);
        if(perLayer <= 0){
            if(ok) helper.problem = helper.name + " has no room left for experts";
            closeHelper();
            return 0;
        }
        helper.perLayer = perLayer;
        helper.bytes = (uint64_t)perLayer * perExpert;
        helper.ready = true;
        return perLayer;
    }

    void closeHelper() {
        CudaLink& cu = cudaLink();
        if(helper.context && cu.setCurrent(helper.context) == 0){
            cu.sync();
            for(uint64_t p : helper.allocations) cu.release(p);
            for(void* p : helper.pinned) cu.hostFree(p);
            for(PackedInput* packed : {&helper.packedIn, &helper.packedHidden}){
                if(packed->values) cu.release(packed->values);
                if(packed->scales) cu.release(packed->scales);
                if(packed->sums) cu.release(packed->sums);
            }
            for(void*& mark : helper.marks){
                if(mark) cudaQueue().eventDestroy(mark);
                mark = nullptr;
            }
            if(helper.done) cudaQueue().eventDestroy(helper.done);
            helper.done = nullptr;
            if(cu.drop) cu.drop(helper.device);
        }
        if(cu.ready) cu.setCurrent(cu.context);
        helper.allocations.clear();
        helper.pinned.clear();
        helper.packedIn = PackedInput();
        helper.packedHidden = PackedInput();
        helper.pascalPair = false;
        helper.module = KernelModule();
        helper.context = nullptr;
        helper.card = nullptr;
        helper.device = -1;
        helper.ready = false;
        helper.perLayer = 0;
        helper.bytes = 0;
        helper.inRows = nullptr;
        helper.slotList = nullptr;
        helper.outRows = nullptr;
        for(GpuLayer& L : layers){
            L.help.clear();
            for(int i = 0; i < 3; i++) L.helpBase[i] = 0;
        }
    }

    bool helperLaunch(GpuLayer& L, int rows) {
        CudaLink& cu = cudaLink();
        if(cu.setCurrent(helper.context) != 0) return false;
        uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride;
        uint64_t upSpan = (uint64_t)L.expertUp.rows * L.expertUp.stride;
        uint64_t downSpan = (uint64_t)L.expertDown.rows * L.expertDown.stride;
        int count = rows * expertWidth;
        void* fn = kernelFunction(helper.module, "swiglu");
        void* args[] = {&helper.gateOut, &helper.upOut, &helper.hiddenOut, &count};
        if(helperDelay > 0){
            unsigned long long nanos = (unsigned long long)helperDelay * 1000000ull;
            void* pause[] = {&nanos};
            launchKernel(kernelFunction(helper.module, "spinFor"), 1, 1, 1, 1, 0, pause);
        }
        if(!helper.pascalPair){
            bool ok = launchExperts(helper.module, (int)L.expertGate.type, L.helpBase[0], L.expertGate.stride, gateSpan, helper.slotMapped, rows, helper.inMapped, (uint64_t)hidden, helper.gateOut, expertWidth, hidden, helper.tables)
                && launchExperts(helper.module, (int)L.expertUp.type, L.helpBase[1], L.expertUp.stride, upSpan, helper.slotMapped, rows, helper.inMapped, (uint64_t)hidden, helper.upOut, expertWidth, hidden, helper.tables)
                && launchKernel(fn, (unsigned int)((count + 255) / 256), 1, 256, 1, 0, args)
                && launchExperts(helper.module, (int)L.expertDown.type, L.helpBase[2], L.expertDown.stride, downSpan, helper.slotMapped, rows, helper.hiddenOut, (uint64_t)expertWidth, helper.outMapped, hidden, expertWidth, helper.tables);
            ok = ok && (!helper.done || cudaQueue().eventRecord(helper.done, nullptr) == 0);
            if(!ok) cu.sync();
            cu.setCurrent(cu.context);
            return ok;
        }
        bool timed = (profile || timeHelper) && helper.marks[4];
        int gateType = (int)L.expertGate.type, upType = (int)L.expertUp.type, downType = (int)L.expertDown.type;
        uint64_t gateRow = L.expertGate.stride, upRow = L.expertUp.stride, downRow = L.expertDown.stride;
        int width = expertWidth, wide = hidden;
        int upStage = (int)((std::max(gateRow, upRow) + 31) / 16 * 16), downStage = (int)((downRow + 31) / 16 * 16);
        unsigned int upShared = (unsigned int)(3232 + 8 * 2 * upStage);
        unsigned int downShared = (unsigned int)(3232 + 8 * downStage);
        void* upArgs[] = {&gateType, &upType, &L.helpBase[0], &L.helpBase[1], &gateRow, &upRow, &gateSpan, &upSpan, &helper.slotsOnCard, &helper.inRowsOnCard, &helper.packedIn.values, &helper.packedIn.scales, &helper.packedIn.sums, &helper.hiddenOut, &width, &wide, &upStage, &helper.tables};
        void* downArgs[] = {&downType, &L.helpBase[2], &downRow, &downSpan, &helper.slotsOnCard, &helper.hiddenOut, &helper.packedHidden.values, &helper.packedHidden.scales, &helper.packedHidden.sums, &helper.outMapped, &wide, &width, &downStage, &helper.tables};
        bool ok = cudaQueue().copyUp(helper.inRowsOnCard, helper.inRows, (size_t)rows * hidden * 4, nullptr) == 0
            && cudaQueue().copyUp(helper.slotsOnCard, helper.slotList, (size_t)rows * 4, nullptr) == 0
            && (!timed || cudaQueue().eventRecord(helper.marks[0], nullptr) == 0)
            && packInput(helper.module, helper.inRowsOnCard, rows * hidden, helper.packedIn)
            && launchKernel(kernelFunction(helper.module, "helperGateUp"), (unsigned int)((expertWidth + 7) / 8), (unsigned int)rows, 32, 8, upShared, upArgs)
            && (!timed || cudaQueue().eventRecord(helper.marks[1], nullptr) == 0)
            && (!timed || cudaQueue().eventRecord(helper.marks[2], nullptr) == 0)
            && (!timed || cudaQueue().eventRecord(helper.marks[3], nullptr) == 0)
            && packInput(helper.module, helper.hiddenOut, count, helper.packedHidden)
            && launchKernel(kernelFunction(helper.module, "helperDown"), (unsigned int)((hidden + 7) / 8), (unsigned int)rows, 32, 8, downShared, downArgs)
            && (!timed || cudaQueue().eventRecord(helper.marks[4], nullptr) == 0)
            && (!helper.done || cudaQueue().eventRecord(helper.done, nullptr) == 0);
        if(!ok) cu.sync();
        cu.setCurrent(cu.context);
        return ok;
    }

    bool load(const std::string& path, int threads) {
        std::string error;
        ModelFile opened;
        if(!openModel(path, opened, error)){
            problem = error;
            return false;
        }
        return loadFile(opened, path, threads);
    }

    bool tail() const {
        return endLayer == (int)layers.size();
    }

    std::string tiled(uint32_t type, bool wide = false, uint64_t fineIn = 1) const {
        return legacy
            ? (wide && legacyWide ? "dotRowsW" : "dotRows") + std::to_string(type) + (fineIn ? "" : "c")
            : wide && wideTiles
                ? "mmaWide"
                : deepTiles
                    ? std::string(type == 14 || type == 17 || type == 22 || type == 29 ? "mmaDeepSplit" : "mmaDeep") + (fineIn ? "" : "Lean") + (wide && deepWide ? (tallTiles && !fineIn ? "X" : "W") : "")
                    : "mmaRows";
    }

    int rowTile(uint32_t type, bool wide = false, uint64_t fineIn = 1) const {
        return tiled(type, wide, fineIn).back() == 'X'
            ? 128
            : 64;
    }

    int tileWidth(bool wide = false) const {
        return legacy
            ? (wide && legacyWide ? 128 : 64)
            : wide && (wideTiles || deepTiles && deepWide)
                ? 128
                : 64;
    }

    bool loadGemma(const std::string& path, int threads);
    bool loadHy4(const std::string& path, int threads);
    bool moveHy4(int slot, bool keep);
    void hySaveHeat();
    void hyFillRam();
    HyBuffers hyRowsView(const HyBuffers& base, int rows, int l);
    void hyStartUploads();
    void hyCollectUploads(bool wait);
    void hyStopUploads();
    void hyExpertRanges(int key, WIN32_MEMORY_RANGE_ENTRY* out);
    void hyHead(int n);
    bool hyRelay(int n);
    void hyFork();
    void hyBack();
    void hyJoin();
    void hyRelayExperts(int l, int n);
    void hySwapIn(int l, int n, const int* ids);
    void hyHostMultiply(const std::vector<HostProduct>& list, bool residual, int rows);
    void hyFillWarm();
    void hyOpenHelper(const std::string& source, const std::vector<uint8_t>& blob);
    bool hyHelperLaunch(HyLayer& L, int n, const std::vector<int>& slots, const std::vector<int>& pairsList);
    bool hyLoadDraft(const std::string& modelPath);
    std::vector<int> hyPropose(int token, int most);
    void hyPack(uint64_t input, int count, int slot);
    void hyProjectSame(const std::vector<std::pair<const GpuMatrix*, uint64_t>>& list, uint64_t input, int n, bool ready = false);
    void hyProject(const GpuMatrix& m, uint64_t input, uint64_t output, int n);
    void hyHcIn(uint64_t streams, uint64_t fn, uint64_t scale, uint64_t base, uint64_t norm, uint64_t out, uint64_t post, int rows, int n, bool pack = false);
    void hyHcOut(uint64_t x, uint64_t norm, int n);
    void hyNormRows(uint64_t in, int inStride, uint64_t out, int outStride, uint64_t weight, int weightStride, int rows, int groups, int cols, float eps);
    void hyEngramHost(const int* positions, int n);
    void hyExpertsHost(HyLayer& L, int n, const std::vector<char>& onCpu);
    void hyExperts(int l, int n);
    void hyUpload(int l, int e, int slot);
    void hyBatch(HyLayer& L, uint64_t base, const int* slots, const int* pairs, uint64_t slotsOn, uint64_t pairsOn, int count);
    void hyLayer(int l, int n, int part = 0);
    void hyAttention(int l, int n);
    bool hyCopyExpert(int l, int e, uint64_t at);
    void hyExpertsWide(int l, int n, uint64_t stage);
    void hyExpertsGrouped(int l, int n, uint64_t stage, const std::vector<std::vector<int>>& byExpert);
    bool hyWide(const int* ids, const int* positions, int n, bool last);
    std::vector<std::pair<uint64_t*, uint64_t>> hySubBuffers(int rows);
    bool hyChunk(const int* ids, const int* positions, int n, bool last);
    std::vector<float> forwardHy4(const std::vector<int>& ids, const std::vector<int>& positions);
    void gemmaPack(uint64_t input, int count, PackedInput& packed);
    void gemmaExpertRows(const GpuMatrix& m, uint64_t expertBytes, int perToken, uint64_t input, PackedInput& packed, uint64_t output, int slots);
    int gemmaSplits(const GemmaLayer& L, int n, int top);
    void gemmaRun(int l, int n, int top);
    bool gemmaRounded = false;
    bool gemmaFew = true;
    bool gemmaFlash = true;
    bool padThree = true;
    bool gemmaGroups = true;
    bool gemmaWide = true;
    int gemmaBlock = 1024;
    bool gemmaChunk(const int* ids, const int* positions, int n, bool last);
    std::vector<float> forwardGemma(const std::vector<int>& ids, const std::vector<int>& positions);
    bool moveGemma(int slot, bool keep);
    GemmaDraft gemmaDraft;
    bool loadGemmaDraft(const std::string& path, const ModelFile& f);
    void gemmaDraftStep(int top);
    std::vector<int> gemmaPropose(int token, int most);
    bool gemmaKeep(int count);

    bool loadFile(const ModelFile& opened, const std::string& path, int threads) {
        keepFullSpeed();
        file = opened;
        if(file.arch == "gemma4") return loadGemma(path, threads);
        if(file.arch == "hy_v4_flash") return loadHy4(path, threads);
        dense = file.arch == "qwen35";
        if(dense) residual = false;
        if(endLayer <= 0 || endLayer > (int)file.layers) endLayer = (int)file.layers;
        if(firstLayer < 0 || firstLayer >= endLayer) firstLayer = 0;
        bool last = endLayer == (int)file.layers;
        bool lead = firstLayer <= 1;
        expertCount = dense
            ? 32
            : file.experts > 0
                ? (int)file.experts
                : 512;
        if(expertCount % 32 != 0 || expertCount > 1024){
            problem = "the model has " + std::to_string(expertCount) + " experts per layer, the engine needs a multiple of 32 up to 1024";
            return false;
        }
        if(!last && endLayer < 2){
            problem = "a pool node needs at least the first two layers";
            return false;
        }
        if(!readShape(file, problem)) return false;
        usePle = usePle && file.arch == "qwen4exp";
        if(dense){
            chunk = 1024;
            streamPrefill = false;
            pinWarm = false;
            foresee = false;
        }
        if(!useCuda()){
            problem = "no cuda device";
            return false;
        }
        CudaQueue& queue = cudaQueue();
        if(!queue.ready || !cudaLink().hostAlloc || !cudaLink().memset){
            problem = "cuda driver entry points missing";
            return false;
        }
        size_t startFree = 0, startTotal = 0;
        cudaLink().info(&startFree, &startTotal);
        freeAtStart = startFree;
        cardAtStart = 0;
        {
            NvmlLink& nv = nvmlLink();
            unsigned long long physical[3] = {0, 0, 0};
            if(nv.ready && nv.getMemory(nv.device, physical) == 0 && physical[0] > 0) cardAtStart = physical[1];
        }
        char luid[8] = {};
        unsigned int nodes = 0;
        if(cudaLink().luid && cudaLink().luid(luid, &nodes, cudaLink().device) == 0) vramBudget.open(luid);
        {
            // el presupuesto que windows le da a este programa en esta tarjeta, para leerlo en el log
            uint64_t budget = 0, usage = 0;
            if(vramBudget.read(budget, usage) && budget > 0){
                startBudget = budget;
                startUsage = usage;
                std::cout << "windows video memory budget for this card " << (budget >> 20) << " MB, used " << (usage >> 20) << " MB, free on the card " << (startFree >> 20) << " MB" << std::endl;
            }
        }
        legacy = legacy || cudaLink().major < 8;
        // la v100 (sm_70 y sm_72) lee el prompt con sus propios tensor cores en volta.h
        // SNEO_VOLTA_ATTENTION=1 lo fuerza en otra nvidia para probarlo y =0 lo apaga
        char voltaText[8] = {};
        bool voltaAsked = GetEnvironmentVariableA("SNEO_VOLTA_ATTENTION", voltaText, sizeof(voltaText)) > 0;
        voltaAttention = !amdCards && cudaLink().major >= 7 && (voltaAsked
            ? voltaText[0] == '1'
            : cudaLink().major == 7 && cudaLink().minor < 5);
        if(voltaAttention) std::cout << "prompt attention on the volta tensor cores" << std::endl;
#ifndef _WIN32
        // wsl deja fijar poco mas de 1 GB de ram para la tarjeta: los anillos que suben expertos al leer el prompt van con dos lados
        if(access("/dev/dxg", F_OK) == 0){
            stageSlots = std::min(stageSlots, 16);
            stageSides = 2;
        }
#endif
        if(legacy){
            warmShare = std::min(warmShare, 2);
            chunk = 512;
            relayLimit = std::min(relayLimit, 5);
            verifyLimit = relayLimit;
            draftBatch = relayLimit;
            superLimit = 2048;
            stageSlots = 16;
            stageSides = 2;
            batchRows = 4096;
            tensorPlain = false;
            wideTiles = false;
            batchFine = false;
            directRows = false;
        }
        verifyLimit = std::max(verifyLimit, relayLimit);
        std::string sizes = "#define HIDDEN " + std::to_string(hidden)
            + "\n#define HC " + std::to_string(hcCount)
            + "\n#define HCW " + std::to_string(hcWidth)
            + "\n#define HCR " + std::to_string(std::max(hcRank, 32))
            + "\n#define KUSED " + std::to_string(expertsUsed)
            + "\n#define EXPERTS " + std::to_string(expertCount % 32 == 0 ? expertCount : 32)
            + "\n#define CONVW " + std::to_string(convWidth)
            + "\n#define INNER " + std::to_string(innerWidth)
            + "\n#define VHEADS " + std::to_string(valueHeads)
            + "\n#define KHEADS " + std::to_string(keyHeads)
            + "\n#define QHEADS " + std::to_string(queryHeads)
            + "\n#define KVHEADS " + std::to_string(kvHeads)
            + "\n#define KVW " + std::to_string(kvHeads * headWidth)
            + "\n#define IHEADS " + std::to_string(indexHeads)
            + "\n#define EXACT_MIX " + std::string(exactMix ? "1" : "0")
            + "\n#define SILU_NORM " + std::string(file.arch != "qwen4exp" ? "1" : "0") + "\n";
        loadSay("Building the GPU kernels");
        if(!buildModule(sizes + quantKernelSource + modelKernelSource + (voltaAttention ? voltaKernelSource : ""), module)){
            problem = "kernel build failed: " + module.log;
            return false;
        }

        for(const std::string& p : file.paths){
            MappedFile m;
            m.file = CreateFileA(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
            if(m.file != INVALID_HANDLE_VALUE) m.mapping = CreateFileMappingA(m.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if(m.mapping) m.base = (const uint8_t*)MapViewOfFile(m.mapping, FILE_MAP_READ, 0, 0, 0);
            maps.push_back(m);
            if(!m.base){
                problem = "cannot map " + p;
                return false;
            }
        }

        uint64_t grabbed = 0, hostBytes = 0;
        int grabs = 0;
        auto grab = [&](uint64_t bytes) -> uint64_t {
            uint64_t ptr = 0;
            int code = cudaLink().alloc(&ptr, (size_t)std::max<uint64_t>(bytes, 256));
            if(code != 0){
                if(problem.empty()){
                    problem = code == 2
                        ? "out of video memory allocating " + std::to_string(bytes >> 20) + " MB"
                        : "cuda error " + std::to_string(code) + " allocating " + std::to_string(bytes >> 20) + " MB";
                    // lo que habia en la tarjeta cuando se nego, para saber si la lleno esta carga u otra cosa
                    size_t freeNow = 0, totalNow = 0;
                    int asked = cudaLink().info(&freeNow, &totalNow);
                    NvmlLink& nv = nvmlLink();
                    unsigned long long physical[3] = {0, 0, 0};
                    if(nv.ready && nv.getMemory) nv.getMemory(nv.device, physical);
                    typedef int (*CuGetCurrent)(void**);
                    HMODULE lib = GetModuleHandleA("nvcuda.dll");
                    auto current = lib ? (CuGetCurrent)GetProcAddress(lib, "cuCtxGetCurrent") : nullptr;
                    void* now = nullptr;
                    if(current) current(&now);
                    uint64_t budget = 0, usage = 0, sharedBudget = 0, sharedUsage = 0;
                    vramBudget.read(budget, usage);
                    vramBudget.readShared(sharedBudget, sharedUsage);
                    std::cout << "the card refused " << bytes << " bytes (cuda error " << code << ") after " << grabs << " allocations of " << (grabbed >> 20) << " MB and " << (hostBytes >> 20) << " MB pinned in this load"
                        << ", cuda free " << (asked == 0 ? std::to_string(freeNow >> 20) : "error " + std::to_string(asked)) << " MB of " << (totalNow >> 20)
                        << ", card free " << (physical[1] >> 20) << " MB, windows budget " << (budget >> 20) << " MB used " << (usage >> 20) << " MB"
                        << ", shared budget " << (sharedBudget >> 20) << " MB used " << (sharedUsage >> 20) << " MB, "
                        << (now == cudaLink().context ? "main context" : now ? "another context is current" : "no context is current") << std::endl;
                }
                return 0;
            }
            grabbed += bytes;
            grabs++;
            allocations.push_back(ptr);
            return ptr;
        };

        auto host = [&](uint64_t bytes) -> void* {
            void* p = nullptr;
            if(cudaLink().hostAlloc(&p, (size_t)bytes, 0) != 0){
                if(problem.empty()) problem = "cannot pin " + std::to_string(bytes >> 20) + " MB of host memory";
                return nullptr;
            }
            hostBytes += bytes;
            pinned.push_back(p);
            return p;
        };

        std::vector<std::pair<const TensorInfo*, uint64_t*>> wanted;
        std::vector<std::pair<std::vector<uint16_t>, uint64_t*>> narrowed;

        auto matrix = [&](const std::string& name, GpuMatrix& out) {
            const TensorInfo* t = findTensor(file, name);
            if(!t || !knownType(t->type)){
                if(problem.empty()) problem = t
                    ? name + " is " + typeInfo(t->type).name + ", a tensor type the engine cannot read yet"
                    : "missing tensor " + name;
                return;
            }
            out.type = t->type;
            out.cols = (int)t->dims[0];
            out.rows = (int)(t->elements / t->dims[0]);
            out.stride = rowBytes(t->type, t->dims[0]);
            wanted.push_back({t, &out.data});
        };

        auto narrow = [&](const std::string& name, GpuMatrix& out) {
            const TensorInfo* t = findTensor(file, name);
            if(!t || t->type == 30 || t->type == 1 || !knownType(t->type)){
                matrix(name, out);
                return;
            }
            out.type = 30;
            out.cols = (int)t->dims[0];
            out.rows = (int)(t->elements / t->dims[0]);
            out.stride = (uint64_t)out.cols * 2;
            const uint8_t* from = maps[t->file].base + t->offset;
            uint64_t inRow = rowBytes(t->type, t->dims[0]);
            std::vector<float> row((size_t)out.cols);
            std::vector<uint16_t> half((size_t)t->elements);
            for(int r = 0; r < out.rows; r++){
                dequantRow(t->type, from + (uint64_t)r * inRow, row.data(), out.cols);
                for(int c = 0; c < out.cols; c++){
                    uint32_t bits;
                    memcpy(&bits, &row[c], 4);
                    bits += 0x7FFF + ((bits >> 16) & 1);
                    half[(size_t)r * out.cols + c] = (uint16_t)(bits >> 16);
                }
            }
            narrowed.push_back({std::move(half), &out.data});
        };

        auto floats = [&](const std::string& name, uint64_t& out) {
            const TensorInfo* t = findTensor(file, name);
            if(!t || t->type != 0){
                if(problem.empty()) problem = name + " is missing or not F32";
                return;
            }
            wanted.push_back({t, &out});
        };

        auto mapped = [&](const std::string& name, Matrix& out) -> const TensorInfo* {
            const TensorInfo* t = findTensor(file, name);
            if(!t || !knownType(t->type)){
                if(problem.empty()) problem = t
                    ? name + " is " + typeInfo(t->type).name + ", a tensor type the engine cannot read yet"
                    : "missing tensor " + name;
                return nullptr;
            }
            out.data = maps[t->file].base + t->offset;
            out.type = t->type;
            out.cols = (int)t->dims[0];
            out.rows = (int)(t->elements / t->dims[0]);
            out.stride = rowBytes(t->type, t->dims[0]);
            return t;
        };

        bool single = hcCount == 1;
        if(last){
            mapped("token_embd.weight", embed);
            if(usePle) mapped("per_layer_token_embd.weight", pleTable);
            matrix(findTensor(file, "output.weight") ? "output.weight" : "token_embd.weight", head);
            if(single) floats("output_norm.weight", outNorm);
            else {
                narrow("output_hc_down.weight", outDown);
                narrow("output_hc_up.weight", outUp);
                floats("output_hc_norm.weight", outNorm);
            }
        }
        Matrix conv;
        if(lead && usePle){
            narrow("blk.1.ple_key.weight", pleKey);
            narrow("blk.1.ple_value.weight", pleValue);
            floats("blk.1.ple_norm_key.weight", pleNormKey);
            floats("blk.1.ple_norm_query.weight", pleNormQuery);
            floats("blk.1.ple_norm_conv.weight", pleNormConv);
            mapped("blk.1.ple_conv1d.weight", conv);
        }

        uint32_t interval = metaWhole(file, file.arch + ".full_attention_interval", 4);
        layers.resize(file.layers);
        for(int l = 0; l < (int)layers.size(); l++){
            GpuLayer& L = layers[l];
            std::string p = "blk." + std::to_string(l) + ".";
            L.attention = (l + 1) % interval == 0;
            if(l < firstLayer || l >= endLayer) continue;

            const char* halves[2] = {"hc_attn_", "hc_ffn_"};
            for(int h = 0; h < 2 && !single; h++){
                floats(p + halves[h] + "norm.weight", L.hcNorm[h]);
                narrow(p + halves[h] + "down.weight", L.hcDown[h]);
                narrow(p + halves[h] + "up.weight", L.hcUp[h]);
                narrow(p + halves[h] + "inject.weight", L.hcInject[h]);
            }
            if(single){
                floats(p + "attn_norm.weight", L.hcNorm[0]);
                floats(p + "post_attention_norm.weight", L.hcNorm[1]);
            }

            if(L.attention){
                matrix(p + "attn_q.weight", L.query);
                matrix(p + "attn_k.weight", L.key);
                matrix(p + "attn_v.weight", L.value);
                matrix(p + "attn_output.weight", L.attnOut);
                floats(p + "attn_q_norm.weight", L.queryNorm);
                floats(p + "attn_k_norm.weight", L.keyNorm);
                if(indexHeads > 0){
                    narrow(p + "indexer.q_proj.weight", L.indexQuery);
                    narrow(p + "indexer.k_proj.weight", L.indexKey);
                    floats(p + "indexer.q_norm.weight", L.indexQueryNorm);
                    floats(p + "indexer.k_norm.weight", L.indexKeyNorm);
                }
            } else {
                matrix(p + "attn_qkv.weight", L.qkv);
                matrix(p + "attn_gate.weight", L.gate);
                narrow(p + "ssm_alpha.weight", L.alpha);
                narrow(p + "ssm_beta.weight", L.beta);
                matrix(p + "ssm_out.weight", L.ssmOut);
                floats(p + "ssm_conv1d.weight", L.convWeight);
                floats(p + "ssm_a", L.decay);
                floats(p + "ssm_dt.bias", L.dtBias);
                floats(p + "ssm_norm.weight", L.ssmNorm);
            }

            if(dense){
                matrix(p + "ffn_gate.weight", L.sharedGate);
                matrix(p + "ffn_up.weight", L.sharedUp);
                matrix(p + "ffn_down.weight", L.sharedDown);
                continue;
            }
            narrow(p + "ffn_gate_inp.weight", L.router);
            narrow(p + "ffn_gate_inp_shexp.weight", L.sharedScalar);
            matrix(p + "ffn_gate_shexp.weight", L.sharedGate);
            matrix(p + "ffn_up_shexp.weight", L.sharedUp);
            matrix(p + "ffn_down_shexp.weight", L.sharedDown);
            L.expertTensor[0] = mapped(p + "ffn_gate_exps.weight", L.expertGate);
            L.expertTensor[1] = mapped(p + "ffn_up_exps.weight", L.expertUp);
            L.expertTensor[2] = mapped(p + "ffn_down_exps.weight", L.expertDown);
            L.expertGate.rows /= expertCount;
            L.expertUp.rows /= expertCount;
            L.expertDown.rows /= expertCount;
        }

        if(!problem.empty()) return false;

        bool shapes = !layers.empty() && (layers.size() < 64 || (dense && layers.size() <= 128))
            && (!last || (embed.cols == hidden && head.cols == hidden && (!usePle || pleTable.cols == pleRowWidth)))
            && (!lead || !usePle || ((conv.type == 1 || conv.type == 0) && conv.cols * conv.rows == 4 * hcWidth));
        if(!shapes){
            problem = "unexpected tensor shapes for " + file.arch;
            return false;
        }

        std::vector<const GpuMatrix*> plainList;
        if(last && !single){
            plainList.push_back(&outDown);
            plainList.push_back(&outUp);
        }
        if(lead && usePle){
            plainList.push_back(&pleKey);
            plainList.push_back(&pleValue);
        }
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            for(int h = 0; h < 2 && !single; h++){
                plainList.push_back(&L.hcDown[h]);
                plainList.push_back(&L.hcUp[h]);
                plainList.push_back(&L.hcInject[h]);
            }
            if(!dense){
                plainList.push_back(&L.router);
                plainList.push_back(&L.sharedScalar);
            }
            if(L.attention && indexHeads > 0){
                plainList.push_back(&L.indexQuery);
                plainList.push_back(&L.indexKey);
            } else if(!L.attention){
                plainList.push_back(&L.alpha);
                plainList.push_back(&L.beta);
            }
            if(dense && (L.sharedGate.rows != expertWidth || L.sharedUp.rows != expertWidth || L.sharedDown.rows != hidden)){
                problem = "unexpected feed forward shapes";
                return false;
            }
            if(!dense && (L.expertGate.rows != expertWidth || L.expertDown.rows != hidden || L.expertGate.type != L.expertUp.type)){
                problem = "unexpected expert shapes";
                return false;
            }
        }
        for(const GpuMatrix* m : plainList){
            if((m->type != 30 && m->type != 1) || m->cols % 8 != 0){
                problem = "expected a BF16 or F16 matrix with 8 aligned rows";
                return false;
            }
        }

        auto multipliers = file.meta.find("qwen4exp.ple.layer_multipliers");
        auto vocab = file.meta.find("qwen4exp.ple.head_vocab_sizes");
        auto offsets = file.meta.find("qwen4exp.ple.head_offsets");
        bool pleMeta = multipliers != file.meta.end() && vocab != file.meta.end() && offsets != file.meta.end() && multipliers->second.wholes.size() == 3 && vocab->second.wholes.size() == pleHeads && offsets->second.wholes.size() == pleHeads;
        if(last && usePle && !pleMeta){
            problem = "ple metadata missing";
            return false;
        }
        if(pleMeta){
            for(int i = 0; i < 3; i++) pleMultipliers[i] = (uint64_t)multipliers->second.wholes[i];
            for(int i = 0; i < pleHeads; i++){
                pleVocab[i] = (uint64_t)vocab->second.wholes[i];
                pleOffset[i] = (uint64_t)offsets->second.wholes[i];
            }
        }

        budget = indexHeads > 0
            ? (int)metaWhole(file, file.arch + ".attention.indexer.top_k", 2048) + 3
            : 4;
        if(budget > 4096){
            problem = "indexer budget above 4096 cells";
            return false;
        }
        if(contextLimit < budget) contextLimit = budget;
        contextLimit = (contextLimit + 3) / 4 * 4;
        if(indexHeads == 0) budget = contextLimit;
        scoreStride = contextLimit / 4 + 4;
        if(chunk < 48 && !legacy) chunk = 48;

        std::vector<uint8_t> blob(33936);
        memcpy(blob.data(), grid2xxs, 2048);
        memcpy(blob.data() + 2048, grid2xs, 4096);
        memcpy(blob.data() + 6144, grid2s, 8192);
        memcpy(blob.data() + 14336, grid3xxs, 1024);
        memcpy(blob.data() + 15360, grid3s, 2048);
        memcpy(blob.data() + 17408, signTable, 128);
        memcpy(blob.data() + 17536, nonLinear4, 16);
        memcpy(blob.data() + 17552, grid1s, 16384);
        uint64_t table = grab(blob.size());
        if(!table || cudaLink().copyUp(table, blob.data(), blob.size()) != 0) return false;
        tables = {table, table + 2048, table + 6144, table + 14336, table + 15360, table + 17408, table + 17536, table + 17552};

        for(auto& w : wanted) denseBytes += (w.first->bytes + 255) / 256 * 256;

        // un denso que no entra en la vram deja la mlp de sus ultimas capas en ram fijada y la gpu la lee de ahi por pcie
        // igual que los expertos fijados de los moe: entra en tarjetas chicas y escribe mas lento
        std::vector<bool> onHost(wanted.size(), false);
        denseHostBytes = 0;
        if(dense){
            uint64_t attentionLayers = 0;
            for(int l = firstLayer; l < endLayer; l++) attentionLayers += layers[l].attention ? 1 : 0;
            uint64_t keep = (uint64_t)std::min(contextLimit, 32768) * attentionLayers * 2 * kvHeads * headWidth * 9 / 8 + (1536ULL << 20);
            size_t freeNow = 0, totalNow = 0;
            cudaLink().info(&freeNow, &totalNow);
            uint64_t room = vramCap > 0
                ? std::min<uint64_t>(freeNow, vramCap > grabbed ? vramCap - grabbed : 0)
                : freeNow;
            room = room > keep ? room - keep : 0;
            std::vector<size_t> order;
            for(size_t i = 0; i < wanted.size(); i++){
                const std::string& name = wanted[i].first->name;
                if(wanted[i].first->layer >= 0 && (name.find("ffn_gate.weight") != std::string::npos || name.find("ffn_up.weight") != std::string::npos || name.find("ffn_down.weight") != std::string::npos)) order.push_back(i);
            }
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b){ return wanted[a].first->layer > wanted[b].first->layer; });
            uint64_t layerFfn = 0;
            for(size_t k = 0; k < order.size() && wanted[order[k]].first->layer == wanted[order[0]].first->layer; k++) layerFfn += (wanted[order[k]].first->bytes + 255) / 256 * 256;
            uint64_t onCard = denseBytes;
            for(size_t k = 0; k < order.size() && onCard + (denseHostBytes ? layerFfn : 0) > room;){
                int layer = wanted[order[k]].first->layer;
                for(; k < order.size() && wanted[order[k]].first->layer == layer; k++){
                    uint64_t bytes = (wanted[order[k]].first->bytes + 255) / 256 * 256;
                    onHost[order[k]] = true;
                    onCard -= bytes;
                    denseHostBytes += bytes;
                }
            }
            if(denseHostBytes) std::cout << "the dense model does not fit in video memory: " << (denseHostBytes >> 20) << " MB of its feed forward stay in pinned ram and the gpu reads them from there, " << (onCard >> 20) << " MB on the card" << std::endl;
        }

        uint64_t arena = grab(denseBytes - denseHostBytes + 256);
        if(!arena) return false;
        // al leer un prompt la mlp de cada capa en ram se copia entera a este lugar y se calcula desde la vram
        if(denseHostBytes){
            uint64_t widest = 0;
            for(int l = firstLayer; l < endLayer; l++){
                const GpuLayer& L = layers[l];
                widest = std::max<uint64_t>(widest, ((uint64_t)L.sharedGate.rows * L.sharedGate.stride + 255) / 256 * 256 + ((uint64_t)L.sharedUp.rows * L.sharedUp.stride + 255) / 256 * 256 + ((uint64_t)L.sharedDown.rows * L.sharedDown.stride + 255) / 256 * 256);
            }
            denseStage = grab(widest);
            if(!denseStage) return false;
        }
        void* hostBlock = nullptr;
        uint64_t hostAlias = 0;
        if(denseHostBytes){
            if(cudaLink().hostAlloc(&hostBlock, (size_t)denseHostBytes, 3) != 0 || queue.hostDevice(&hostAlias, hostBlock, 0) != 0){
                problem = "the dense model needs " + std::to_string(denseHostBytes >> 20) + " MB of pinned ram for the layers that do not fit in video memory and windows did not give it";
                return false;
            }
            hostBytes += denseHostBytes;
            pinned.push_back(hostBlock);
        }
        uint64_t used = 0, hostUsed = 0;
        std::vector<std::array<uint64_t, 4>> reads, denseHostReads;
        for(size_t i = 0; i < wanted.size(); i++){
            auto& w = wanted[i];
            uint64_t bytes = (w.first->bytes + 255) / 256 * 256;
            if(onHost[i]){
                *w.second = hostAlias + hostUsed;
                denseHostReads.push_back({w.first->file, w.first->offset, w.first->bytes, (uint64_t)hostBlock + hostUsed});
                hostUsed += bytes;
                continue;
            }
            *w.second = arena + used;
            reads.push_back({w.first->file, w.first->offset, w.first->bytes, arena + used});
            used += bytes;
        }
        if(stopLoading){
            problem = "loading stopped";
            return false;
        }
        loadSay("Reading the weights into video memory");
        if(!fetchAll(reads)) return false;
        reads.clear();
        if(!denseHostReads.empty()){
            loadSay("Copying the layers that do not fit to RAM");
            if(!fetchAll(denseHostReads, true)) return false;
            auto onCpu = [&](const GpuMatrix& m, Matrix& out) {
                if(m.data < hostAlias || m.data >= hostAlias + denseHostBytes) return false;
                out.data = (const uint8_t*)hostBlock + (m.data - hostAlias);
                out.type = m.type;
                out.cols = m.cols;
                out.rows = m.rows;
                out.stride = m.stride;
                return true;
            };
            for(int l = firstLayer; l < endLayer; l++){
                GpuLayer& L = layers[l];
                L.hostFfn = onCpu(L.sharedGate, L.hostGate) && onCpu(L.sharedUp, L.hostUp) && onCpu(L.sharedDown, L.hostDown)
                    && hostDotType(L.hostGate.type) && hostDotType(L.hostUp.type) && hostDotType(L.hostDown.type);
            }
        }
        for(auto& n : narrowed){
            *n.second = grab(n.first.size() * 2);
            if(!*n.second || cudaLink().copyUp(*n.second, n.first.data(), n.first.size() * 2) != 0) return false;
        }
        narrowed.clear();

        if(lead && usePle){
            std::vector<float> convFloats((size_t)4 * hcWidth);
            dequantRow(conv.type, conv.data, convFloats.data(), (int64_t)convFloats.size());
            pleConv = grab(convFloats.size() * 4);
            if(!pleConv || cudaLink().copyUp(pleConv, convFloats.data(), convFloats.size() * 4) != 0) return false;
        }

        auto base = file.meta.find(file.arch + ".rope.freq_base");
        double theta = base == file.meta.end() ? 1e7 : base->second.real;
        ropeTheta = theta;
        auto sections = file.meta.find(file.arch + ".rope.dimension_sections");
        if(sections != file.meta.end() && sections->second.wholes.size() >= 3) for(int i = 0; i < 3; i++) ropeSpans[i] = (int)sections->second.wholes[i];
        imageToken = (int)metaWhole(file, file.arch + ".ple.image_token_id", (uint32_t)-1);
        std::vector<float> cosTable((size_t)contextLimit * 32), sinTable((size_t)contextLimit * 32);
        for(int p = 0; p < contextLimit; p++){
            for(int i = 0; i < 32; i++){
                double angle = p * pow(theta, -2.0 * i / 64.0);
                cosTable[(size_t)p * 32 + i] = (float)cos(angle);
                sinTable[(size_t)p * 32 + i] = (float)sin(angle);
            }
        }
        ropeCos = grab(cosTable.size() * 4);
        ropeSin = grab(sinTable.size() * 4);
        if(!ropeCos || !ropeSin) return false;
        cudaLink().copyUp(ropeCos, cosTable.data(), cosTable.size() * 4);
        cudaLink().copyUp(ropeSin, sinTable.data(), sinTable.size() * 4);
        if(vision && file.arch == "qwen4exp") placesOnGpu = grab((uint64_t)(contextLimit + 8) * 12);

        auto lined = [&](uint64_t width) -> uint64_t {
            uint64_t total = (uint64_t)contextLimit * width * 9 / 8;
            uint64_t base = stretchBuffer(total, {{0, width}, {(uint64_t)contextLimit * width, width / 8}});
            return base ? base : grab(total);
        };
        auto pooledRows = [&]() -> uint64_t {
            uint64_t total = (uint64_t)scoreStride * indexWidth * 4;
            uint64_t base = stretchBuffer(total, {{0, (uint64_t)indexWidth}});
            return base ? base : grab(total);
        };
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            if(L.attention){
                L.keys = lined((uint64_t)kvHeads * headWidth);
                L.values = lined((uint64_t)kvHeads * headWidth);
                if(indexHeads > 0){
                    L.indexRaw = lined((uint64_t)indexWidth);
                    L.pooled = pooledRows();
                }
                cacheBytes += (uint64_t)contextLimit * 2 * kvHeads * headWidth * 9 / 8 + (indexHeads > 0 ? (uint64_t)contextLimit * indexWidth * 9 / 8 + (uint64_t)scoreStride * indexWidth * 4 : 0);
            } else {
                L.state = grab((uint64_t)valueHeads * ssmSize * ssmSize * 4);
                L.convHistory = grab((uint64_t)3 * convWidth * 4);
                cacheBytes += (uint64_t)valueHeads * ssmSize * ssmSize * 4 + (uint64_t)3 * convWidth * 4;
            }
        }
        if(!stretches.empty() && !growContext(std::min(contextLimit, kvStart))) return false;
        pleHistory = grab((uint64_t)pleTaps * hcWidth * 4);
        replayStore = grab(4ULL * (replayStride() * layers.size() + (uint64_t)relayLimit * hcWidth));
        if(vocabPath.empty()) vocabPath = (std::filesystem::path(runtimeFolder()).parent_path() / "profiles" / (std::filesystem::path(file.paths[0]).stem().string() + ".draftvocab")).string();
        if(stopLoading){
            problem = "loading stopped";
            return false;
        }
        if(last && file.arch == "qwen4exp" && !draftPath.empty() && std::filesystem::exists(draftPath)) loadSay("Loading the draft layer");
        if(last && file.arch == "qwen4exp" && !draftPath.empty() && std::filesystem::exists(draftPath) && !loadDraft(draftPath)){
            std::string why = problem;
            problem.clear();
            draftProblem = why;
        }
        if(last && file.arch != "qwen4exp" && !loadTiedDraft((int)file.layers) && !problem.empty()){
            draftProblem = problem;
            problem.clear();
        }

        std::string packPath = packPathFor(path);
        std::error_code absent;
        if(!std::filesystem::exists(packPath, absent)) packPath = (std::filesystem::path(runtimeFolder()).parent_path() / "modelos" / std::filesystem::path(packPath).filename()).string();
        packed = streamPrefill && expertPack.open(packPath, file);
        stageStride = 0;
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride;
            uint64_t upSpan = (uint64_t)L.expertUp.rows * L.expertUp.stride;
            uint64_t downSpan = (uint64_t)L.expertDown.rows * L.expertDown.stride;
            bool fits = packed && expertPack.layers[l].gateBytes == gateSpan && expertPack.layers[l].upBytes == upSpan && expertPack.layers[l].downBytes == downSpan;
            if(packed && !fits) packed = false;
            L.slice = packed
                ? expertPack.layers[l].stride
                : (gateSpan + upSpan + downSpan + 4095) / 4096 * 4096;
            L.piece[0] = 0;
            L.piece[1] = gateSpan;
            L.piece[2] = gateSpan + upSpan;
            stageStride = std::max(stageStride, L.slice);
        }

        uint64_t C = (uint64_t)chunk;
        crestRows = streamPrefill && !legacy
            ? std::min(std::max(chunk, crestRows), superLimit)
            : chunk;
        uint64_t W = (uint64_t)crestRows;
        uint64_t T = (uint64_t)std::max(chunk, streamPrefill ? superLimit : chunk);
        uint64_t A = C * expertsUsed;
        uint64_t B = dense ? 1 : std::max<uint64_t>(A, (uint64_t)batchRows);
        listSpan = (int)(T * expertsUsed + expertsUsed);
        uint64_t S = (uint64_t)listSpan;
        uint64_t R = A + 2 * expertsUsed;
        embedRows = grab(T * hidden * 4);
        stack = grab(T * hcWidth * 4);
        xn = grab(W * hcWidth * 4);
        mixGate = grab(W * hcWidth * 4);
        lo = grab(W * hcRank * 4);
        inj = grab(T * hcCount * 4);
        x = grab(T * hidden * 4);
        y = grab(W * hidden * 4);
        qkv = grab(W * convWidth * 4);
        z = grab(W * innerWidth * 4);
        alphaOut = grab(W * valueHeads * 4);
        betaOut = grab(W * valueHeads * 4);
        u = grab(W * convWidth * 4);
        o = grab(W * innerWidth * 4);
        q = grab(W * queryHeads * 2 * headWidth * 4);
        k = grab(W * kvHeads * headWidth * 4);
        v = grab(W * kvHeads * headWidth * 4);
        iq = grab(W * indexHeads * indexWidth * 4);
        ik = grab(W * indexWidth * 4);
        scores = grab(indexHeads > 0 ? W * scoreStride * 4 : 256);
        cells = grab(W * 4096 * 4);
        positionsOnGpu = grab(T * 4);
        routerOut = grab(W * expertCount * 4);
        scalar = grab(T * 4);
        sharedGateOut = grab(W * expertWidth * 4);
        sharedUpOut = grab(W * expertWidth * 4);
        sharedHidden = grab(W * expertWidth * 4);
        shared = grab(T * hidden * 4);
        moeOut = grab(T * hidden * 8);
        splitScratch = grab(4ULL << 20);
        bool blasWanted = useBlas && streamPrefill && !legacy && !dense && blasLink().ready;
        if(blasWanted){
            uint64_t widest = hcWidth;
            for(int l = firstLayer; l < endLayer; l++){
                uint64_t need = 0;
                for(const GpuMatrix* m : denseOf(layers[l])){
                    need += halfBytes(*m);
                    widest = std::max<uint64_t>(widest, m->cols);
                }
                for(const GpuMatrix* m : mixOf(layers[l])) need += (2ULL * m->rows * m->cols + 255) / 256 * 256;
                halfRoom = std::max(halfRoom, need);
            }
            wideRoom = ((uint64_t)std::max(W, (uint64_t)loanCrest) * widest * 2 + 255) / 256 * 256;
            halfWeights = grab(halfRoom + 256);
            wideRows = grab(2 * wideRoom);
            blasWork = grab(32ULL << 20);
        }
        routeIds = grab(T * expertsUsed * 4);
        routeWeights = grab(T * expertsUsed * 4);
        nextIds = grab(C * expertsUsed * 4);
        nextWeights = grab(C * expertsUsed * 4);
        uint64_t D = (uint64_t)expertsUsed * draftBatch;
        planWidth = std::max<uint64_t>(expertCount, (uint64_t)expertsUsed * std::max(std::max(relayLimit, promptRelay), 16));
        lists = grab(std::max(2 * S + 6 * planWidth, D + 6 * std::max<uint64_t>(expertCount, D)) * 4);
        teamLists = grab(2 * teamSpan());
        expertGateOut = grab(B * expertWidth * 4);
        expertUpOut = grab(B * expertWidth * 4);
        expertHidden = grab(B * expertWidth * 4);
        expertOut = grab(R * hidden * 4);
        pleEmbed = grab(W * hidden * 4);
        pleKeyOut = grab(W * hcWidth * 4);
        pleValueOut = grab(W * hidden * 4);
        gated = grab(W * hcWidth * 4);
        normd = grab(W * hcWidth * 4);
        fin = grab(hidden * 4);
        vocabOut = grab((uint64_t)verifyLimit * head.rows * 4);

        for(PackedInput* packed : {&packs[0], &packs[1], &fines[0], &fines[1]}){
            int capacity = packed == &packs[0] || packed == &fines[0]
                ? (int)std::max(W * innerWidth, T * hidden)
                : (int)(B * expertWidth + W * expertWidth);
            if(cudaLink().alloc(&packed->values, capacity) != 0 || cudaLink().alloc(&packed->scales, capacity / 8) != 0 || cudaLink().alloc(&packed->sums, capacity / 8) != 0){
                problem = "out of video memory for packed inputs";
                return false;
            }
            packed->capacity = capacity;
        }

        auto exchange = [&](uint64_t bytes, uint64_t& alias) -> void* {
            void* p = nullptr;
            bytes = std::max<uint64_t>(bytes, 256);
            if(cudaLink().hostAlloc(&p, (size_t)bytes, 3) != 0 || queue.hostDevice(&alias, p, 0) != 0){
                if(problem.empty()) problem = "cannot map " + std::to_string(bytes >> 20) + " MB of host memory for the gpu";
                return nullptr;
            }
            pinned.push_back(p);
            return p;
        };
        uint64_t H = streamPrefill && !legacy && !dense
            ? std::max(T, (uint64_t)loanRows)
            : T;
        uint64_t spanH = H * expertsUsed + expertsUsed;
        hostEmbed = (float*)exchange(H * hidden * 4, mappedEmbed);
        hostPle = (float*)exchange(H * hidden * 4, mappedPle);
        hostPositions = (int*)exchange(H * 4, mappedPositions);
        uint64_t staged = (uint64_t)layers.size() * std::max(relayLimit, 1);
        hostIds = (int*)exchange(std::max(H, staged) * expertsUsed * 4, mappedIds);
        hostNext = (int*)exchange(staged * expertsUsed * 4, mappedNext);
        hostWeights = (float*)exchange(H * expertsUsed * 4, mappedWeights);
        hostX = (float*)exchange(std::max(C, staged) * hidden * 4, mappedX);
        hostXRows = std::max(C, staged);
        hostLists = (int*)exchange((2 * spanH + 6 * expertCount) * 4, mappedLists);
        hostExpertOut = (float*)exchange(std::max(R, staged * expertsUsed) * hidden * 4, mappedExpertOut);
        hostExpertRows = std::max(R, staged * expertsUsed);
        hostLogits = (float*)exchange((uint64_t)verifyLimit * head.rows * 4, mappedLogits);
        hostFlags = (volatile uint32_t*)exchange(256, mappedFlags);
        hostTimeline = (volatile unsigned long long*)exchange((uint64_t)layers.size() * 32, mappedTimeline);
        cpuSaw.assign(layers.size(), 0.0);
        cpuDone.assign(layers.size(), 0.0);
        cpuGather.assign(layers.size(), 0.0);
        cpuSeen.assign(layers.size(), 0);
        spillCounts = grab((uint64_t)layers.size() * 4);
        if(dense && denseHostBytes && spillCounts){
            hostDenseOut = (float*)exchange(C * hidden * 4, mappedDenseOut);
            std::vector<int> waits(layers.size(), 0);
            for(int l = firstLayer; l < endLayer; l++) waits[l] = layers[l].hostFfn ? 1 : 0;
            if(cudaLink().copyUp(spillCounts, waits.data(), waits.size() * 4) != 0) problem = "cannot mark the layers the cpu computes";
        }
        if(hostFlags){
            for(int i = 0; i < 64; i++) hostFlags[i] = 0;
        }
        slotTable = grab((uint64_t)layers.size() * expertCount * 4);
        if(streamPrefill){
            stageSides = std::min(std::max(stageSides, 2), 8);
            for(int i = 0; i < stageSides; i++){
                stage[i] = grab((uint64_t)stageSlots * stageStride + 256);
                ring[i] = (uint8_t*)host((uint64_t)stageSlots * stageStride + 4096);
            }
        }
        if(!problem.empty()) return false;
        bool events = queue.eventCreate(&routed, 2) == 0 && queue.streamCreate(&stream, 0) == 0;
        if(streamPrefill){
            events = events && queue.streamCreate(&copyStream, 1) == 0;
            for(int i = 0; i < stageSides && events; i++) events = queue.eventCreate(&moved[i], 2) == 0 && queue.eventCreate(&stageFree[i], 2) == 0;
        }
        if(!events){
            problem = "cannot create a cuda event";
            return false;
        }
        if(halfWeights && wideRows && blasWork && blasLink().create(&blas) == 0){
            if(blasLink().setStream(blas, stream) != 0 || blasLink().setWorkspace(blas, (void*)blasWork, 32ULL << 20) != 0){
                blasLink().destroy(blas);
                blas = nullptr;
            }
        }

        if(dense){
            // sin fijar los hilos: con las capas en ram los nucleos e tambien suman y fijarlos a los p fue 13% mas lento
            pool.start(std::max(0, threads - 1));
            ready = true;
            reset();
            size_t freeNow = 0, totalNow = 0;
            cudaLink().info(&freeNow, &totalNow);
            vramHeld = freeAtStart > freeNow ? freeAtStart - freeNow : 0;
            return problem.empty();
        }

        if(heat.empty()){
            std::string path = heatPath.empty()
                ? heatFor(file.paths[0], layers.size(), expertCount)
                : heatPath;
            if(!path.empty() && readHeat(path)) heatSource = path;
        }
        recent.assign(layers.size() * expertCount, 0.0f);
        learned.assign(layers.size() * expertCount, 0);
        guessMark.assign(layers.size() * expertCount, 0);

        uint64_t spans[3] = {};
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            spans[0] += (uint64_t)L.expertGate.rows * L.expertGate.stride;
            spans[1] += (uint64_t)L.expertUp.rows * L.expertUp.stride;
            spans[2] += (uint64_t)L.expertDown.rows * L.expertDown.stride;
        }
        uint64_t perExpert = spans[0] + spans[1] + spans[2];

        if(vision && placesOnGpu) openVision(path);
        bool told = expertBudget > 0 || vramCap > 0;
        // extraKeep crece cuando otro programa se quedo con la memoria que esta carga iba a pedir despues y la tarjeta se lleno
        uint64_t keep = (told
            ? std::min<uint64_t>(vramReserve, 64ULL << 20)
            : vramReserve) + extraKeep;
        uint64_t snapshotBytes = (uint64_t)pleTaps * hcWidth * 4;
        for(int l = firstLayer; l < endLayer; l++){
            if(!layers[l].attention) snapshotBytes += (uint64_t)valueHeads * ssmSize * ssmSize * 4 + (uint64_t)3 * convWidth * 4;
        }
        uint64_t later = (256ULL << 20) + 3 * snapshotBytes;
        auto expertRoom = [&]() -> uint64_t {
            size_t freeBytes = 0, totalBytes = 0;
            cudaLink().info(&freeBytes, &totalBytes);
            NvmlLink& nv = nvmlLink();
            unsigned long long physical[3] = {0, 0, 0};
            bool whole = nv.ready && nv.getMemory(nv.device, physical) == 0 && physical[0] > 0;
            if(whole) freeBytes = (size_t)physical[1];
            uint64_t room = freeBytes > keep + later
                ? freeBytes - keep - later
                : 0;
            uint64_t spent = freeAtStart > freeBytes ? freeAtStart - freeBytes : 0;
            // lo libre de nvml ve toda la tarjeta y lo de cuda en windows solo este proceso: con otro motor en la misma tarjeta
            // se contaba como gastado por este lo que tenia el otro y el segundo modelo cargaba sin expertos en la vram
            // ahora se mide con nvml de punta a punta, y solo con un motor da lo mismo que antes
            if(whole && cardAtStart > 0){
                uint64_t used = cardAtStart > freeBytes ? cardAtStart - freeBytes : 0;
                uint64_t lent = cardAtStart > freeAtStart ? cardAtStart - freeAtStart : 0;
                spent = used > lent ? used - lent : 0;
            }
            if(vramCap > 0) room = std::min<uint64_t>(room, vramCap > spent + later ? vramCap - spent - later : 0);
            if(expertBudget > 0) room = std::min<uint64_t>(room, expertBudget);
            // windows le da a este programa un presupuesto en la tarjeta menor que lo libre que ve nvml: los expertos lo llenaban
            // y lo que se pide despues (la tabla de calor de 96 KB) fallaba, o windows mandaba parte de la vram a la ram
            uint64_t budgetNow = 0, usageNow = 0;
            if(vramBudget.read(budgetNow, usageNow) && budgetNow > 0){
                uint64_t allowed = budgetNow > usageNow + keep + later ? budgetNow - usageNow - keep - later : 0;
                room = std::min(room, allowed);
            }
            return room;
        };
        int planned = (int)std::min<uint64_t>(expertCount, expertRoom() / perExpert);

        MEMORYSTATUSEX memory = {};
        memory.dwLength = sizeof(memory);
        GlobalMemoryStatusEx(&memory);
        hostCap = hostBudget > 0
            ? hostBudget
            : memory.ullAvailPhys > (5ULL << 30)
                ? memory.ullAvailPhys - (4ULL << 30)
                : 1ULL << 30;
        uint64_t pinnedRoom = hostCap > (3ULL << 30) ? hostCap - (2ULL << 30) : 0;
        uint64_t availableNow = memory.ullAvailPhys > (5ULL << 29) ? memory.ullAvailPhys - (5ULL << 29) : 0;
        uint64_t pinnable = memory.ullTotalPhys / 100 * std::max(10, std::min(95, pinShare));
        uint64_t shared = pinGb > 0
            ? (uint64_t)(pinGb * 1073741824.0)
            : pinnable > (1ULL << 30)
                ? pinnable - (1ULL << 30)
                : 0;
        pinnedRoom = std::min({pinnedRoom, availableNow, shared});
        // la memoria fijada y mapeada cuenta contra la ram que windows deja ver a la tarjeta (la mitad de la ram): fijar hasta que
        // el driver se negaba dejaba a veces menos de 1 GB y despues fallaba una tabla de 96 KB o la carga de kernels
        // lo que suelta el modelo anterior o un intento fallido tarda en volver a esa cuenta: se espera a que deje de bajar antes de medir
        const uint64_t sharedMargin = 5ULL << 29;
        auto sharedRoom = [&]() -> uint64_t {
            uint64_t budget = 0, usage = 0, last = ~0ULL;
            uint64_t began = GetTickCount64();
            for(int steady = 0; steady < 3 && GetTickCount64() - began < 10000; Sleep(100)){
                if(!vramBudget.readShared(budget, usage) || budget == 0) return ~0ULL;
                steady = usage + (64ULL << 20) >= last ? steady + 1 : 0;
                last = std::min(last, usage);
            }
            if(!vramBudget.readShared(budget, usage) || budget == 0) return ~0ULL;
            return budget > usage + sharedMargin ? budget - usage - sharedMargin : 0;
        };
        pinnedRoom = std::min<uint64_t>(pinnedRoom, sharedRoom());
#ifndef _WIN32
        // wsl deja fijar poco mas de 1 GB de ram para la tarjeta aunque haya mucha mas libre y sin presupuesto que lo diga:
        // se mide lo que queda y 512 MB se guardan para los buffers que se fijan despues, el resto de los expertos lo hace la cpu
        if(pinWarm && access("/dev/dxg", F_OK) == 0){
            std::vector<void*> taken;
            uint64_t got = 0;
            const uint64_t piece = 64ULL << 20;
            while(got < pinnedRoom + (512ULL << 20)){
                void* p = nullptr;
                if(cudaLink().hostAlloc(&p, (size_t)piece, 0) != 0) break;
                taken.push_back(p);
                got += piece;
            }
            for(void* p : taken) cudaLink().hostFree(p);
            pinnedRoom = std::min<uint64_t>(pinnedRoom, got > (512ULL << 20) ? got - (512ULL << 20) : 0);
            std::cout << "wsl lets the card pin " << (got >> 20) << " MB of ram, " << (pinnedRoom >> 20) << " MB of it for experts" << std::endl;
        }
#endif
        warmPerLayer = pinWarm
            ? (int)std::min<uint64_t>(expertCount - planned, pinnedRoom / perExpert)
            : 0;
        // lugares de sobra para los expertos que salen de la vram cuando crece el contexto
        int spareRooms = pinWarm && spareWarm && !stretches.empty() && planned < expertCount
            ? (int)std::min<uint64_t>((uint64_t)recentFor(planned) + recentSpare, pinnedRoom / perExpert - (uint64_t)warmPerLayer)
            : 0;
        int warmTries = 0;
        loadSay("Pinning the experts in RAM");
        while(warmPerLayer > 0){
            if(stopLoading){
                problem = "loading stopped";
                return false;
            }
            size_t mark = pinned.size();
            bool fits = true;
            for(int l = firstLayer; l < endLayer && fits; l++){
                GpuLayer& L = layers[l];
                const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
                for(int i = 0; i < 3 && fits; i++){
                    void* p = nullptr;
                    uint64_t alias = 0;
                    fits = cudaLink().hostAlloc(&p, (size_t)((uint64_t)(warmPerLayer + spareRooms) * parts[i]->rows * parts[i]->stride + 256), 3) == 0 && queue.hostDevice(&alias, p, 0) == 0;
                    if(p) pinned.push_back(p);
                    L.warmHost[i] = (uint8_t*)p;
                    L.warmBase[i] = alias;
                }
            }
            if(fits) break;
            while(pinned.size() > mark){
                cudaLink().hostFree(pinned.back());
                pinned.pop_back();
            }
            warmTries++;
            if(spareRooms > 0) spareRooms = 0;
            else warmPerLayer -= std::max(1, warmPerLayer / 8);
            // el intento fallido se devuelve de a poco: reintentar enseguida volvia a fallar hasta quedar en cero expertos fijados
            uint64_t left = sharedRoom();
            if(left != ~0ULL) warmPerLayer = (int)std::min<uint64_t>((uint64_t)std::max(0, warmPerLayer), left / perExpert);
            // con lugar de sobra segun windows y igual sin poder fijar nada, este proceso quedo roto despues de cambiar de modelo:
            // uno nuevo si fija, asi que se pide reiniciar en vez de seguir y fallar despues en una tabla de 96 KB
            if(warmTries >= 6 && left != ~0ULL && left > (4ULL << 30)){
                pinLost = true;
                problem = "pinned memory stopped working in this process after the model change (" + std::to_string(left >> 30) + " GB of room left)";
                std::cout << "pinning the experts failed " << warmTries << " times with " << (left >> 30) << " GB of room, the engine starts fresh" << std::endl;
                return false;
            }
        }
        if(warmTries > 0) std::cout << "pinning the experts in ram took " << warmTries + 1 << " tries, " << warmPerLayer << " per layer fit" << std::endl;
        if(warmPerLayer == 0){
            spareRooms = 0;
            for(GpuLayer& L : layers){
                for(int i = 0; i < 3; i++){
                    L.warmHost[i] = nullptr;
                    L.warmBase[i] = 0;
                }
            }
        }
        warmRooms = warmPerLayer + spareRooms;
        warmPinnedBytes = (uint64_t)warmRooms * perExpert;

        spareStride = 0;
        for(int l = firstLayer; l < endLayer; l++) spareStride = std::max<uint64_t>(spareStride, (expertBytes(l) + 255) / 256 * 256);
        spareBase = 0;
        if(warmPerLayer > 0 && spareCount > 0 && cudaLink().alloc(&spareBase, (size_t)(spareStride * spareCount)) == 0) allocations.push_back(spareBase);
        else spareBase = 0;
        moving.clear();
        leaving.clear();

        uint64_t room = expertRoom();
        residentExperts = (int)std::min<uint64_t>(expertCount - warmPerLayer, room / perExpert);
        uint64_t growth = 0;
        std::vector<char> pending;
        for(const Stretch& s : stretches) growth += stretchNeed(s, contextLimit, pending);
        if(growth > 0){
            uint64_t need = growth + 3ULL * (uint64_t)(endLayer - firstLayer) * pageGrain + (512ULL << 20);
            auto shortfall = [&](int resident) -> uint64_t {
                uint64_t cover = keep + (uint64_t)recentFor(resident) * perExpert;
                return need > cover ? need - cover : 0;
            };
            while(residentExperts > 0 && (uint64_t)residentExperts * perExpert + shortfall(residentExperts) > room) residentExperts--;
        }
        while(residentExperts > 0){
            size_t mark = allocations.size(), spanMark = poolSpans.size();
            int recentGuess = recentFor(residentExperts);
            int spare = std::min(recentGuess > 0 ? 16 : 0, residentExperts - recentGuess);
            bool fits = true;
            for(int l = firstLayer; l < endLayer && fits; l++){
                GpuLayer& L = layers[l];
                const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
                for(int i = 0; i < 3 && fits; i++){
                    uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride;
                    uint64_t ptr = poolBuffer((uint64_t)residentExperts * span + 256, (uint64_t)(residentExperts - recentGuess - spare) * span);
                    fits = ptr != 0;
                    L.resident[i] = ptr;
                }
            }
            if(fits) break;
            while(allocations.size() > mark){
                cudaLink().release(allocations.back());
                allocations.pop_back();
            }
            while(poolSpans.size() > spanMark){
                dropPool(poolSpans.back());
                poolSpans.pop_back();
            }
            residentExperts--;
        }
        residentBytes = (uint64_t)residentExperts * perExpert;
        {
            uint64_t budgetNow = 0, usageNow = 0, sharedBudget = 0, sharedUsage = 0;
            vramBudget.readShared(sharedBudget, sharedUsage);
            if(vramBudget.read(budgetNow, usageNow) && budgetNow > 0) std::cout << "experts in video memory take " << (residentBytes >> 20) << " MB, windows budget " << (budgetNow >> 20) << " MB used " << (usageNow >> 20) << " MB"
                << ", shared budget " << (sharedBudget >> 20) << " MB used " << (sharedUsage >> 20) << " MB (" << (warmPinnedBytes >> 20) << " MB of experts pinned)" << std::endl;
        }
        recentPerLayer = recentFor(residentExperts);
        fixedExperts = residentExperts - recentPerLayer;
        int held = residentExperts - std::min(recentSpare, recentPerLayer / 2);
        ramFree.assign(layers.size(), {});
        spareFree.assign(layers.size(), {});
        ramComing.assign(layers.size(), 0);
        recentHolds.assign(layers.size() * recentPerLayer, -1);
        recentUsed.assign(layers.size() * recentPerLayer, 0);
        recentBusy.assign(layers.size() * recentPerLayer, 0);
        arriving.assign(layers.size() * expertCount, 0);
        uint64_t warmTable = grab((uint64_t)layers.size() * expertCount * 4);
        if(!warmTable) return false;
        int helpPerLayer = openHelper(sizes + quantKernelSource + modelKernelSource, blob, perExpert, expertCount - held);
        warmPerLayer = std::min(warmPerLayer, expertCount - held - helpPerLayer);

        // el ayudante toma los expertos que siguen a los de la tarjeta principal o los que quedan despues de la ram fijada
        // una tarjeta lenta como una 1070 rinde igual que la ram y va despues, una rapida como otra 3090 rinde mas antes
        bool helperAhead = helperFirst || recentPerLayer == 0;
        int helpFrom = helperAhead
            ? held
            : held + warmPerLayer;
        int warmFrom = helperAhead
            ? held + helpPerLayer
            : held;
        std::vector<std::array<uint64_t, 4>> hostReads;
        std::vector<std::array<uint64_t, 4>> helperReads;
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            std::vector<int> order(expertCount);
            std::iota(order.begin(), order.end(), 0);
            if(heat.size() == layers.size() * expertCount){
                const uint64_t* use = heat.data() + (size_t)l * expertCount;
                std::stable_sort(order.begin(), order.end(), [&](int a, int b){ return use[a] > use[b]; });
            }
            std::sort(order.begin() + warmFrom, order.begin() + std::min(expertCount, warmFrom + warmPerLayer));
            L.slot.assign(expertCount, -1);
            L.warm.assign(expertCount, -1);
            L.help.assign(expertCount, -1);
            for(int s = 0; s < held; s++) L.slot[order[s]] = s;
            for(int s = fixedExperts; s < held; s++) recentHolds[(size_t)l * recentPerLayer + s - fixedExperts] = order[s];
            for(int s = 0; s < helpPerLayer; s++) L.help[order[helpFrom + s]] = s;
            for(int s = 0; s < warmPerLayer; s++) L.warm[order[warmFrom + s]] = s;
            for(int r = warmRooms - 1; r >= warmPerLayer; r--) spareFree[l].push_back(r);
            L.slots = slotTable + 4ULL * l * expertCount;
            L.warmSlots = warmTable + 4ULL * l * expertCount;
            if(cudaLink().copyUp(L.slots, L.slot.data(), expertCount * 4) != 0 || cudaLink().copyUp(L.warmSlots, L.warm.data(), expertCount * 4) != 0){
                problem = "cannot upload the expert slots";
                return false;
            }

            const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
            for(int s = 0; s < held;){
                int run = 1;
                while(s + run < held && order[s + run] == order[s] + run) run++;
                for(int i = 0; i < 3; i++){
                    uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride;
                    reads.push_back({L.expertTensor[i]->file, L.expertTensor[i]->offset + (uint64_t)order[s] * span, (uint64_t)run * span, L.resident[i] + (uint64_t)s * span});
                }
                s += run;
            }
            for(int s = 0; s < helpPerLayer;){
                int run = 1;
                const int* chosen = order.data() + helpFrom;
                while(s + run < helpPerLayer && chosen[s + run] == chosen[s] + run) run++;
                for(int i = 0; i < 3; i++){
                    uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride;
                    helperReads.push_back({L.expertTensor[i]->file, L.expertTensor[i]->offset + (uint64_t)chosen[s] * span, (uint64_t)run * span, L.helpBase[i] + (uint64_t)s * span});
                }
                s += run;
            }
            for(int s = 0; s < warmPerLayer;){
                int run = 1;
                const int* chosen = order.data() + warmFrom;
                while(s + run < warmPerLayer && chosen[s + run] == chosen[s] + run) run++;
                for(int i = 0; i < 3; i++){
                    uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride;
                    hostReads.push_back({L.expertTensor[i]->file, L.expertTensor[i]->offset + (uint64_t)chosen[s] * span, (uint64_t)run * span, (uint64_t)(L.warmHost[i] + (uint64_t)s * span)});
                }
                s += run;
            }
        }

        if(stopLoading){
            problem = "loading stopped";
            return false;
        }
        loadSay("Reading the experts into video memory");
        if(!fetchAll(reads)) return false;
        loadSay("Copying the experts to RAM");
        if(!fetchAll(hostReads, true)) return false;
        if(helpPerLayer > 0) loadSay("Filling the second GPU");
        if(helpPerLayer > 0 && !fetchAll(helperReads, false, helper.context)){
            if(stopLoading) return false;
            helper.problem = "could not fill the second gpu: " + problem;
            problem.clear();
            closeHelper();
        }

        expertTally.assign(expertCount, 0);
        expertBegin.assign(expertCount, 0);
        expertHits.assign(layers.size() * expertCount, 0);
        if(recentPerLayer > 0 && !startLifting()){
            recentPerLayer = 0;
            fixedExperts = residentExperts;
        }
        if(recentPerLayer > 0 && streamPrefill && !legacy && !dense && pageGrain > 0 && !poolSpans.empty()){
            uint64_t most = (uint64_t)loanTail(0).size() * pageGrain;
            MEMORYSTATUSEX memory = {};
            memory.dwLength = sizeof(memory);
            GlobalMemoryStatusEx(&memory);
            void* room = nullptr;
            if(most > 0 && memory.ullAvailPhys > most + ramHeadroom + (8ULL << 30) && cudaLink().hostAlloc(&room, (size_t)most, 0) == 0){
                pinned.push_back(room);
                loanStore = (uint8_t*)room;
                loanStoreSize = most;
            }
        }
        warmUp();
        std::vector<GROUP_AFFINITY> places = coresFor(cpuPlan);
        hostPlace = places.empty() ? GROUP_AFFINITY{} : places[0];
        if(places.size() > 1) pool.start((int)places.size() - 1, std::vector<GROUP_AFFINITY>(places.begin() + 1, places.end()));
        else pool.start(std::max(0, threads - 1));
        ready = true;
        reset();
        size_t freeNow = 0, totalNow = 0;
        cudaLink().info(&freeNow, &totalNow);
        vramHeld = freeAtStart > freeNow ? freeAtStart - freeNow : 0;
        return problem.empty();
    }

    uint64_t expertBytes(int l) {
        GpuLayer& L = layers[l];
        return (uint64_t)L.expertGate.rows * L.expertGate.stride + (uint64_t)L.expertUp.rows * L.expertUp.stride + (uint64_t)L.expertDown.rows * L.expertDown.stride;
    }

    void forget(int id) {
        int l = id / expertCount, e = id % expertCount;
        warmOrder.erase({warmClock[id], id});
        warmClock[id] = 0;
        warmBytes -= expertBytes(l);
        GpuLayer& L = layers[l];
        const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
        for(int i = 0; i < 3; i++){
            uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride;
            VirtualUnlock((LPVOID)(parts[i]->data + (uint64_t)e * span), (SIZE_T)span);
        }
    }

    void touch(int l, int e) {
        if(layers[l].warmHost[0] && layers[l].warm[e] >= 0) return;
        if(hostExpert(l, 0, e, nullptr)) return;
        uint64_t now = GetTickCount64();
        if(now - warmChecked > 1000){
            warmChecked = now;
            MEMORYSTATUSEX memory = {};
            memory.dwLength = sizeof(memory);
            GlobalMemoryStatusEx(&memory);
            uint64_t ceiling = hostCap > warmPinnedBytes + (2ULL << 30) ? hostCap - warmPinnedBytes - (2ULL << 30) : 0;
            uint64_t room = warmBytes + (memory.ullAvailPhys > ramHeadroom ? memory.ullAvailPhys - ramHeadroom : 0);
            uint64_t next = std::max(holdRam || ramWhole ? warmKeep : 0, std::min(ceiling, room));
            if(next > warmBudget + (256ULL << 20)) SetProcessWorkingSetSizeEx(GetCurrentProcess(), (SIZE_T)(next + (512ULL << 20)), (SIZE_T)(hostCap + (1ULL << 30)), QUOTA_LIMITS_HARDWS_MAX_ENABLE | QUOTA_LIMITS_HARDWS_MIN_DISABLE);
            warmBudget = next;
        }
        int id = l * expertCount + e;
        if(warmClock[id]) warmOrder.erase({warmClock[id], id});
        else warmBytes += expertBytes(l);
        warmClock[id] = ++warmTick;
        warmOrder.insert({warmTick, id});
        while(warmBytes > warmBudget && warmOrder.size() > 1 && warmOrder.begin()->second != id) forget(warmOrder.begin()->second);
    }

    void warmUp() {
        warmClock.assign(layers.size() * expertCount, 0);
        warmOrder.clear();
        warmBytes = 0;
        // todo el modelo en ram con la tabla ple los embeddings y los expertos sin copia fija tambien los de la vram
        std::vector<std::pair<const uint8_t*, uint64_t>> tables;
        uint64_t tableBytes = 0;
        if(ramWhole){
            if(usePle && pleTable.data) tables.push_back({pleTable.data, (uint64_t)pleTable.rows * pleTable.stride});
            if(embed.data) tables.push_back({embed.data, (uint64_t)embed.rows * embed.stride});
            for(const auto& t : tables) tableBytes += t.second;
        }
        warmBudget = hostCap > warmPinnedBytes + (2ULL << 30) ? hostCap - warmPinnedBytes - (2ULL << 30) : 0;
        MEMORYSTATUSEX memory = {};
        memory.dwLength = sizeof(memory);
        GlobalMemoryStatusEx(&memory);
        uint64_t spare = memory.ullAvailPhys > ramHeadroom ? memory.ullAvailPhys - ramHeadroom : 0;
        tableBytes = std::min(tableBytes, spare);
        warmBudget = std::min<uint64_t>(warmBudget, spare - tableBytes);
        warmKeep = warmBudget;
        uint64_t floor = warmBudget + tableBytes + (512ULL << 20);
        SetProcessWorkingSetSizeEx(GetCurrentProcess(), (SIZE_T)floor, (SIZE_T)std::max(hostCap + (1ULL << 30), floor + (1ULL << 30)), QUOTA_LIMITS_HARDWS_MAX_ENABLE | QUOTA_LIMITS_HARDWS_MIN_DISABLE);
        std::vector<int> order;
        for(int l = firstLayer; l < endLayer; l++){
            for(int e = 0; e < expertCount; e++){
                if((ramWhole || layers[l].slot[e] < 0) && layers[l].warm[e] < 0 && (layers[l].help.empty() || layers[l].help[e] < 0)) order.push_back(l * expertCount + e);
            }
        }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b){
            if(!heat.empty()) return heat[a] > heat[b];
            return a % expertCount != b % expertCount ? a % expertCount < b % expertCount : a < b;
        });
        std::vector<int> chosen;
        uint64_t total = 0;
        for(int id : order){
            uint64_t bytes = expertBytes(id / expertCount);
            if(total + bytes > warmBudget) break;
            chosen.push_back(id);
            total += bytes;
        }
        for(int i = (int)chosen.size() - 1; i >= 0; i--){
            warmClock[chosen[i]] = ++warmTick;
            warmOrder.insert({warmTick, chosen[i]});
        }
        warmBytes = total;
        warmExperts = (int)chosen.size();
        wholeBytes = ramWhole ? total + tableBytes : 0;
        warmer = std::thread([this, chosen, tables, tableBytes]{
            std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
            for(size_t i = 0; i < chosen.size() && !warmStop; i++){
                int l = chosen[i] / expertCount, e = chosen[i] % expertCount;
                GpuLayer& L = layers[l];
                const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
                for(int p = 0; p < 3; p++){
                    uint64_t span = (uint64_t)parts[p]->rows * parts[p]->stride;
                    ranges.push_back({(PVOID)(parts[p]->data + (uint64_t)e * span), (SIZE_T)span});
                }
                if(ranges.size() >= 192 || i + 1 == chosen.size()){
                    PrefetchVirtualMemory(GetCurrentProcess(), ranges.size(), ranges.data(), 0);
                    uint32_t sink = 0;
                    for(const WIN32_MEMORY_RANGE_ENTRY& r : ranges){
                        const volatile uint8_t* p = (const volatile uint8_t*)r.VirtualAddress;
                        for(SIZE_T at = 0; at < r.NumberOfBytes && !warmStop; at += 4096) sink += p[at];
                    }
                    touched += sink & 1;
                    for(const WIN32_MEMORY_RANGE_ENTRY& r : ranges){
                        if(!locking) break;
                        locking = VirtualLock(r.VirtualAddress, r.NumberOfBytes) != 0;
                    }
                    ranges.clear();
                }
            }
            uint64_t left = tableBytes;
            for(const auto& t : tables){
                for(uint64_t at = 0; at < t.second && left > 0 && !warmStop; at += 64ULL << 20){
                    uint64_t bytes = std::min<uint64_t>({64ULL << 20, t.second - at, left});
                    WIN32_MEMORY_RANGE_ENTRY r = {(PVOID)(t.first + at), (SIZE_T)bytes};
                    PrefetchVirtualMemory(GetCurrentProcess(), 1, &r, 0);
                    uint32_t sink = 0;
                    const volatile uint8_t* p = (const volatile uint8_t*)r.VirtualAddress;
                    for(SIZE_T step = 0; step < r.NumberOfBytes && !warmStop; step += 4096) sink += p[step];
                    touched += sink & 1;
                    if(locking) locking = VirtualLock(r.VirtualAddress, r.NumberOfBytes) != 0;
                    left -= bytes;
                }
            }
            warmDone = true;
        });
    }

    bool readHeat(const std::string& path) {
        FILE* f = nullptr;
        fopen_s(&f, path.c_str(), "rb");
        if(!f) return false;
        char tag[4] = {};
        uint32_t version = 0, count = 0, width = 0;
        uint64_t seen = 0;
        bool ok = fread(tag, 1, 4, f) == 4 && fread(&version, 4, 1, f) == 1 && fread(&count, 4, 1, f) == 1 && fread(&width, 4, 1, f) == 1 && fread(&seen, 8, 1, f) == 1;
        ok = ok && memcmp(tag, "HEAT", 4) == 0 && version == 1 && count == layers.size() && width == (uint32_t)expertCount;
        std::vector<uint64_t> counts((size_t)layers.size() * expertCount);
        ok = ok && fread(counts.data(), 8, counts.size(), f) == counts.size();
        fclose(f);
        if(!ok) return false;
        heat = counts;
        heatTokens = seen;
        return true;
    }

    bool writeHeat(const std::string& path, const std::vector<uint64_t>& counts, uint64_t seen) {
        FILE* f = nullptr;
        fopen_s(&f, path.c_str(), "wb");
        if(!f) return false;
        uint32_t version = 1, count = (uint32_t)layers.size(), width = (uint32_t)expertCount;
        fwrite("HEAT", 1, 4, f);
        fwrite(&version, 4, 1, f);
        fwrite(&count, 4, 1, f);
        fwrite(&width, 4, 1, f);
        fwrite(&seen, 8, 1, f);
        bool ok = fwrite(counts.data(), 8, counts.size(), f) == counts.size();
        fclose(f);
        return ok;
    }

    bool saveHeat() {
        if(heatSave.empty() || learned.size() != layers.size() * expertCount || endLayer <= firstLayer || expertsUsed <= 0) return false;
        uint64_t routes = 0;
        for(int l = firstLayer; l < endLayer; l++){
            for(int e = 0; e < expertCount; e++) routes += learned[(size_t)l * expertCount + e];
        }
        uint64_t fresh = routes / ((uint64_t)(endLayer - firstLayer) * expertsUsed);
        if(fresh == 0) return false;
        if(heat.size() != learned.size()){
            heat.assign(learned.size(), 0);
            heatTokens = 0;
        }
        double fade = 65536.0 / (65536.0 + (double)fresh);
        for(int l = firstLayer; l < endLayer; l++){
            for(int e = 0; e < expertCount; e++){
                size_t at = (size_t)l * expertCount + e;
                heat[at] = (uint64_t)((double)heat[at] * fade + 0.5) + learned[at];
                learned[at] = 0;
            }
        }
        heatTokens = (uint64_t)((double)heatTokens * fade + 0.5) + fresh;
        std::error_code error;
        std::filesystem::create_directories(std::filesystem::path(heatSave).parent_path(), error);
        std::string part = heatSave + ".part";
        if(!writeHeat(part, heat, heatTokens)) return false;
        if(!MoveFileExA(part.c_str(), heatSave.c_str(), MOVEFILE_REPLACE_EXISTING)) return false;
        heatSource = heatSave;
        return true;
    }

    struct Arrival {
        int layer;
        int expert;
        int slot;
    };

    std::vector<Arrival> arrivals;
    std::vector<char> arriving;
    void* swapDone = nullptr;

    bool settle(bool wait) {
        if(arrivals.empty()) return true;
        CudaQueue& queue = cudaQueue();
        if(wait) queue.eventSync(swapDone);
        else if(queue.eventQuery(swapDone) != 0) return false;
        std::vector<char> dirty(layers.size(), 0);
        for(const Arrival& a : arrivals){
            GpuLayer& L = layers[a.layer];
            L.slot[a.expert] = a.slot;
            int id = a.layer * expertCount + a.expert;
            arriving[id] = 0;
            if(warmClock[id]) forget(id);
            dirty[a.layer] = 1;
        }
        arrivals.clear();
        for(size_t l = 0; l < layers.size(); l++){
            if(dirty[l]) cudaLink().copyUp(layers[l].slots, layers[l].slot.data(), expertCount * 4);
        }
        return true;
    }

    int adapt(int limit) {
        if(!ready || residentExperts == 0 || residentExperts >= expertCount || !useCuda()) return 0;
        if(!settle(false)) return 0;
        CudaQueue& queue = cudaQueue();
        if(!swapDone && queue.eventCreate(&swapDone, 2) != 0) return 0;
        if(arriving.size() != layers.size() * expertCount) arriving.assign(layers.size() * expertCount, 0);
        double rate = heatTokens > 0 ? 64.0 / (double)heatTokens : 0.0;
        auto score = [&](int l, int e){
            size_t at = (size_t)l * expertCount + e;
            double base = heat.empty() ? 0.0 : (double)heat[at] * rate;
            return base + (double)recent[at];
        };

        int most = std::min(limit, ring[0] ? 2 * stageSlots : limit);
        std::vector<char> dirty(layers.size(), 0);
        while((int)arrivals.size() < most){
            double gain = 1.0;
            int layer = -1, incoming = -1, outgoing = -1;
            for(int l = firstLayer; l < endLayer; l++){
                GpuLayer& L = layers[l];
                int hot = -1, cold = -1;
                double hotScore = -1, coldScore = 1e30;
                for(int e = 0; e < expertCount; e++){
                    double v = score(l, e);
                    if(L.slot[e] < 0 && !arriving[(size_t)l * expertCount + e] && v > hotScore){
                        hotScore = v;
                        hot = e;
                    }
                    if(L.slot[e] >= 0 && L.slot[e] < fixedExperts && v < coldScore){
                        coldScore = v;
                        cold = e;
                    }
                }
                if(hot >= 0 && cold >= 0 && hotScore - coldScore > gain){
                    gain = hotScore - coldScore;
                    layer = l;
                    incoming = hot;
                    outgoing = cold;
                }
            }
            if(layer < 0) break;
            GpuLayer& L = layers[layer];
            arrivals.push_back({layer, incoming, L.slot[outgoing]});
            arriving[(size_t)layer * expertCount + incoming] = 1;
            L.slot[outgoing] = -1;
            dirty[layer] = 1;
        }
        for(float& r : recent) r *= 0.5f;
        if(arrivals.empty()) return 0;

        for(size_t l = 0; l < layers.size(); l++){
            if(dirty[l]) cudaLink().copyUp(layers[l].slots, layers[l].slot.data(), expertCount * 4);
        }

        std::vector<int> staged;
        for(int i = 0; i < (int)arrivals.size(); i++){
            const GpuLayer& L = layers[arrivals[i].layer];
            if(!(L.warmHost[0] && L.warm[arrivals[i].expert] >= 0)) staged.push_back(i);
        }
        if(!staged.empty() && !ring[0]){
            for(Arrival& a : arrivals){
                GpuLayer& L = layers[a.layer];
                const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
                for(int i = 0; i < 3; i++){
                    uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride;
                    cudaLink().copyUp(L.resident[i] + (uint64_t)a.slot * span, parts[i]->data + (uint64_t)a.expert * span, (size_t)span);
                }
            }
            swapCount += arrivals.size();
            queue.eventRecord(swapDone, stream);
            settle(true);
            return 0;
        }
        pool.run((int)staged.size() * 3, [&](int job){
            int k = job / 3, i = job % 3;
            const Arrival& a = arrivals[staged[k]];
            const GpuLayer& L = layers[a.layer];
            const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
            uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride;
            uint8_t* into = ring[k / stageSlots] + (uint64_t)(k % stageSlots) * stageStride + L.piece[i];
            memcpy(into, parts[i]->data + (uint64_t)a.expert * span, (size_t)span);
        });

        int k = 0;
        for(int i = 0; i < (int)arrivals.size(); i++){
            const Arrival& a = arrivals[i];
            const GpuLayer& L = layers[a.layer];
            const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
            bool pinnedCopy = L.warmHost[0] && L.warm[a.expert] >= 0;
            for(int part = 0; part < 3; part++){
                uint64_t span = (uint64_t)parts[part]->rows * parts[part]->stride;
                const uint8_t* from = pinnedCopy
                    ? L.warmHost[part] + (uint64_t)L.warm[a.expert] * span
                    : ring[k / stageSlots] + (uint64_t)(k % stageSlots) * stageStride + L.piece[part];
                if(queue.copyUp(L.resident[part] + (uint64_t)a.slot * span, from, (size_t)span, copyStream) != 0) failed = true;
            }
            if(!pinnedCopy) k++;
        }
        queue.eventRecord(swapDone, copyStream);
        swapCount += arrivals.size();
        return (int)arrivals.size();
    }

    struct Exchange {
        int layer;
        int incoming;
        int outgoing;
        int slot;
        int warmSlot;
        int spare;
    };

    std::vector<Exchange> moving;
    std::vector<Exchange> leaving;
    void* movedEvent = nullptr;
    void* leftEvent = nullptr;
    uint64_t spareBase = 0;
    uint64_t spareStride = 0;
    int spareCount = 16;
    bool exchanging = true;
    float recentFade = 0.9f;
    uint64_t exchangeCount = 0;
    int* tableStage = nullptr;
    int exchangeBatch = 4;
    uint64_t ramHeadroom = 2ULL << 30;
    double exchangeMs = 0;

    bool exchangeReady() {
        return exchanging && spareBase && warmPerLayer > 0 && residentExperts > 0 && residentExperts < expertCount;
    }

    int exchange(int n) {
        if(!ready || !exchangeReady() || !useCuda()) return 0;
        float fade = (float)std::pow((double)recentFade, (double)std::max(1, n));
        for(float& r : recent) r *= fade;
        CudaQueue& queue = cudaQueue();
        if(!movedEvent && (queue.eventCreate(&movedEvent, 2) != 0 || queue.eventCreate(&leftEvent, 2) != 0)) return 0;
        if(arriving.size() != layers.size() * expertCount) arriving.assign(layers.size() * expertCount, 0);
        std::vector<char> dirty(layers.size(), 0);
        auto upload = [&]{
            int first = -1, last = -1;
            for(int l = firstLayer; l < endLayer; l++){
                if(!dirty[l]) continue;
                if(first < 0) first = l;
                last = l;
            }
            if(first < 0) return;
            size_t words = (size_t)layers.size() * expertCount;
            if(!tableStage){
                void* room = nullptr;
                if(cudaLink().hostAlloc(&room, 2 * words * 4, 0) != 0) return;
                pinned.push_back(room);
                tableStage = (int*)room;
            }
            int* slotsOut = tableStage;
            int* warmOut = tableStage + words;
            for(int l = first; l <= last; l++){
                memcpy(slotsOut + (size_t)l * expertCount, layers[l].slot.data(), expertCount * 4);
                memcpy(warmOut + (size_t)l * expertCount, layers[l].warm.data(), expertCount * 4);
            }
            size_t span = (size_t)(last - first + 1) * expertCount * 4;
            if(queue.copyUp(layers[first].slots, slotsOut + (size_t)first * expertCount, span, stream) != 0) failed = true;
            if(queue.copyUp(layers[first].warmSlots, warmOut + (size_t)first * expertCount, span, stream) != 0) failed = true;
        };
        auto pieces = [&](GpuLayer& L, uint64_t* spans){
            spans[0] = (uint64_t)L.expertGate.rows * L.expertGate.stride;
            spans[1] = (uint64_t)L.expertUp.rows * L.expertUp.stride;
            spans[2] = (uint64_t)L.expertDown.rows * L.expertDown.stride;
        };
        if(!leaving.empty()){
            if(queue.eventQuery(leftEvent) != 0) return 0;
            for(const Exchange& x : leaving){
                layers[x.layer].warm[x.outgoing] = x.warmSlot;
                dirty[x.layer] = 1;
            }
            leaving.clear();
        }
        if(!moving.empty()){
            if(queue.eventQuery(movedEvent) != 0){
                upload();
                return 0;
            }
            for(const Exchange& x : moving){
                GpuLayer& L = layers[x.layer];
                L.slot[x.incoming] = x.slot;
                L.warm[x.incoming] = -1;
                arriving[(size_t)x.layer * expertCount + x.incoming] = 0;
                dirty[x.layer] = 1;
            }
            upload();
            for(const Exchange& x : moving){
                GpuLayer& L = layers[x.layer];
                uint64_t spans[3];
                pieces(L, spans);
                uint64_t at = spareBase + (uint64_t)x.spare * spareStride;
                for(int p = 0; p < 3; p++){
                    if(queue.copyDown(L.warmHost[p] + (uint64_t)x.warmSlot * spans[p], at, (size_t)spans[p], copyStream) != 0) failed = true;
                    at += spans[p];
                }
            }
            queue.eventRecord(leftEvent, copyStream);
            exchangeCount += moving.size();
            leaving.swap(moving);
            moving.clear();
            return (int)leaving.size();
        }

        if(exchangeHold) return 0;
        float rate = heatTokens > 0 ? 0.1f / (float)heatTokens : 0.0f;
        bool heated = heat.size() == recent.size();
        std::vector<std::array<float, 4>> picks;
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            const float* seen = recent.data() + (size_t)l * expertCount;
            const uint64_t* use = heated ? heat.data() + (size_t)l * expertCount : nullptr;
            int hot = -1, cold = -1;
            float hotScore = -1.0f, coldScore = 1e30f;
            for(int e = 0; e < expertCount; e++){
                float v = seen[e] + (use ? (float)use[e] * rate : 0.0f);
                if(L.slot[e] >= 0){
                    if(L.slot[e] < fixedExperts && v < coldScore){
                        coldScore = v;
                        cold = e;
                    }
                    continue;
                }
                if(L.warm[e] < 0 || arriving[(size_t)l * expertCount + e]) continue;
                if(v > hotScore){
                    hotScore = v;
                    hot = e;
                }
            }
            if(hot >= 0 && cold >= 0 && hotScore - coldScore > 1.0f) picks.push_back({hotScore - coldScore, (float)l, (float)hot, (float)cold});
        }
        std::sort(picks.begin(), picks.end(), [](const std::array<float, 4>& a, const std::array<float, 4>& b){ return a[0] > b[0]; });
        for(const auto& pick : picks){
            if((int)moving.size() >= std::min(spareCount, exchangeBatch)) break;
            int layer = (int)pick[1], incoming = (int)pick[2], outgoing = (int)pick[3];
            GpuLayer& L = layers[layer];
            moving.push_back({layer, incoming, outgoing, L.slot[outgoing], L.warm[incoming], (int)moving.size()});
            L.slot[outgoing] = -1;
            arriving[(size_t)layer * expertCount + incoming] = 1;
            dirty[layer] = 1;
        }
        upload();
        for(const Exchange& x : moving){
            GpuLayer& L = layers[x.layer];
            uint64_t spans[3];
            pieces(L, spans);
            uint64_t at = spareBase + (uint64_t)x.spare * spareStride;
            for(int p = 0; p < 3; p++){
                uint64_t place = L.resident[p] + (uint64_t)x.slot * spans[p];
                if(queue.copyAcross(at, place, (size_t)spans[p], copyStream) != 0) failed = true;
                if(queue.copyUp(place, L.warmHost[p] + (uint64_t)x.warmSlot * spans[p], (size_t)spans[p], copyStream) != 0) failed = true;
                at += spans[p];
            }
        }
        if(!moving.empty()) queue.eventRecord(movedEvent, copyStream);
        return (int)moving.size();
    }

    void reset() {
        feedWaiting = false;
        settleTail();
        tokens.clear();
        draftPendingAt = -1;
        draftConsumed = true;
        if(gemma){
            gemmaBuffers.ringTop = -1;
            return;
        }
        if(hy4){
            hyBuffers.ringTop = -1;
            return;
        }
        if(!ready || !useCuda()) return;
        if(remote) remote->state(STATE_RESET, 0);
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            if(L.attention) continue;
            cudaLink().memset(L.state, 0, (size_t)valueHeads * ssmSize * ssmSize * 4);
            cudaLink().memset(L.convHistory, 0, (size_t)3 * convWidth * 4);
        }
        cudaLink().memset(pleHistory, 0, (size_t)pleTaps * hcWidth * 4);
        cudaLink().sync();
    }

    void pack(uint64_t input, int count, int slot) {
        bool ready = xPacked && input == x && slot == 0;
        if(slot == 0) xPacked = false;
        if(ready) return;
        if(!fine){
            if(!packInput(module, input, count, packs[slot])) failed = true;
            return;
        }
        PackedInput& a = packs[slot];
        PackedInput& b = fines[slot];
        go("quantizeFine", (unsigned int)((count / 32 + 7) / 8), 1, 32, 8, input, count, a.values, a.scales, a.sums, b.values, b.scales, b.sums);
    }

    void swigluPack(uint64_t gate, uint64_t up, uint64_t out, int count, int slot) {
        if(!fine || !fusedMix || count % 32 != 0){
            go("swiglu", (unsigned int)((count + 255) / 256), 1, 256, 1, gate, up, out, count);
            pack(out, count, slot);
            return;
        }
        PackedInput& a = packs[slot];
        PackedInput& b = fines[slot];
        go("swigluFine", (unsigned int)((count / 32 + 7) / 8), 1, 32, 8, gate, up, out, count, a.values, a.scales, a.sums, b.values, b.scales, b.sums);
    }

    std::vector<const GpuMatrix*> denseOf(const GpuLayer& L) const {
        return L.attention
            ? std::vector<const GpuMatrix*>{&L.query, &L.key, &L.value, &L.attnOut, &L.sharedGate, &L.sharedUp, &L.sharedDown}
            : std::vector<const GpuMatrix*>{&L.qkv, &L.gate, &L.ssmOut, &L.sharedGate, &L.sharedUp, &L.sharedDown};
    }

    uint64_t halfBytes(const GpuMatrix& m) const {
        bool known = m.type == 7 || m.type == 8 || (m.type >= 12 && m.type <= 14) || (m.type >= 16 && m.type <= 18) || (m.type >= 20 && m.type <= 23) || m.type == 42;
        return m.data && known && m.cols % 64 == 0
            ? (2ULL * m.rows * m.cols + 255) / 256 * 256
            : 0;
    }

    std::vector<const GpuMatrix*> mixOf(const GpuLayer& L) const {
        std::vector<const GpuMatrix*> list;
        if(!halfMix || hcCount != 4) return list;
        for(int i = 0; i < 2; i++){
            for(const GpuMatrix* m : {&L.hcDown[i], &L.hcInject[i]}){
                if(m->data && m->type == 30 && m->cols % 2 == 0 && m->stride == 2ULL * m->cols) list.push_back(m);
            }
        }
        if(!halfGates || L.attention) return list;
        for(const GpuMatrix* m : {&L.alpha, &L.beta}){
            if(m->data && m->type == 30 && m->cols % 2 == 0 && m->stride == 2ULL * m->cols) list.push_back(m);
        }
        return list;
    }

    bool halvedAll(std::initializer_list<const GpuMatrix*> list) const {
        for(const GpuMatrix* m : list){
            if(!halved.count(m)) return false;
        }
        return true;
    }

    void halve(const GpuLayer& L) {
        halved.clear();
        if(!blas) return;
        uint64_t at = 0;
        for(const GpuMatrix* m : denseOf(L)){
            uint64_t bytes = halfBytes(*m);
            if(bytes == 0 || at + bytes > halfRoom) continue;
            go("halfUnpack", (unsigned int)(((uint64_t)m->rows * (m->cols / 32) + 255) / 256), 1, 256, 1, (int)m->type, m->data, m->stride, halfWeights + at, m->rows, m->cols, tables);
            halved[m] = halfWeights + at;
            at += bytes;
        }
        for(const GpuMatrix* m : mixOf(L)){
            uint64_t bytes = (2ULL * m->rows * m->cols + 255) / 256 * 256;
            if(at + bytes > halfRoom) continue;
            uint64_t pairs = (uint64_t)m->rows * m->cols / 2;
            go("bf16ToHalf", (unsigned int)((pairs + 255) / 256), 1, 256, 1, m->data, halfWeights + at, pairs);
            halved[m] = halfWeights + at;
            at += bytes;
        }
    }

    void matrix(const char* name, int rows, int n, int cols, uint64_t weights, int kind, int weightStride, uint64_t input, float keep, uint64_t output, int outputStride) {
        if(profile){
            if(markUsed == marks.size()){
                std::pair<void*, void*> pair = {nullptr, nullptr};
                cudaQueue().eventCreate(&pair.first, 0);
                cudaQueue().eventCreate(&pair.second, 0);
                marks.push_back({"", pair});
            }
            marks[markUsed].first = profileShapes
                ? phase + " " + name + " " + std::to_string(rows) + "x" + std::to_string(n) + "x" + std::to_string(cols)
                : phase + " " + name;
            cudaQueue().eventRecord(marks[markUsed].second.first, stream);
        }
        float one = 1.0f;
        int code = blasLink().gemm(blas, 1, 0, rows, n, cols, &one, (const void*)weights, kind, weightStride, (const void*)input, kind, cols, &keep, (void*)output, 0, outputStride, 68, -1);
        if(code != 0 && !failed) launchProblem = std::string(name) + " (matrix library " + std::to_string(code) + ")";
        if(code != 0) failed = true;
        if(profile) cudaQueue().eventRecord(marks[markUsed++].second.second, stream);
    }

    void plain(const GpuMatrix& m, uint64_t input, int inputStride, uint64_t output, int outputStride, int n, int mode, bool fresh = true) {
        if(blas && n > 16 && m.type == 30 && m.cols % 2 == 0 && inputStride % 2 == 0 && (uint64_t)n * m.cols * 2 <= wideRoom){
            uint64_t pairs = (uint64_t)n * (m.cols / 2);
            if(fresh) go("splitBf16", (unsigned int)((pairs + 255) / 256), 1, 256, 1, input, inputStride, wideRows, wideRows + wideRoom, n, m.cols);
            matrix("bf16Matrix", m.rows, n, m.cols, m.data, 14, (int)(m.stride / 2), wideRows, 0.0f, output, outputStride);
            if(m.cols > 512) matrix("bf16Matrix", m.rows, n, m.cols, m.data, 14, (int)(m.stride / 2), wideRows + wideRoom, 1.0f, output, outputStride);
            if(mode == 1) go("gateRows", (unsigned int)(((uint64_t)n * m.rows + 255) / 256), 1, 256, 1, output, outputStride, m.rows, n);
            return;
        }
        if(n > 16 && (tensorPlain || (legacy && tilePlain)) && m.type == 30 && m.cols % 32 == 0){
            int rowTiles = (m.rows + 63) / 64;
            int tokenTiles = (n + 63) / 64;
            int splits = std::max(1, std::min(m.cols / 256, 240 / (rowTiles * tokenTiles)));
            while(splits > 1 && (uint64_t)splits * n * m.rows * 4 > (4ULL << 20)) splits--;
            unsigned int spread = (unsigned int)((m.rows * n + 255) / 256);
            uint64_t target = splits > 1
                ? splitScratch
                : output;
            int stride = splits > 1
                ? m.rows
                : outputStride;
            go(legacy ? "plainTile" : "plainTensor", (unsigned int)rowTiles, (unsigned int)(tokenTiles * splits), legacy ? 256 : 128, 1, m.data, m.stride, input, inputStride, target, stride, m.rows, m.cols, n, splits, mode);
            if(splits > 1) go("mergeSplits", spread, 1, 256, 1, splitScratch, output, outputStride, m.rows, n, splits, mode);
            return;
        }
        int split = m.rows >= 8192
            ? 1
            : m.rows >= 2048
                ? 2
                : m.rows >= 512
                    ? 4
                    : 8;
        int perBlock = 8 / split;
        std::string name = n >= 2 && n <= fewRows
            ? "plainRows" + std::to_string(n)
            : "plainRows";
        go(name.c_str(), (unsigned int)((m.rows + perBlock - 1) / perBlock), 1, 256, 1, (int)m.type, m.data, m.stride, input, inputStride, output, outputStride, m.rows, m.cols, n, split, mode);
    }

    void project(const GpuMatrix& m, uint64_t input, int slot, uint64_t output, int n) {
        auto wide = halved.find(&m);
        if((m.type == 30 || m.type == 1) && !(wide != halved.end() && (uint64_t)n * m.cols * 2 <= wideRoom)){
            plain(m, input, m.cols, output, m.rows, n, 0);
            return;
        }
        if(wide != halved.end() && (uint64_t)n * m.cols * 2 <= wideRoom){
            uint64_t pairs = (uint64_t)n * m.cols / 2;
            go("halfInput", (unsigned int)((pairs + 255) / 256), 1, 256, 1, input, wideRows, pairs);
            matrix("halfMatrix", m.rows, n, m.cols, wide->second, 2, m.cols, wideRows, 0.0f, output, m.rows);
            return;
        }
        if(n == 1 && exact){
            if(!launchRows(module, (int)m.type, m.data, m.stride, input, output, m.rows, m.cols, tables)) failed = true;
            return;
        }
        if(n == 1 && !fine){
            if(!launchRowsQ8(module, (int)m.type, m.data, m.stride, input, packs[slot], output, m.rows, m.cols, tables)) failed = true;
            return;
        }
        uint64_t none = 0;
        PackedInput& a = packs[slot];
        PackedInput b = fine && residual ? fines[slot] : PackedInput();
        bool known = m.type == 12 || m.type == 13 || m.type == 14 || m.type == 20 || m.type == 21 || m.type == 23 || m.type == 42 || m.type == 7 || m.type == 8;
        // de 2 a 16 tokens con los nucleos tensoriales: cada pedazo de tokens se lee una vez para 16 filas de pesos
        // Q8_0 no entra: en matrices de 6 MB va a 333-367 GB/s contra 461-599 de dos filas por warp
        if(fewMma && known && m.type != 8 && n >= fewMmaFrom && n <= 16 && fine && !exact && !legacy && !amdCards && m.cols % 32 == 0){
            std::string name = "rowsMma" + std::to_string(m.type) + "t" + std::to_string((n + 7) / 8);
            go(name.c_str(), (unsigned int)((m.rows + 15) / 16), 1, 256, 1, m.data, m.stride, a.values, a.scales, a.sums, b.values, b.scales, b.sums, output, m.rows, m.cols, n, tables);
            return;
        }
        if(typedRows && known && n <= fewRows && fine && !exact && (directRows || (pairRows && n >= 3) || m.stride <= 5104)){
            int stage = (int)((m.stride + 31) / 16 * 16);
            // medido en la 3090 con las matrices de un 27B denso: Q4_K y Q5_K van mejor de a una fila hasta 5 tokens
            // (4 tokens 644 contra 500 GB/s) y Q6_K de a dos desde 2 tokens (555 contra 515)
            bool measured = !legacy && !amdCards;
            bool paired = pairRows && (!measured ? n >= 3 : m.type == 12 || m.type == 13 ? n > 5 : m.type == 14 || m.type == 8 ? n >= 2 : n >= 3);
            // 3 filas van con el kernel de 4 (Q4_K 562 contra 654 GB/s) y las salidas tienen lugar para la cuarta que nadie lee
            int counted = padThree && n == 3 && (m.type == 12 || m.type == 13 || m.type == 8) ? 4 : n;
            std::string name = (paired ? "rowsPair" : directRows ? "rowsDirect" : "rowsType") + std::to_string(m.type) + "x" + std::to_string(counted);
            unsigned int blocks = paired
                ? (unsigned int)((m.rows + 15) / 16)
                : (unsigned int)((m.rows + 7) / 8);
            goShared(directRows || paired ? 0 : 8 * stage, name.c_str(), blocks, 1, 256, 1, m.data, m.stride, a.values, a.scales, a.sums, b.values, b.scales, b.sums, output, m.rows, m.cols, stage, tables);
            return;
        }
        if(n == 1 && m.stride <= 5104){
            int stage = (int)((m.stride + 31) / 16 * 16);
            goShared(8 * stage, "rowsStaged", (unsigned int)((m.rows + 7) / 8), 1, 256, 1, (int)m.type, m.data, m.stride, none, none, none, 0, a.values, a.scales, a.sums, b.values, b.scales, b.sums, output, m.rows, m.cols, stage, tables);
            return;
        }
        if(n <= 16 && fine && !exact && m.stride <= 5104){
            int stage = (int)((m.stride + 31) / 16 * 16);
            const char* name = n == 2
                ? "rowsBatch2"
                : n == 3
                    ? "rowsBatch3"
                    : n == 4
                        ? "rowsBatch4"
                        : "rowsBatch";
            goShared(8 * stage, name, (unsigned int)((m.rows + 7) / 8), 1, 256, 1, (int)m.type, m.data, m.stride, a.values, a.scales, a.sums, b.values, b.scales, b.sums, output, m.rows, m.cols, n, stage, tables);
            return;
        }
        if(wideTiles && m.rows >= 4096){
            int wide = (n + 127) / 128;
            go("mmaWide", (unsigned int)((m.rows + 63) / 64), (unsigned int)wide, 256, 1, (int)m.type, m.data, m.stride, none, none, none, none, none, a.values, a.scales, a.sums, b.values, b.scales, b.sums, output, m.rows, m.cols, n, wide, none, none, none, tables);
            return;
        }
        int width = tileWidth(n >= 128);
        int tiles = (n + width - 1) / width;
        PackedInput e = batchFine ? b : PackedInput();
        go(tiled(m.type, n >= 128, e.values).c_str(), (unsigned int)((m.rows + rowTile(m.type, n >= 128, e.values) - 1) / rowTile(m.type, n >= 128, e.values)), (unsigned int)tiles, width == 128 ? 256 : 128, 1, (int)m.type, m.data, m.stride, none, none, none, none, none, a.values, a.scales, a.sums, e.values, e.scales, e.sums, output, m.rows, m.cols, n, tiles, none, none, none, tables);
    }

    void projectMany(const std::vector<std::pair<const GpuMatrix*, uint64_t>>& list, uint64_t input, int slot, int n) {
        std::vector<std::pair<const GpuMatrix*, uint64_t>> grouped, wide;
        int converted = 0;
        for(const auto& item : list){
            const GpuMatrix& w = *item.first;
            bool fits = groupedRows && n == 1 && fine && !exact && w.type != 30 && w.type != 1 && w.stride <= 5104 && grouped.size() < 6 && (grouped.empty() || grouped[0].first->cols == w.cols);
            bool paired = fusedMix && n <= fewRows && w.type == 30 && w.cols % 8 == 0;
            auto half = halved.find(&w);
            bool shared = !fits && !paired && half != halved.end() && (uint64_t)n * w.cols * 2 <= wideRoom;
            if(fits) grouped.push_back(item);
            else if(paired) wide.push_back(item);
            else if(shared && converted == w.cols) matrix("halfMatrix", w.rows, n, w.cols, half->second, 2, w.cols, wideRows, 0.0f, item.second, w.rows);
            else project(w, input, slot, item.second, n);
            if(fits || paired) continue;
            converted = shared
                ? w.cols
                : 0;
        }
        if(wide.size() == 2 && wide[0].first->cols == wide[1].first->cols){
            const GpuMatrix& a = *wide[0].first;
            const GpuMatrix& b = *wide[1].first;
            std::string pairName = "plainPair" + std::to_string(n);
            go(pairName.c_str(), (unsigned int)(a.rows + b.rows), 1, 256, 1, a.data, b.data, a.stride, b.stride, a.rows, a.rows + b.rows, input, a.cols, wide[0].second, a.rows, wide[1].second, b.rows, a.cols, 0);
        } else {
            for(const auto& item : wide) project(*item.first, input, slot, item.second, n);
        }
        if(grouped.size() < 2){
            for(const auto& item : grouped) project(*item.first, input, slot, item.second, n);
            return;
        }
        int cols = grouped[0].first->cols;
        uint64_t widest = 0;
        RowSet set = {};
        set.count = (int)grouped.size();
        int blocks = 0;
        for(int i = 0; i < set.count; i++){
            const GpuMatrix& w = *grouped[i].first;
            set.type[i] = (int)w.type;
            set.rows[i] = w.rows;
            set.start[i] = blocks;
            set.weights[i] = w.data;
            set.stride[i] = w.stride;
            set.output[i] = grouped[i].second;
            widest = std::max<uint64_t>(widest, w.stride);
            blocks += (w.rows + 7) / 8;
        }
        set.start[set.count] = blocks;
        int stage = (int)((widest + 31) / 16 * 16);
        PackedInput& a = packs[slot];
        PackedInput b = residual ? fines[slot] : PackedInput();
        goShared(8 * stage, "rowsMulti", (unsigned int)blocks, 1, 256, 1, set, a.values, a.scales, a.sums, b.values, b.scales, b.sums, cols, stage, tables);
    }

    void mix(uint64_t norm, const GpuMatrix& down, const GpuMatrix& up, const GpuMatrix* inject, uint64_t R, int n, uint64_t out, uint64_t injected, uint64_t added = 0) {
        if(hcCount == 1 && dense && fine && fusedMix && hidden % 32 == 0 && out == x){
            PackedInput& a = packs[0];
            PackedInput b = residual ? fines[0] : PackedInput();
            go("addNormPack", (unsigned int)n, 1, 32, 32, R, added, norm, out, a.values, a.scales, a.sums, b.values, b.scales, b.sums);
            xPacked = true;
            return;
        }
        bool joined = added && fusedNorm && hcCount > 1 && hidden % 128 == 0 && blas && n > fewRows && n > 16 && hcWidth % 2 == 0 && (uint64_t)n * hcWidth * 2 <= wideRoom && down.type == 30 && inject && inject->type == 30 && halved.count(&down) && halved.count(inject);
        if(added && !joined) go("writeBack", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, R, added, injected, n);
        if(hcCount == 1){
            go("streamNorm", (unsigned int)n, 1, 256, 1, R, norm, out);
            return;
        }
        bool halves = blas && n > fewRows && n > 16 && hcWidth % 2 == 0 && (uint64_t)n * hcWidth * 2 <= wideRoom && down.type == 30 && inject && inject->type == 30;
        bool sixteen = halves && halved.count(&down) && halved.count(inject);
        if(sixteen){
            bool fusedUp = mixUp && !legacy && !amdCards && hcCount == 4 && hcRank % 32 == 0 && hidden % 64 == 0 && up.type == 30 && up.rows == hcWidth && up.cols == hcRank && up.stride == 2ULL * up.cols;
            if(fusedNorm && hidden % 128 == 0) go("writeNormHalf", (unsigned int)((n * hcCount + 7) / 8), 1, 256, 1, R, joined ? added : 0ULL, injected, norm, fusedUp ? 0ULL : xn, wideRows, n * hcCount);
            else go("streamNormHalf", (unsigned int)(n * hcCount), 1, 256, 1, R, norm, xn, wideRows);
            bool together = joinInject && down.rows == hcRank && inject->rows == hcCount && halved[inject] == halved[&down] + 2ULL * down.rows * hcWidth && (uint64_t)n * (hcRank + hcCount) * 4 <= (4ULL << 20);
            if(together){
                matrix("halfMatrix", hcRank + hcCount, n, hcWidth, halved[&down], 2, hcWidth, wideRows, 0.0f, splitScratch, hcRank + hcCount);
                go("gateTake", (unsigned int)(((uint64_t)n * (hcRank + hcCount) + 255) / 256), 1, 256, 1, splitScratch, lo, injected, hcRank, hcCount, n);
            } else {
                matrix("halfMatrix", down.rows, n, hcWidth, halved[&down], 2, hcWidth, wideRows, 0.0f, lo, hcRank);
                go("gateRows", (unsigned int)(((uint64_t)n * hcRank + 255) / 256), 1, 256, 1, lo, hcRank, hcRank, n);
                matrix("halfMatrix", inject->rows, n, hcWidth, halved[inject], 2, hcWidth, wideRows, 0.0f, injected, hcCount);
            }
            if(fusedUp){
                go("hcMixUp", (unsigned int)(hidden / 64), (unsigned int)((n + 63) / 64), 256, 1, up.data, up.stride, lo, wideRows, out, n, hcRank);
                return;
            }
            plain(up, lo, hcRank, mixGate, hcWidth, n, 0);
            go("mixStreams", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, xn, mixGate, out, n);
            return;
        }
        if(halves) go("streamNormSplit", (unsigned int)(n * hcCount), 1, 256, 1, R, norm, xn, wideRows, wideRows + wideRoom);
        else go("streamNorm", (unsigned int)(n * hcCount), 1, 256, 1, R, norm, xn);
        bool fused = fusedMix && n <= fewRows && hcCount == 4 && hcRank >= 32 && hcRank % 8 == 0 && hcWidth % 8 == 0 && inject
            && down.type == 30 && up.type == 30 && inject->type == 30 && down.rows == hcRank && inject->rows == hcCount && up.rows == hcWidth;
        if(fused){
            std::string pairName = "plainPair" + std::to_string(n);
            std::string mixName = "hcUpMix" + std::to_string(n);
            go(pairName.c_str(), (unsigned int)(hcRank + hcCount), 1, 256, 1, down.data, inject->data, down.stride, inject->stride, hcRank, hcRank + hcCount, xn, hcWidth, lo, hcRank, injected, hcCount, hcWidth, 1);
            go(mixName.c_str(), (unsigned int)((hidden + 7) / 8), 1, 256, 1, up.data, up.stride, lo, xn, out);
            return;
        }
        plain(down, xn, hcWidth, lo, hcRank, n, 1, !halves);
        if(inject) plain(*inject, xn, hcWidth, injected, hcCount, n, 0, !halves);
        plain(up, lo, hcRank, mixGate, hcWidth, n, 0);
        go("mixStreams", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, xn, mixGate, out, n);
    }

    void embedCell(int id, float* out) {
        if(id >= 0){
            dequantRow(embed.type, embed.data + (uint64_t)(id < embed.rows ? id : 0) * embed.stride, out, hidden);
            return;
        }
        int code = -1 - id, slot = code / 32768, k = code % 32768;
        if(slot < (int)pictures.size() && (size_t)(k + 1) * hidden <= pictures[slot].rows.size()){
            memcpy(out, pictures[slot].rows.data() + (size_t)k * hidden, (size_t)hidden * 4);
            return;
        }
        memset(out, 0, (size_t)hidden * 4);
    }

    int cellToken(int id) {
        return id >= 0 || imageToken < 0 ? id : imageToken;
    }

    void placePictures(const std::vector<int>& ids, const std::vector<int>& positions) {
        if(pictures.empty() || !placesOnGpu || ids.empty()) return;
        size_t from = (size_t)positions.front(), end = from + ids.size();
        size_t start = from > 4 ? from - 4 : 0;
        hostPlaces.clear();
        int shift = 0, base = 0;
        size_t after = start;
        bool listed = false;
        for(size_t c = 0; c < end; c++){
            int id = c < from ? (c < tokens.size() ? tokens[c] : 0) : ids[c - from];
            int t = (int)c - shift, h = t, w = t;
            if(id < 0){
                int code = -1 - id, slot = code / 32768, k = code % 32768;
                int high = slot < (int)pictures.size() ? std::max(1, pictures[slot].high) : 1;
                int wide = slot < (int)pictures.size() ? std::max(1, pictures[slot].wide) : 1;
                if(k == 0) base = (int)c - shift;
                t = base;
                h = base + k / wide;
                w = base + k % wide;
                if(k == high * wide - 1) shift += high * wide - std::max(high, wide);
                if(c >= start){
                    listed = true;
                    after = c + 1;
                }
            }
            if(c < start) continue;
            hostPlaces.push_back(t);
            hostPlaces.push_back(h);
            hostPlaces.push_back(w);
        }
        if(!listed && ropeTailFrom <= start && ropeTailShift == shift) return;
        waitStream();
        if(cudaLink().copyUp(placesOnGpu, hostPlaces.data(), hostPlaces.size() * 4) != 0){
            failed = true;
            return;
        }
        go("placeRope", (unsigned int)(contextLimit - start), 1, 32, 1, ropeCos, ropeSin, placesOnGpu, (int)start, (int)(end - start), shift, ropeTheta, ropeSpans[1], ropeSpans[2]);
        ropeTailFrom = listed ? after : start;
        ropeTailShift = shift;
    }

    std::string findReader(const std::string& path) {
        auto named = file.meta.find("general.basename");
        std::string base = named == file.meta.end() ? "" : named->second.text;
        std::filesystem::path folder = std::filesystem::path(path).parent_path();
        for(const std::filesystem::path& place : {folder, folder.parent_path()}){
            std::error_code fault;
            for(const auto& entry : std::filesystem::directory_iterator(place, fault)){
                std::string name = entry.path().filename().string();
                if(name.rfind("mmproj", 0) != 0 || entry.path().extension() != ".gguf") continue;
                if(place != folder && !base.empty() && name.find(base) == std::string::npos){
                    ModelFile other;
                    std::string why;
                    if(!openModel(entry.path().string(), other, why)) continue;
                    auto same = other.meta.find("general.basename");
                    if(same == other.meta.end() || same->second.text != base) continue;
                }
                return entry.path().string();
            }
        }
        return "";
    }

    void openVision(const std::string& path) {
        std::string reader = findReader(path);
        if(reader.empty()){
            visionProblem = "Vision is on but the picture reader file (mmproj) is not next to the model, get it in Models with Get picture reader";
            return;
        }
        CudaLink& cu = cudaLink();
        int device = cu.device;
        for(int d = 0; d < cu.count && device == cu.device; d++){
            int dev = -1, major = 0, minor = 0;
            if(visionGpu >= 0 && d != visionGpu) continue;
            if(!cu.deviceGet || cu.deviceGet(&dev, d) != 0 || dev == cu.device) continue;
            cu.attribute(&major, 75, dev);
            cu.attribute(&minor, 76, dev);
            if(major * 10 + minor >= 61) device = dev;
        }
        seeing = seer.open(reader, device, visionProblem);
        if(!seeing && device != cu.device){
            // si la segunda tarjeta no lo toma cae en la principal y le quita memoria a los expertos
            std::cout << "vision: the second card did not take the picture reader (" << visionProblem << "), it goes to the main card" << std::endl;
            visionProblem.clear();
            device = cu.device;
            seeing = seer.open(reader, device, visionProblem);
        }
        visionCard = device == cu.device ? "the main card" : "the second card";
        cu.setCurrent(cu.context);
    }

    int seePicture(const std::string& bytes, std::string& error) {
        uint64_t hash = 1469598103934665603ULL;
        for(unsigned char c : bytes){
            hash ^= c;
            hash *= 1099511628211ULL;
        }
        auto found = pictureSlots.find(hash);
        int slot = found == pictureSlots.end() ? (int)pictures.size() : found->second;
        if(found != pictureSlots.end() && !pictures[slot].rows.empty()){
            pictures[slot].used = ++pictureClock;
            return slot;
        }
        if(slot >= 65536){
            error = "too many different pictures since the model loaded, load the model again";
            return -1;
        }
        Picture picture;
        if(!readPicture(bytes, picture, error)) return -1;
        SeenPicture seen;
        seen.hash = hash;
        bool ok = seer.see(picture, seen.rows, seen.high, seen.wide, error);
        cudaLink().setCurrent(cudaLink().context);
        if(!ok) return -1;
        seen.used = ++pictureClock;
        if(slot == (int)pictures.size()) pictures.push_back(std::move(seen));
        else pictures[slot] = std::move(seen);
        pictureSlots[hash] = slot;
        std::vector<std::pair<uint64_t, int>> kept;
        for(int i = 0; i < (int)pictures.size(); i++) if(!pictures[i].rows.empty()) kept.push_back({pictures[i].used, i});
        std::sort(kept.begin(), kept.end());
        for(size_t i = 0; i + 16 < kept.size(); i++){
            pictures[kept[i].second].rows.clear();
            pictures[kept[i].second].rows.shrink_to_fit();
        }
        return slot;
    }

    // la fila de la tabla por capa sale del token y de los dos anteriores
    template<class At> uint64_t pleRowOf(size_t s, int h, At tokenAt) {
        uint64_t ctx[3] = {(uint64_t)cellToken(tokenAt(s)), 0, 0};
        bool cut = false;
        for(int i = 1; i <= 2; i++){
            bool missing = s < (size_t)i;
            int previous = missing ? endToken : cellToken(tokenAt(s - i));
            cut = cut || missing || previous == endToken;
            ctx[i] = cut ? (uint64_t)endToken : (uint64_t)previous;
        }
        uint64_t mixed = ctx[0] * pleMultipliers[0] ^ ctx[1] * pleMultipliers[1];
        if(h >= 8) mixed ^= ctx[2] * pleMultipliers[2];
        return mixed % pleVocab[h] + pleOffset[h];
    }

    void pleRows(int n, size_t first) {
        auto clock = std::chrono::steady_clock::now();
        if(pleRowIds.size() < (size_t)n * pleHeads) pleRowIds.resize((size_t)n * pleHeads);
        pool.run(n * pleHeads, [&](int job){
            int t = job / pleHeads, h = job % pleHeads;
            uint64_t row = pleRowOf(first + t, h, [&](size_t s){ return tokens[s]; });
            pleRowIds[job] = row;
            dequantRow(pleTable.type, pleTable.data + row * pleTable.stride, hostPle + (size_t)t * hidden + h * pleRowWidth, pleRowWidth);
        });
        pleMs += since(clock);
    }

    void perLayer(int n) {
        go("copyWords", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, mappedPle, pleEmbed, n * hidden);
        plain(pleKey, pleEmbed, hidden, pleKeyOut, hcWidth, n, 0);
        plain(pleValue, pleEmbed, hidden, pleValueOut, hidden, n, 0);
        go("pleGate", hcCount, (unsigned int)n, 256, 1, pleKeyOut, pleValueOut, stack, pleNormKey, pleNormQuery, pleNormConv, gated, normd);
        if(recording && replayStore) cudaQueue().copyAcross(replayStore + 4ULL * replayStride() * layers.size(), normd, (size_t)n * hcWidth * 4, stream);
        go("pleConv", (unsigned int)((n * hcWidth + 255) / 256), 1, 256, 1, gated, normd, pleHistory, pleConv, stack, n);
        if(!leaveState) go("pleHistory", hcWidth / 256, 1, 256, 1, normd, pleHistory, n);
    }

    void deltaNet(GpuLayer& L, int n) {
        if(!halvedAll({&L.qkv, &L.gate})) pack(x, n * hidden, 0);
        projectMany({{&L.qkv, qkv}, {&L.gate, z}, {&L.alpha, alphaOut}, {&L.beta, betaOut}}, x, 0, n);
        go("gdnConv", 2 * keyHeads + valueHeads, (unsigned int)n, ssmSize, 1, qkv, L.convHistory, L.convWeight, u);
        if(!leaveState) go("gdnHistory", convWidth / 256, 1, 256, 1, qkv, L.convHistory, n);
        if(groupScan && n > 16 && valueHeads == 3 * keyHeads && ssmSize == 128) go("gdnScanGroup", ssmSize / 16, keyHeads, 64, 1, u, alphaOut, betaOut, L.dtBias, L.decay, L.state, o, n, leaveState ? 0 : 1);
        else go("gdnScan", ssmSize / 32, valueHeads, 128, 1, u, alphaOut, betaOut, L.dtBias, L.decay, L.state, o, n, leaveState ? 0 : 1);
        if(recording && replayStore){
            CudaQueue& queue = cudaQueue();
            uint64_t at = replayStore + 4ULL * replayStride() * (uint64_t)(&L - layers.data());
            queue.copyAcross(at, qkv, (size_t)n * convWidth * 4, stream);
            queue.copyAcross(at + 4ULL * relayLimit * convWidth, u, (size_t)n * convWidth * 4, stream);
            queue.copyAcross(at + 8ULL * relayLimit * convWidth, alphaOut, (size_t)n * valueHeads * 4, stream);
            queue.copyAcross(at + 8ULL * relayLimit * convWidth + 4ULL * relayLimit * valueHeads, betaOut, (size_t)n * valueHeads * 4, stream);
        }
        go("gdnNorm", valueHeads, (unsigned int)n, ssmSize, 1, o, z, L.ssmNorm);
        if(!halvedAll({&L.ssmOut})) pack(o, n * innerWidth, 0);
        project(L.ssmOut, o, 0, y, n);
    }

    void attend(GpuLayer& L, int n, int top, bool keysOnly = false) {
        if(!halvedAll({&L.query, &L.key, &L.value})) pack(x, n * hidden, 0);
        if(keysOnly && indexHeads > 0) projectMany({{&L.key, k}, {&L.value, v}, {&L.indexKey, ik}}, x, 0, n);
        else if(keysOnly) projectMany({{&L.key, k}, {&L.value, v}}, x, 0, n);
        else if(indexHeads > 0) projectMany({{&L.query, q}, {&L.key, k}, {&L.value, v}, {&L.indexQuery, iq}, {&L.indexKey, ik}}, x, 0, n);
        else projectMany({{&L.query, q}, {&L.key, k}, {&L.value, v}}, x, 0, n);
        uint64_t keyScales = L.keys + (uint64_t)contextLimit * kvHeads * headWidth;
        uint64_t valueScales = L.values + (uint64_t)contextLimit * kvHeads * headWidth;
        uint64_t indexScales = L.indexRaw + (uint64_t)contextLimit * indexWidth;
        go("attnPrepare", queryHeads + kvHeads + indexHeads + 1, (unsigned int)n, 256, 1, q, k, v, iq, ik, L.queryNorm, L.keyNorm, L.indexQueryNorm, L.keys, keyScales, L.values, valueScales, L.indexRaw, indexScales, ropeCos, ropeSin, positionsOnGpu);
        if(indexHeads > 0) go("poolIndex", (unsigned int)n, 1, indexWidth, 1, L.indexRaw, indexScales, L.pooled, L.indexKeyNorm, ropeCos, ropeSin, positionsOnGpu);
        if(keysOnly) return;
        if(fixed ? span > 0 : top + 1 > budget){
            int blocks = fixed
                ? span / 4
                : (top + 1) / 4;
            if(tileScores && n >= 32 && indexHeads == 4 && indexWidth == 128) go("indexTiles", (unsigned int)((blocks + 127) / 128), (unsigned int)((n + 31) / 32), 256, 1, iq, L.pooled, scores, positionsOnGpu, budget, scoreStride, n);
            else go("indexScores", (unsigned int)((blocks + 7) / 8), (unsigned int)n, 32, 8, iq, L.pooled, scores, positionsOnGpu, budget, scoreStride);
            go("selectCells", (unsigned int)n, 1, 1024, 1, scores, cells, positionsOnGpu, budget, scoreStride);
        }
        bool dense = fixed
            ? span == 0
            : top + 1 <= budget;
        int tileTokens = std::max(1, 16 / (queryHeads / kvHeads));
        int tiles = (n + tileTokens - 1) / tileTokens;
        int stretch = indexHeads == 0 && dense && n <= 4 * tileTokens ? reachStep(top) : 0;
        if(stretch > 0){
            int room = (int)((4ULL << 20) / ((uint64_t)n * queryHeads * 258 * 4));
            int want = stretch == 1 ? 256 : stretch == 2 ? 384 : 512;
            int splits = std::max(2, std::min(room, (want + kvHeads * tiles - 1) / (kvHeads * tiles)));
            go("attentionSpread", kvHeads, (unsigned int)(tiles * splits), 256, 1, q, L.keys, keyScales, L.values, valueScales, positionsOnGpu, splitScratch, n, splits);
            go("attentionMerge", queryHeads, (unsigned int)n, 256, 1, q, splitScratch, o, splits);
        }
        else if(n > 1 && dense){
            int rowsPer = kvHeads > 0 ? queryHeads / kvHeads : 16;
            int pack = packedAttention && rowsPer < 16 ? 16 / rowsPer : 1;
            bool tensor = intAttention && mmaAttention && !legacy && !amdCards && !fixed && n >= 32 && kvHeads > 0 && queryHeads <= 16 * kvHeads && headWidth == 256;
            bool volta = voltaAttention && !fixed && n >= 32 && kvHeads > 0 && queryHeads <= 16 * kvHeads && headWidth == 256;
            if(volta) go("attentionVolta", kvHeads, (unsigned int)n, 128, 1, q, L.keys, keyScales, L.values, valueScales, cells, positionsOnGpu, o, budget);
            else if(tensor && pack > 1) go("attentionIntPacked", kvHeads, (unsigned int)((n + pack - 1) / pack), 128, 1, q, L.keys, keyScales, L.values, valueScales, positionsOnGpu, o, n);
            else if(tensor) go("attentionInt", kvHeads, (unsigned int)n, 128, 1, q, L.keys, keyScales, L.values, valueScales, cells, positionsOnGpu, o, budget);
            else go("attentionTiles", kvHeads, (unsigned int)tiles, 256, 1, q, L.keys, keyScales, L.values, valueScales, positionsOnGpu, o, n);
        }
        else if(splitAttention && n * queryHeads < 256){
            int splits = std::max(2, std::min(16, 384 / (queryHeads * n)));
            go("attentionSplit", queryHeads, (unsigned int)(n * splits), 256, 1, q, L.keys, keyScales, L.values, valueScales, cells, positionsOnGpu, splitScratch, budget, splits);
            go("attentionMerge", queryHeads, (unsigned int)n, 256, 1, q, splitScratch, o, splits);
        }
        else if(voltaAttention && n >= 32 && kvHeads > 0 && queryHeads <= 16 * kvHeads && headWidth == 256) go("attentionVolta", kvHeads, (unsigned int)n, 128, 1, q, L.keys, keyScales, L.values, valueScales, cells, positionsOnGpu, o, budget);
        else if(mmaAttention && !legacy && !amdCards && n >= 32 && kvHeads > 0 && queryHeads <= 16 * kvHeads && headWidth == 256) go(intAttention ? "attentionInt" : "attentionMma", kvHeads, (unsigned int)n, 128, 1, q, L.keys, keyScales, L.values, valueScales, cells, positionsOnGpu, o, budget);
        else if(groupAttention && n >= 32 && kvHeads > 0 && queryHeads <= 24 * kvHeads) go("attentionGroup", kvHeads, (unsigned int)n, 256, 1, q, L.keys, keyScales, L.values, valueScales, cells, positionsOnGpu, o, budget);
        else go("attention", queryHeads, (unsigned int)n, 256, 1, q, L.keys, keyScales, L.values, valueScales, cells, positionsOnGpu, o, budget);
        if(!halvedAll({&L.attnOut})) pack(o, n * innerWidth, 0);
        project(L.attnOut, o, 0, y, n);
    }

    int reachStep(int top) const {
        if(indexHeads > 0) return 0;
        return top < 1024
            ? 0
            : top < 8192
                ? 1
                : top < 32768
                    ? 2
                    : 3;
    }

    bool loadDraft(const std::string& path) {
        ModelFile shape;
        std::string error;
        if(!openModel(path, shape, error)){
            problem = "draft layer: " + error;
            return false;
        }
        for(const TensorInfo& t : shape.tensors){
            if(!t.expert || t.dimCount < 3 || (int)t.dims[2] == expertCount) continue;
            problem = "the draft layer has " + std::to_string(t.dims[2]) + " experts and this model " + std::to_string(expertCount);
            return false;
        }
        FILE* f = nullptr;
        fopen_s(&f, path.c_str(), "rb");
        if(!f){
            problem = "draft layer: cannot open " + path;
            return false;
        }
        auto upload = [&](const std::string& name, const TensorInfo** info) -> uint64_t {
            const TensorInfo* t = findTensor(shape, name);
            if(!t){
                if(problem.empty()) problem = "draft layer is missing " + name;
                return 0;
            }
            std::vector<uint8_t> bytes((size_t)t->bytes);
            _fseeki64(f, (int64_t)t->offset, SEEK_SET);
            uint64_t ptr = 0;
            if(fread(bytes.data(), 1, bytes.size(), f) != bytes.size() || cudaLink().alloc(&ptr, (size_t)std::max<uint64_t>(t->bytes, 256)) != 0){
                if(problem.empty()) problem = "draft layer: cannot load " + name;
                return 0;
            }
            allocations.push_back(ptr);
            cudaLink().copyUp(ptr, bytes.data(), bytes.size());
            if(info) *info = t;
            return ptr;
        };
        auto matrix = [&](const std::string& name, GpuMatrix& m) {
            const TensorInfo* t = nullptr;
            m.data = upload(name, &t);
            if(!t) return;
            m.type = t->type;
            m.cols = (int)t->dims[0];
            m.rows = (int)(t->elements / t->dims[0]);
            m.stride = rowBytes(t->type, t->dims[0]);
        };
        auto grabbed = [&](uint64_t bytes) -> uint64_t {
            uint64_t ptr = 0;
            if(cudaLink().alloc(&ptr, (size_t)std::max<uint64_t>(bytes, 256)) != 0){
                if(problem.empty()) problem = "draft layer: out of video memory";
                return 0;
            }
            allocations.push_back(ptr);
            return ptr;
        };

        const TensorInfo* normInfo = findTensor(shape, "draft.embed_norm");
        if(normInfo && normInfo->elements == (uint64_t)hidden){
            std::vector<float> wide((size_t)hcWidth);
            _fseeki64(f, (int64_t)normInfo->offset, SEEK_SET);
            fread(wide.data(), 4, hidden, f);
            for(int c = 1; c < hcCount; c++) memcpy(wide.data() + (size_t)c * hidden, wide.data(), (size_t)hidden * 4);
            draftEmbedNorm = grabbed((uint64_t)hcWidth * 4);
            if(draftEmbedNorm) cudaLink().copyUp(draftEmbedNorm, wide.data(), wide.size() * 4);
        }
        draftHiddenNorm = upload("draft.hidden_norm", nullptr);
        draftOutNorm = upload("draft.out_hc_norm", nullptr);
        matrix("draft.embed_proj", draftEmbedProj);
        matrix("draft.hidden_proj", draftHiddenProj);
        matrix("draft.out_hc_down", draftOutDown);
        matrix("draft.out_hc_up", draftOutUp);

        draft.attention = true;
        const char* halves[2] = {"hc_attn_", "hc_ffn_"};
        for(int h = 0; h < 2; h++){
            draft.hcNorm[h] = upload(std::string("blk.0.") + halves[h] + "norm.weight", nullptr);
            matrix(std::string("blk.0.") + halves[h] + "down.weight", draft.hcDown[h]);
            matrix(std::string("blk.0.") + halves[h] + "up.weight", draft.hcUp[h]);
            matrix(std::string("blk.0.") + halves[h] + "inject.weight", draft.hcInject[h]);
        }
        matrix("blk.0.attn_q.weight", draft.query);
        matrix("blk.0.attn_k.weight", draft.key);
        matrix("blk.0.attn_v.weight", draft.value);
        matrix("blk.0.attn_output.weight", draft.attnOut);
        matrix("blk.0.indexer.q_proj.weight", draft.indexQuery);
        matrix("blk.0.indexer.k_proj.weight", draft.indexKey);
        draft.queryNorm = upload("blk.0.attn_q_norm.weight", nullptr);
        draft.keyNorm = upload("blk.0.attn_k_norm.weight", nullptr);
        draft.indexQueryNorm = upload("blk.0.indexer.q_norm.weight", nullptr);
        draft.indexKeyNorm = upload("blk.0.indexer.k_norm.weight", nullptr);
        matrix("blk.0.ffn_gate_inp.weight", draft.router);
        matrix("blk.0.ffn_gate_inp_shexp.weight", draft.sharedScalar);
        matrix("blk.0.ffn_gate_shexp.weight", draft.sharedGate);
        matrix("blk.0.ffn_up_shexp.weight", draft.sharedUp);
        matrix("blk.0.ffn_down_shexp.weight", draft.sharedDown);

        const char* expertNames[3] = {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"};
        Matrix* parts[3] = {&draft.expertGate, &draft.expertUp, &draft.expertDown};
        for(int i = 0; i < 3; i++){
            const TensorInfo* t = nullptr;
            draft.resident[i] = upload(expertNames[i], &t);
            if(!t) break;
            parts[i]->data = nullptr;
            parts[i]->type = t->type;
            parts[i]->cols = (int)t->dims[0];
            parts[i]->rows = (int)t->dims[1];
            parts[i]->stride = rowBytes(t->type, t->dims[0]);
        }
        fclose(f);

        draft.slot.resize(expertCount);
        for(int e = 0; e < expertCount; e++) draft.slot[e] = e;
        draft.slots = grabbed((uint64_t)expertCount * 4);
        if(draft.slots) cudaLink().copyUp(draft.slots, draft.slot.data(), expertCount * 4);
        auto lined = [&](uint64_t width) -> uint64_t {
            uint64_t total = (uint64_t)contextLimit * width * 9 / 8;
            uint64_t base = stretchBuffer(total, {{0, width}, {(uint64_t)contextLimit * width, width / 8}});
            return base ? base : grabbed(total);
        };
        draft.keys = lined((uint64_t)kvHeads * headWidth);
        draft.values = lined((uint64_t)kvHeads * headWidth);
        draft.indexRaw = lined((uint64_t)indexWidth);
        uint64_t pooledBytes = (uint64_t)scoreStride * indexWidth * 4;
        draft.pooled = stretchBuffer(pooledBytes, {{0, (uint64_t)indexWidth}});
        if(!draft.pooled) draft.pooled = grabbed(pooledBytes);
        draftStack = grabbed((uint64_t)draftBatch * hcWidth * 4);
        draftRows = grabbed((uint64_t)draftBatch * hcWidth * 4);
        draftPending = grabbed((uint64_t)hcWidth * 4);
        draftOwn = grabbed((uint64_t)hcWidth * 4);
        draftHead = head;
        draftVocab.clear();
        std::vector<int> picks;
        std::ifstream list(vocabPath, std::ios::binary | std::ios::ate);
        if(list){
            picks.resize((size_t)list.tellg() / 4);
            list.seekg(0);
            list.read((char*)picks.data(), picks.size() * 4);
        }
        bool usable = !picks.empty() && picks.size() < (size_t)head.rows;
        for(int id : picks) usable = usable && id >= 0 && id < head.rows;
        uint64_t rows = usable ? grabbed((uint64_t)picks.size() * head.stride) : 0;
        if(rows){
            for(size_t i = 0; i < picks.size(); i++) cudaQueue().copyAcross(rows + i * head.stride, head.data + (uint64_t)picks[i] * head.stride, (size_t)head.stride, stream);
            draftHead.data = rows;
            draftHead.rows = (int)picks.size();
            draftVocab = picks;
        }
        if(!problem.empty() || cudaLink().sync() != 0) return false;
        drafting = true;
        return true;
    }

    void settleDraft(int n, bool guess) {
        go("streamNorm", 1, 1, 256, 1, draftStack + 4ULL * (n - 1) * hcWidth, draftOutNorm, draftOwn);
        if(!guess) return;
        pack(draftOwn, hidden, 0);
        project(draftHead, draftOwn, 0, vocabOut, 1);
        go("pickTop", 1, 1, 1024, 1, vocabOut, draftHead.rows, mappedFlags + 224);
    }

    bool loadTiedDraft(int block) {
        std::string p = "blk." + std::to_string(block) + ".";
        if(!findTensor(file, p + "nextn.eh_proj.weight") || hcCount != 1 || indexHeads > 0) return false;
        auto upload = [&](const std::string& name, const TensorInfo** info) -> uint64_t {
            const TensorInfo* t = findTensor(file, name);
            if(!t){
                if(problem.empty()) problem = "the built in draft layer is missing " + name;
                return 0;
            }
            uint64_t ptr = 0;
            if(cudaLink().alloc(&ptr, (size_t)std::max<uint64_t>(t->bytes, 256)) != 0 || cudaLink().copyUp(ptr, maps[t->file].base + t->offset, (size_t)t->bytes) != 0){
                if(problem.empty()) problem = "the built in draft layer does not fit: " + name;
                return 0;
            }
            allocations.push_back(ptr);
            if(info) *info = t;
            return ptr;
        };
        auto matrix = [&](const std::string& name, GpuMatrix& m) {
            const TensorInfo* t = nullptr;
            m.data = upload(name, &t);
            if(!t) return;
            m.type = t->type;
            m.cols = (int)t->dims[0];
            m.rows = (int)(t->elements / t->dims[0]);
            m.stride = rowBytes(t->type, t->dims[0]);
        };
        auto narrowed = [&](const std::string& name, GpuMatrix& m) {
            const TensorInfo* t = findTensor(file, name);
            if(!t || !knownType(t->type)){
                if(problem.empty()) problem = "the built in draft layer is missing " + name;
                return;
            }
            m.type = 30;
            m.cols = (int)t->dims[0];
            m.rows = (int)(t->elements / t->dims[0]);
            m.stride = (uint64_t)m.cols * 2;
            const uint8_t* from = maps[t->file].base + t->offset;
            uint64_t inRow = rowBytes(t->type, t->dims[0]);
            std::vector<float> row((size_t)m.cols);
            std::vector<uint16_t> half((size_t)t->elements);
            for(int r = 0; r < m.rows; r++){
                dequantRow(t->type, from + (uint64_t)r * inRow, row.data(), m.cols);
                for(int c = 0; c < m.cols; c++){
                    uint32_t bits;
                    memcpy(&bits, &row[c], 4);
                    bits += 0x7FFF + ((bits >> 16) & 1);
                    half[(size_t)r * m.cols + c] = (uint16_t)(bits >> 16);
                }
            }
            if(cudaLink().alloc(&m.data, half.size() * 2) != 0 || cudaLink().copyUp(m.data, half.data(), half.size() * 2) != 0){
                if(problem.empty()) problem = "the built in draft layer does not fit: " + name;
                m.data = 0;
                return;
            }
            allocations.push_back(m.data);
        };
        auto grabbed = [&](uint64_t bytes) -> uint64_t {
            uint64_t ptr = 0;
            if(cudaLink().alloc(&ptr, (size_t)std::max<uint64_t>(bytes, 256)) != 0){
                if(problem.empty()) problem = "the built in draft layer: out of video memory";
                return 0;
            }
            allocations.push_back(ptr);
            return ptr;
        };

        GpuMatrix joined;
        matrix(p + "nextn.eh_proj.weight", joined);
        TypeInfo info = typeInfo(joined.type);
        if(!problem.empty() || joined.cols != 2 * hidden || joined.rows != hidden || info.blockElements == 0 || hidden % info.blockElements != 0){
            if(problem.empty()) problem = "unexpected shape of the built in draft projection";
            return false;
        }
        draftEmbedProj = joined;
        draftEmbedProj.cols = hidden;
        draftHiddenProj = joined;
        draftHiddenProj.cols = hidden;
        draftHiddenProj.data += (uint64_t)hidden / info.blockElements * info.blockBytes;
        draftEmbedNorm = upload(p + "nextn.enorm.weight", nullptr);
        draftHiddenNorm = upload(p + "nextn.hnorm.weight", nullptr);
        draftOutNorm = upload(p + "nextn.shared_head_norm.weight", nullptr);

        draft = GpuLayer();
        draft.attention = true;
        draft.hcNorm[0] = upload(p + "attn_norm.weight", nullptr);
        draft.hcNorm[1] = upload(p + "post_attention_norm.weight", nullptr);
        matrix(p + "attn_q.weight", draft.query);
        matrix(p + "attn_k.weight", draft.key);
        matrix(p + "attn_v.weight", draft.value);
        matrix(p + "attn_output.weight", draft.attnOut);
        draft.queryNorm = upload(p + "attn_q_norm.weight", nullptr);
        draft.keyNorm = upload(p + "attn_k_norm.weight", nullptr);
        if(dense){
            matrix(p + "ffn_gate.weight", draft.sharedGate);
            matrix(p + "ffn_up.weight", draft.sharedUp);
            matrix(p + "ffn_down.weight", draft.sharedDown);
        } else {
            narrowed(p + "ffn_gate_inp.weight", draft.router);
            narrowed(p + "ffn_gate_inp_shexp.weight", draft.sharedScalar);
            matrix(p + "ffn_gate_shexp.weight", draft.sharedGate);
            matrix(p + "ffn_up_shexp.weight", draft.sharedUp);
            matrix(p + "ffn_down_shexp.weight", draft.sharedDown);
            const char* expertNames[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
            Matrix* parts[3] = {&draft.expertGate, &draft.expertUp, &draft.expertDown};
            for(int i = 0; i < 3; i++){
                const TensorInfo* t = nullptr;
                draft.resident[i] = upload(p + expertNames[i], &t);
                if(!t) break;
                parts[i]->data = nullptr;
                parts[i]->type = t->type;
                parts[i]->cols = (int)t->dims[0];
                parts[i]->rows = (int)t->dims[1];
                parts[i]->stride = rowBytes(t->type, t->dims[0]);
            }
            draft.slot.resize(expertCount);
            for(int e = 0; e < expertCount; e++) draft.slot[e] = e;
            draft.slots = grabbed((uint64_t)expertCount * 4);
            if(draft.slots) cudaLink().copyUp(draft.slots, draft.slot.data(), expertCount * 4);
        }
        uint64_t lineBytes = (uint64_t)contextLimit * kvHeads * headWidth * 9 / 8;
        std::vector<std::array<uint64_t, 2>> lineParts = {{0, (uint64_t)kvHeads * headWidth}, {(uint64_t)contextLimit * kvHeads * headWidth, (uint64_t)kvHeads * headWidth / 8}};
        draft.keys = stretchBuffer(lineBytes, lineParts);
        if(!draft.keys) draft.keys = grabbed(lineBytes);
        draft.values = stretchBuffer(lineBytes, lineParts);
        if(!draft.values) draft.values = grabbed(lineBytes);
        draftStack = grabbed((uint64_t)draftBatch * hcWidth * 4);
        draftRows = grabbed((uint64_t)draftBatch * hcWidth * 4);
        draftPending = grabbed((uint64_t)hcWidth * 4);
        draftOwn = grabbed((uint64_t)hcWidth * 4);
        draftHead = head;
        draftVocab.clear();
        // SNEO_DRAFT_HEAD_ROWS cambia cuantos ids mira el borrador (0 = la cabeza entera)
        char rowsText[16] = {};
        if(GetEnvironmentVariableA("SNEO_DRAFT_HEAD_ROWS", rowsText, sizeof(rowsText)) > 0) draftHeadRows = atoi(rowsText);
        if(draftHeadRows > 0 && draftHead.rows > draftHeadRows) draftHead.rows = draftHeadRows;
        if(!problem.empty() || cudaLink().sync() != 0) return false;
        tiedDraft = true;
        drafting = true;
        return true;
    }

    void draftLayer(uint64_t rows, uint64_t embeds, int n, int top, bool guess) {
        CudaQueue& queue = cudaQueue();
        if(tiedDraft){
            uint64_t steady = rows;
            if(rows != draftOwn){
                go("streamNorm", (unsigned int)n, 1, 256, 1, rows, outNorm, normd);
                steady = normd;
            }
            go("streamNorm", (unsigned int)n, 1, 256, 1, steady, draftHiddenNorm, xn);
            go("streamNorm", (unsigned int)n, 1, 256, 1, embeds, draftEmbedNorm, x);
            pack(x, n * hidden, 0);
            project(draftEmbedProj, x, 0, y, n);
            pack(xn, n * hidden, 0);
            project(draftHiddenProj, xn, 0, mixGate, n);
        } else {
            go("streamNorm", (unsigned int)n, 1, 256, 1, embeds, draftEmbedNorm, x);
            plain(draftEmbedProj, x, hidden, y, hidden, n, 0);
            go("streamNorm", (unsigned int)(n * hcCount), 1, 256, 1, rows, draftHiddenNorm, xn);
            plain(draftHiddenProj, xn, hidden, mixGate, hidden, n * hcCount, 0);
        }
        go("addSpread", (unsigned int)((n * hcWidth + 255) / 256), 1, 256, 1, mixGate, y, draftStack, n * hcWidth);

        mix(draft.hcNorm[0], draft.hcDown[0], draft.hcUp[0], &draft.hcInject[0], draftStack, n, x, inj);
        attend(draft, n, top, !guess);
        if(!guess) return;
        go("writeBack", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, draftStack, y, inj, n);
        mix(draft.hcNorm[1], draft.hcDown[1], draft.hcUp[1], &draft.hcInject[1], draftStack, n, x, inj);

        if(dense){
            pack(x, n * hidden, 0);
            projectMany({{&draft.sharedGate, sharedGateOut}, {&draft.sharedUp, sharedUpOut}}, x, 0, n);
            go("swiglu", (unsigned int)((n * expertWidth + 255) / 256), 1, 256, 1, sharedGateOut, sharedUpOut, sharedHidden, n * expertWidth);
            pack(sharedHidden, n * expertWidth, 1);
            project(draft.sharedDown, sharedHidden, 1, shared, n);
            go("writeBack", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, draftStack, shared, inj, n);
            settleDraft(n, guess);
            return;
        }

        pack(x, n * hidden, 0);
        plain(draft.router, x, hidden, routerOut, expertCount, n, 0);
        plain(draft.sharedScalar, x, hidden, scalar, 1, n, 0);
        projectMany({{&draft.sharedGate, sharedGateOut}, {&draft.sharedUp, sharedUpOut}}, x, 0, n);
        go("routeTop", (unsigned int)n, 1, expertCount, 1, routerOut, routeIds, routeWeights);
        uint64_t room = (uint64_t)std::max(expertCount, expertsUsed * n);
        uint64_t plan = lists + 4ULL * expertsUsed * n;
        uint64_t none = 0;
        go("planBatch", 1, 1, 32, 1, routeIds, draft.slots, none, n, 0, lists, plan, plan + 4ULL * room, plan + 8ULL * room, plan + 12ULL * room, plan + 16ULL * room, plan + 20ULL * room, spillCounts, routeWeights, 0, none, none);
        go("swiglu", (unsigned int)((n * expertWidth + 255) / 256), 1, 256, 1, sharedGateOut, sharedUpOut, sharedHidden, n * expertWidth);
        pack(sharedHidden, n * expertWidth, 1);
        project(draft.sharedDown, sharedHidden, 1, shared, n);

        int count = expertsUsed * n;
        uint64_t gateSpan = (uint64_t)draft.expertGate.rows * draft.expertGate.stride;
        uint64_t upSpan = (uint64_t)draft.expertUp.rows * draft.expertUp.stride;
        uint64_t downSpan = (uint64_t)draft.expertDown.rows * draft.expertDown.stride;
        PackedInput& a = packs[0];
        PackedInput b = residual ? fines[0] : PackedInput();
        int stage = (int)((draft.expertGate.stride + 31) / 16 * 16);
        goShared(8 * stage, "rowsRouted", expertWidth / 8, (unsigned int)count, 256, 1, (int)draft.expertGate.type, draft.resident[0], draft.expertGate.stride, gateSpan, plan, plan + 8ULL * room, plan + 4ULL * room, a.values, a.scales, a.sums, b.values, b.scales, b.sums, expertGateOut, expertWidth, hidden, stage, tables);
        goShared(8 * stage, "rowsRouted", expertWidth / 8, (unsigned int)count, 256, 1, (int)draft.expertUp.type, draft.resident[1], draft.expertUp.stride, upSpan, plan, plan + 8ULL * room, plan + 4ULL * room, a.values, a.scales, a.sums, b.values, b.scales, b.sums, expertUpOut, expertWidth, hidden, stage, tables);
        go("swiglu", (unsigned int)((count * expertWidth + 255) / 256), 1, 256, 1, expertGateOut, expertUpOut, expertHidden, count * expertWidth);
        pack(expertHidden, count * expertWidth, 1);
        PackedInput& c = packs[1];
        PackedInput d = residual ? fines[1] : PackedInput();
        int downStage = (int)((draft.expertDown.stride + 31) / 16 * 16);
        goShared(8 * downStage, "rowsStaged", hidden / 8, (unsigned int)count, 256, 1, (int)draft.expertDown.type, draft.resident[2], draft.expertDown.stride, downSpan, plan, plan + 8ULL * room, expertWidth, c.values, c.scales, c.sums, d.values, d.scales, d.sums, expertOut, hidden, expertWidth, downStage, tables);
        int spillBase = 2 * count;
        go("writeExperts", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, draftStack, shared, scalar, expertOut, none, spillBase, routeWeights, lists, inj, n);
        if(tiedDraft){
            settleDraft(n, guess);
            return;
        }
        queue.copyAcross(draftOwn, draftStack + 4ULL * (n - 1) * hcWidth, (size_t)hcWidth * 4, stream);

        if(guess){
            mix(draftOutNorm, draftOutDown, draftOutUp, nullptr, draftStack + 4ULL * (n - 1) * hcWidth, 1, fin, 0);
            pack(fin, hidden, 0);
            project(draftHead, fin, 0, vocabOut, 1);
            go("pickTop", 1, 1, 1024, 1, vocabOut, draftHead.rows, mappedFlags + 224);
        }
    }

    bool draftCells(uint64_t rows, const int* next, const int* at, int n, int* guess, float* sure = nullptr) {
        if(!drafting || n <= 0 || n > draftBatch) return false;
        int furthest = 0;
        for(int t = 0; t < n; t++) furthest = std::max(furthest, at[t]);
        if(furthest >= contextLimit || !reachContext(furthest + 1)) return false;
        waitStream();
        if(n >= 64){
            pool.run(n, [&](int t){
                embedCell(next[t], hostEmbed + (size_t)t * hidden);
            });
        }
        else {
            for(int t = 0; t < n; t++) embedCell(next[t], hostEmbed + (size_t)t * hidden);
        }
        for(int t = 0; t < n; t++) hostPositions[t] = at[t];
        int top = 0;
        for(int t = 0; t < n; t++) top = std::max(top, at[t]);
        go("copyWords", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, mappedEmbed, embedRows, n * hidden);
        go("copyWords", (unsigned int)((n + 255) / 256), 1, 256, 1, mappedPositions, positionsOnGpu, n);
        fixed = false;
        draftLayer(rows, embedRows, n, top, guess != nullptr);
        if(waitStream() != 0 || failed){
            failed = false;
            return false;
        }
        int index = std::max(0, (int)hostFlags[56]);
        if(guess) *guess = draftVocab.empty()
            ? std::min(index, (int)head.rows - 1)
            : draftVocab[std::min(index, (int)draftVocab.size() - 1)];
        if(guess && sure){
            uint32_t bits = hostFlags[57];
            memcpy(sure, &bits, 4);
        }
        return true;
    }

    void draftFeed(const std::vector<int>& ids, const std::vector<int>& positions, size_t at, int n) {
        if(!drafting || n <= 0) return;
        auto clock = std::chrono::steady_clock::now();
        CudaQueue& queue = cudaQueue();
        phase = "draft";
        int known = 0;
        for(int i = 0; i < n; i++){
            if(at + i + 1 < ids.size()) known++;
        }
        int batch = std::max(1, std::min(draftBatch, crestRows));
        int inside = std::min(known, n - 1), done = 0;
        if(fastFeed && inside > 0 && positions[at + inside - 1] < contextLimit && reachContext(positions[at + inside - 1] + 1)){
            for(int first = 0; first < inside; first += batch){
                int count = std::min(batch, inside - first);
                go("copyWords", (unsigned int)((count * hidden + 255) / 256), 1, 256, 1, mappedEmbed + 4ULL * (first + 1) * hidden, embedRows, count * hidden);
                uint64_t keptPositions = positionsOnGpu;
                positionsOnGpu += 4ULL * first;
                fixed = false;
                draftLayer(stack + 4ULL * first * hcWidth, embedRows, count, positions[at + first + count - 1], false);
                positionsOnGpu = keptPositions;
                if(failed) return;
            }
            done = inside;
        }
        for(int first = done; first < known; first += batch){
            int count = std::min(batch, known - first);
            queue.copyAcross(draftRows, stack + 4ULL * first * hcWidth, (size_t)count * hcWidth * 4, stream);
            std::vector<int> next(count), cells(count);
            for(int i = 0; i < count; i++){
                next[i] = ids[at + first + i + 1];
                cells[i] = positions[at + first + i];
            }
            if(!draftCells(draftRows, next.data(), cells.data(), count, nullptr)) return;
        }
        waitStream();
        if(known < n){
            queue.copyAcross(draftPending, stack + 4ULL * (n - 1) * hcWidth, (size_t)hcWidth * 4, stream);
            draftPendingAt = positions[at + n - 1];
            draftConsumed = false;
        }
        feedMs += since(clock);
    }

    void settleTail() {
        if(feedWaiting){
            feedWaiting = false;
            draftFeed(feedIds, feedPositions, feedAt, feedRows);
        }
        if(repayWaiting){
            repayWaiting = false;
            repay();
        }
    }

    std::vector<int> draftPropose(int token, int most) {
        if(gemma) return gemmaPropose(token, most);
        if(hy4) return hyPropose(token, most);
        settleTail();
        std::vector<int> drafts;
        if(!drafting || draftPendingAt < 0 || most <= 0) return drafts;
        int guess = -1;
        float sure = 0.0f;
        int at = draftPendingAt;
        if(!draftCells(draftPending, &token, &at, 1, &guess, &sure)) return drafts;
        draftConsumed = true;
        if(sure < draftSure) return drafts;
        drafts.push_back(guess);
        for(int k = 1; k < most; k++){
            int place = at + k;
            if(place >= contextLimit) break;
            int previous = drafts.back();
            if(!draftCells(draftOwn, &previous, &place, 1, &guess, &sure) || sure < draftSure) break;
            drafts.push_back(guess);
        }
        return drafts;
    }

    uint64_t replayStride() {
        return (uint64_t)relayLimit * (2 * convWidth + 2 * valueHeads);
    }

    bool commitRows(const std::vector<int>& accepted) {
        int count = (int)accepted.size();
        if(gemma){
            if(count < 1 || count > gemmaBuffers.everyCap) return false;
            tokens.resize(std::min(tokens.size(), savedTokens[1]));
            tokens.insert(tokens.end(), accepted.begin(), accepted.end());
            return gemmaKeep(count);
        }
        if(hy4){
            if(count < 1 || count > hyBuffers.everyCap) return false;
            tokens.resize(std::min(tokens.size(), savedTokens[1]));
            tokens.insert(tokens.end(), accepted.begin(), accepted.end());
            hyDraft.from = hyBuffers.normedRows + 4ULL * (count - 1) * hyShape.hidden;
            hyDraft.at = (int)tokens.size() - 1;
            hyDraft.steps++;
            hyDraft.kept += count - 1;
            if(count > 1) hyDraft.firstKept++;
            return true;
        }
        if(!replayStore || replayRows < 2 || count < 1 || count > replayRows || !useCuda()) return false;
        if(settling){
            tokens.resize(std::min(tokens.size(), savedTokens[1]));
            draftPendingAt = -1;
            draftConsumed = true;
        } else if(!moveState(1, false, false)) return false;
        if(remote && !remote->state(STATE_COMMIT, count)) return false;
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            if(L.attention) continue;
            uint64_t at = replayStore + 4ULL * replayStride() * l;
            uint64_t inputs = at + 4ULL * relayLimit * convWidth;
            uint64_t alphas = at + 8ULL * relayLimit * convWidth;
            uint64_t betas = alphas + 4ULL * relayLimit * valueHeads;
            go("gdnHistory", convWidth / 256, 1, 256, 1, at, L.convHistory, count);
            go("gdnScan", ssmSize / 32, valueHeads, 128, 1, inputs, alphas, betas, L.dtBias, L.decay, L.state, o, count, 1);
        }
        if(usePle && firstLayer <= 1) go("pleHistory", hcWidth / 256, 1, 256, 1, replayStore + 4ULL * replayStride() * layers.size(), pleHistory, count);
        tokens.insert(tokens.end(), accepted.begin(), accepted.end());
        if(drafting){
            cudaQueue().copyAcross(draftPending, stack + 4ULL * (count - 1) * hcWidth, (size_t)hcWidth * 4, stream);
            draftPendingAt = (int)tokens.size() - 1;
            draftConsumed = false;
        }
        replayRows = 0;
        bool ok = waitStream() == 0 && !failed;
        failed = false;
        return ok;
    }

    bool settles(int rows) {
        return !gemma && (!remote || (remote->features & ROWS_LEAVE)) && replayStore && rows > 1 && rows <= relayLimit && fine && !exact && ((!profile && useGraphs) || relayAlways);
    }

    bool markRows(int rows) {
        settling = settles(rows);
        if(!settling) return saveState(1);
        savedTokens[1] = tokens.size();
        return true;
    }

    bool rewindRows() {
        if(!settling) return restoreState(1);
        tokens.resize(std::min(tokens.size(), savedTokens[1]));
        draftPendingAt = -1;
        draftConsumed = true;
        replayRows = 0;
        return true;
    }

    bool settleRows(const std::vector<int>& rows) {
        if(hy4){
            if(!rows.empty() && (int)rows.size() <= hyBuffers.everyCap){
                hyDraft.from = hyBuffers.normedRows + 4ULL * (rows.size() - 1) * hyShape.hidden;
                hyDraft.at = (int)tokens.size() - 1;
                hyDraft.steps++;
                hyDraft.kept += rows.size() - 1;
                if(rows.size() > 1) hyDraft.firstKept++;
            }
            return true;
        }
        if(gemma){
            if(!rows.empty() && (int)rows.size() <= gemmaBuffers.everyCap) gemmaKeep((int)rows.size());
            return true;
        }
        return !settling || commitRows(rows);
    }

    bool tailActive() {
        return tailSkip > 0 && residentExperts * 10 < expertCount * 9;
    }

    bool teamed(int n) const {
        return teams && relay && n > 1 && n <= std::min(std::max(teamRows, 16), 128) && expertCount % 32 == 0 && expertCount <= 1024;
    }

    uint64_t teamSpan() const {
        return 16ULL * expertsUsed * std::min(std::max(teamRows, 16), 128);
    }

    uint64_t stageOf(int l) const {
        return (uint64_t)(stageRing > 0 ? l % stageRing : l);
    }

    uint64_t stretchBuffer(uint64_t size, const std::vector<std::array<uint64_t, 2>>& parts) {
        CudaLink& cu = cudaLink();
        if(!growKv || !cu.stretchable || contextLimit <= kvStart) return 0;
        if(pageGrain == 0){
            CuMemoryKind kind;
            kind.device = cu.device;
            size_t grain = 0;
            if(cu.grain(&grain, &kind, 0) != 0 || grain == 0) return 0;
            pageGrain = grain;
        }
        uint64_t total = (size + pageGrain - 1) / pageGrain * pageGrain;
        uint64_t base = 0;
        if(cu.reserve(&base, (size_t)total, (size_t)pageGrain, 0, 0) != 0) return 0;
        Stretch s;
        s.base = base;
        s.size = total;
        s.parts = parts;
        s.mapped.assign((size_t)(total / pageGrain), 0);
        stretches.push_back(s);
        if(kvRoom > 0 && !mapStretch(stretches.back(), kvRoom)) return 0;
        return base;
    }

    uint64_t stretchNeed(const Stretch& s, int room, std::vector<char>& need) const {
        need.assign(s.mapped.size(), 0);
        for(const auto& part : s.parts){
            uint64_t end = std::min<uint64_t>(s.size, part[0] + (uint64_t)room * part[1] + (64ULL << 10));
            for(uint64_t p = part[0] / pageGrain; p * pageGrain < end; p++) need[p] = !s.mapped[p];
        }
        uint64_t bytes = 0;
        for(char n : need) bytes += n ? pageGrain : 0;
        return bytes;
    }

    bool mapStretch(Stretch& s, int room) {
        CudaLink& cu = cudaLink();
        std::vector<char> need;
        if(stretchNeed(s, room, need) == 0) return true;
        CuMemoryKind kind;
        kind.device = cu.device;
        CuMemoryAccess access;
        access.device = cu.device;
        for(size_t p = 0; p < need.size(); p++){
            if(!need[p]) continue;
            if(!freePages.empty()){
                uint64_t at = s.base + p * pageGrain;
                unsigned long long handle = freePages.back();
                if(cu.map(at, (size_t)pageGrain, 0, handle, 0) != 0){
                    problem = "the context memory could not be mapped";
                    return false;
                }
                freePages.pop_back();
                stretchPages.push_back({at, pageGrain, handle});
                if(cu.allow(at, (size_t)pageGrain, &access, 1) != 0){
                    problem = "the context memory could not be opened to the gpu";
                    return false;
                }
                s.mapped[p] = 1;
                continue;
            }
            size_t end = p;
            while(end < need.size() && need[end]) end++;
            uint64_t at = s.base + p * pageGrain, bytes = (end - p) * pageGrain;
            unsigned long long handle = 0;
            if(cu.carve(&handle, (size_t)bytes, &kind, 0) != 0){
                problem = "out of video memory growing the context by " + std::to_string(bytes >> 20) + " MB";
                return false;
            }
            if(cu.map(at, (size_t)bytes, 0, handle, 0) != 0){
                cu.forget(handle);
                problem = "the context memory could not be mapped";
                return false;
            }
            stretchPages.push_back({at, bytes, handle});
            if(cu.allow(at, (size_t)bytes, &access, 1) != 0){
                problem = "the context memory could not be opened to the gpu";
                return false;
            }
            for(size_t q = p; q < end; q++) s.mapped[q] = 1;
            p = end;
        }
        return true;
    }

    int recentFor(int resident) {
        return recentSlots > 0 && resident >= 4 * recentFloor && resident < expertCount && cudaQueue().streamSync
            ? std::min(recentSlots, resident / 4)
            : 0;
    }

    uint64_t poolBuffer(uint64_t size, uint64_t solid) {
        CudaLink& cu = cudaLink();
        if(stretches.empty() || pageGrain == 0){
            uint64_t ptr = 0;
            if(cu.alloc(&ptr, (size_t)size) != 0) return 0;
            allocations.push_back(ptr);
            return ptr;
        }
        PoolSpan fresh;
        fresh.size = (size + pageGrain - 1) / pageGrain * pageGrain;
        fresh.solid = std::min(fresh.size, solid / pageGrain * pageGrain);
        if(cu.reserve(&fresh.base, (size_t)fresh.size, (size_t)pageGrain, 0, 0) != 0) return 0;
        fresh.pages.assign((size_t)((fresh.size - fresh.solid) / pageGrain), 0);
        poolSpans.push_back(fresh);
        PoolSpan& p = poolSpans.back();
        CuMemoryKind kind;
        kind.device = cu.device;
        CuMemoryAccess access;
        access.device = cu.device;
        if(p.solid > 0){
            if(cu.carve(&p.whole, (size_t)p.solid, &kind, 0) != 0){
                p.whole = 0;
                return 0;
            }
            if(cu.map(p.base, (size_t)p.solid, 0, p.whole, 0) != 0){
                cu.forget(p.whole);
                p.whole = 0;
                return 0;
            }
        }
        for(size_t i = 0; i < p.pages.size(); i++){
            if(cu.carve(&p.pages[i], (size_t)pageGrain, &kind, 0) != 0){
                p.pages[i] = 0;
                return 0;
            }
            if(cu.map(p.base + p.solid + i * pageGrain, (size_t)pageGrain, 0, p.pages[i], 0) != 0){
                cu.forget(p.pages[i]);
                p.pages[i] = 0;
                return 0;
            }
        }
        if(cu.allow(p.base, (size_t)p.size, &access, 1) != 0) return 0;
        return p.base;
    }

    void dropPool(PoolSpan& p) {
        CudaLink& cu = cudaLink();
        if(p.whole){
            cu.unmap(p.base, (size_t)p.solid);
            cu.forget(p.whole);
        }
        for(size_t i = 0; i < p.pages.size(); i++){
            if(!p.pages[i]) continue;
            cu.unmap(p.base + p.solid + i * pageGrain, (size_t)pageGrain);
            cu.forget(p.pages[i]);
        }
        if(p.base) cu.unreserve(p.base, (size_t)p.size);
        p = PoolSpan();
    }

    const uint8_t* hostExpert(int layer, int part, int e, const uint8_t* fallback) {
        if(heldCopies.empty()) return fallback;
        const uint8_t* own = heldCopies[(size_t)layer * expertCount + e];
        if(!own) return fallback;
        GpuLayer& L = layers[layer];
        uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride;
        uint64_t upSpan = (uint64_t)L.expertUp.rows * L.expertUp.stride;
        return own + (part == 0 ? 0 : part == 1 ? gateSpan : gateSpan + upSpan);
    }

    void keepAside(int l, int e, uint64_t from, uint8_t*& block) {
        GpuLayer& L = layers[l];
        CudaLink& cu = cudaLink();
        const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
        if(L.warmHost[0] && L.warm[e] >= 0) return;
        // los lugares de sobra solo reciben expertos que salen de la vram cuando crece el contexto
        std::vector<int>& rooms = !ramFree[l].empty() ? ramFree[l] : spareFree[l];
        if(L.warmHost[0] && !rooms.empty()){
            int room = rooms.back();
            rooms.pop_back();
            for(int p = 0; p < 3; p++){
                uint64_t span = (uint64_t)parts[p]->rows * parts[p]->stride;
                cu.copyDown(L.warmHost[p] + (uint64_t)room * span, L.resident[p] + from * span, (size_t)span);
            }
            L.warm[e] = room;
            return;
        }
        if(!block) return;
        if(heldCopies.empty()) heldCopies.assign(layers.size() * expertCount, nullptr);
        uint8_t* own = block;
        block += expertBytes(l);
        uint64_t at = 0;
        for(int p = 0; p < 3; p++){
            uint64_t span = (uint64_t)parts[p]->rows * parts[p]->stride;
            cu.copyDown(own + at, L.resident[p] + from * span, (size_t)span);
            at += span;
        }
        heldCopies[(size_t)l * expertCount + e] = own;
    }

    std::vector<std::pair<uint64_t*, uint64_t>> loanParts(uint64_t T, uint64_t W) {
        std::vector<std::pair<uint64_t*, uint64_t>> parts;
        if(T > 0){
            uint64_t S = T * expertsUsed + expertsUsed, D = (uint64_t)expertsUsed * draftBatch;
            uint64_t P = std::max(std::max<uint64_t>(W, crestRows) * innerWidth, T * hidden);
            uint64_t G = std::max<uint64_t>(batchRows, T);
            uint64_t Q = (G + std::max<uint64_t>(W, crestRows)) * expertWidth;
            parts = {
                {&stack, T * hcWidth * 4},
                {&x, T * hidden * 4},
                {&moeOut, T * hidden * 8},
                {&inj, T * hcCount * 4},
                {&positionsOnGpu, T * 4},
                {&scalar, T * 4},
                {&routeIds, T * expertsUsed * 4},
                {&routeWeights, T * expertsUsed * 4},
                {&lists, std::max(2 * S + 6 * std::max<uint64_t>(expertCount, planWidth), D + 6 * std::max<uint64_t>(expertCount, D)) * 4},
                {&packs[0].values, P},
                {&packs[0].scales, P / 8},
                {&packs[0].sums, P / 8},
                {&fines[0].values, P},
                {&fines[0].scales, P / 8},
                {&fines[0].sums, P / 8},
                {&expertGateOut, G * expertWidth * 4},
                {&expertUpOut, G * expertWidth * 4},
                {&expertHidden, G * expertWidth * 4},
                {&packs[1].values, Q},
                {&packs[1].scales, Q / 8},
                {&packs[1].sums, Q / 8},
                {&fines[1].values, Q},
                {&fines[1].scales, Q / 8},
                {&fines[1].sums, Q / 8},
                {&embedRows, 0},
            };
            if(!blas) parts.push_back({&shared, T * hidden * 4});
        }
        if(W > 0){
            std::vector<std::pair<uint64_t*, uint64_t>> crest = {
                {&xn, W * hcWidth * 4},
                {&mixGate, W * hcWidth * 4},
                {&lo, W * hcRank * 4},
                {&y, W * hidden * 4},
                {&qkv, W * convWidth * 4},
                {&z, W * innerWidth * 4},
                {&alphaOut, W * valueHeads * 4},
                {&betaOut, W * valueHeads * 4},
                {&u, W * convWidth * 4},
                {&o, W * innerWidth * 4},
                {&q, W * queryHeads * 2 * headWidth * 4},
                {&k, W * kvHeads * headWidth * 4},
                {&v, W * kvHeads * headWidth * 4},
                {&iq, W * indexHeads * indexWidth * 4},
                {&ik, W * indexWidth * 4},
                {&scores, indexHeads > 0 ? W * scoreStride * 4 : 256},
                {&cells, W * 4096 * 4},
                {&routerOut, W * expertCount * 4},
                {&sharedGateOut, W * expertWidth * 4},
                {&sharedUpOut, W * expertWidth * 4},
                {&sharedHidden, W * expertWidth * 4},
                {&pleEmbed, W * hidden * 4},
                {&pleKeyOut, W * hcWidth * 4},
                {&pleValueOut, W * hidden * 4},
                {&gated, W * hcWidth * 4},
                {&normd, W * hcWidth * 4},
            };
            parts.insert(parts.end(), crest.begin(), crest.end());
        }
        return parts;
    }

    uint64_t loanBytes(uint64_t T, uint64_t W) {
        uint64_t total = 0;
        for(auto& part : loanParts(T, W)) total += (part.second + 255) / 256 * 256;
        return total;
    }

    std::vector<std::array<uint64_t, 2>> loanTail(int from) {
        std::vector<std::array<uint64_t, 2>> pages;
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
            for(int i = 0; i < 3; i++){
                uint64_t start = (uint64_t)(fixedExperts + from) * parts[i]->rows * parts[i]->stride;
                for(PoolSpan& p : poolSpans){
                    if(p.base != L.resident[i]) continue;
                    for(size_t q = 0; q < p.pages.size(); q++){
                        uint64_t at = p.solid + q * pageGrain;
                        if(p.pages[q] && at >= start) pages.push_back({p.base + at, p.pages[q]});
                    }
                }
            }
        }
        return pages;
    }

    bool lend(int rows) {
        loanUsed = 0;
        if(lent || recentPerLayer == 0 || poolSpans.empty() || pageGrain == 0 || dense || legacy || !streamPrefill) return false;
        uint64_t T = rows > superLimit
            ? (uint64_t)std::min(rows, loanRows)
            : 0;
        uint64_t W = rows > crestRows
            ? (uint64_t)std::min(std::min(rows, loanCrest), superLimit)
            : 0;
        if(W <= (uint64_t)crestRows) W = 0;
        if(T == 0 && W == 0) return false;
        CudaLink& cu = cudaLink();
        CudaQueue& queue = cudaQueue();
        liftHold = true;
        exchangeHold = true;
        auto release = [&]{
            liftHold = false;
            exchangeHold = false;
            return false;
        };
        auto clock = std::chrono::steady_clock::now();
        while(liftPending > 0 && since(clock) < 20000.0){
            refresh();
            if(liftPending > 0) Sleep(1);
        }
        while((!moving.empty() || !leaving.empty()) && since(clock) < 20000.0){
            if(!moving.empty()) queue.eventSync(movedEvent);
            else queue.eventSync(leftEvent);
            exchange(0);
        }
        if(liftPending > 0 || !moving.empty() || !leaving.empty()) return release();
        waitStream();

        uint64_t most = (uint64_t)loanTail(0).size() * pageGrain;
        while(W > (uint64_t)crestRows && loanBytes(T, W) > most) W /= 2;
        if(W <= (uint64_t)crestRows) W = 0;
        while(T > (uint64_t)superLimit && loanBytes(T, W) > most) T -= 512;
        if(T <= (uint64_t)superLimit) T = 0;
        if(T == 0 && W == 0) return release();
        uint64_t need = loanBytes(T, W);
        int from = recentPerLayer;
        std::vector<std::array<uint64_t, 2>> pages;
        while(from > 0 && (uint64_t)pages.size() * pageGrain < need) pages = loanTail(--from);
        uint64_t size = (uint64_t)pages.size() * pageGrain;
        if(size < need) return release();

        if(loanStoreSize < size){
            if(loanStore){
                pinned.erase(std::remove(pinned.begin(), pinned.end(), (void*)loanStore), pinned.end());
                cu.hostFree(loanStore);
                loanStore = nullptr;
                loanStoreSize = 0;
            }
            void* room = nullptr;
            if(cu.hostAlloc(&room, (size_t)size, 0) != 0) return release();
            pinned.push_back(room);
            loanStore = (uint8_t*)room;
            loanStoreSize = size;
        }
        bool saved = true;
        for(size_t k = 0; k < pages.size() && saved; k++) saved = queue.copyDown(loanStore + k * pageGrain, pages[k][0], (size_t)pageGrain, stream) == 0;
        if(waitStream() != 0 || !saved) return release();

        loanPages.clear();
        loanSaved.clear();
        loanSlots.clear();
        loanBase = 0;
        loanSize = size;
        lent = true;
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            size_t before = loanSlots.size();
            for(int e = 0; e < expertCount; e++){
                if(L.slot[e] < fixedExperts + from) continue;
                loanSlots.push_back({l, e, L.slot[e]});
                L.slot[e] = -1;
            }
            if(loanSlots.size() > before && cu.copyUp(L.slots, L.slot.data(), expertCount * 4) != 0){
                repay();
                problem = "cannot upload the expert slots";
                return false;
            }
        }
        if(cu.reserve(&loanBase, (size_t)size, (size_t)pageGrain, 0, 0) != 0){
            loanBase = 0;
            repay();
            return false;
        }
        CuMemoryAccess access;
        access.device = cu.device;
        bool mapped = true;
        for(size_t k = 0; k < pages.size() && mapped; k++){
            loanPages.push_back(pages[k]);
            mapped = cu.unmap(pages[k][0], (size_t)pageGrain) == 0 && cu.map(loanBase + k * pageGrain, (size_t)pageGrain, 0, pages[k][1], 0) == 0;
        }
        if(mapped) mapped = cu.allow(loanBase, (size_t)size, &access, 1) == 0;
        if(!mapped){
            repay();
            problem = "the prompt memory could not be borrowed from the expert cache";
            return false;
        }

        uint64_t at = 0;
        for(auto& part : loanParts(T, W)){
            loanSaved.push_back({part.first, *part.first});
            *part.first = loanBase + at;
            at += (part.second + 255) / 256 * 256;
        }
        loanKept[0] = superLimit;
        loanKept[1] = listSpan;
        loanKept[2] = packs[0].capacity;
        loanKept[3] = fines[0].capacity;
        loanKept[4] = packs[1].capacity;
        loanKept[5] = fines[1].capacity;
        loanKept[6] = crestRows;
        if(W > 0) crestRows = (int)W;
        if(T > 0){
            embedRows = moeOut;
            superLimit = (int)T;
            listSpan = (int)(T * expertsUsed + expertsUsed);
            packs[0].capacity = (int)std::max((uint64_t)crestRows * innerWidth, T * hidden);
            fines[0].capacity = packs[0].capacity;
            packs[1].capacity = (int)((std::max<uint64_t>(batchRows, T) + crestRows) * expertWidth);
            fines[1].capacity = packs[1].capacity;
        }
        loanUsed = (int)std::max(T, W);
        return true;
    }

    void repay() {
        if(!lent) return;
        CudaLink& cu = cudaLink();
        CudaQueue& queue = cudaQueue();
        waitStream();
        if(!loanSaved.empty()){
            for(auto& kept : loanSaved) *kept.first = kept.second;
            superLimit = loanKept[0];
            listSpan = loanKept[1];
            packs[0].capacity = loanKept[2];
            fines[0].capacity = loanKept[3];
            packs[1].capacity = loanKept[4];
            fines[1].capacity = loanKept[5];
            crestRows = loanKept[6];
        }
        CuMemoryAccess access;
        access.device = cu.device;
        bool whole = true;
        for(size_t k = 0; k < loanPages.size(); k++){
            if(loanBase) cu.unmap(loanBase + k * pageGrain, (size_t)pageGrain);
            whole = cu.map(loanPages[k][0], (size_t)pageGrain, 0, loanPages[k][1], 0) == 0
                && cu.allow(loanPages[k][0], (size_t)pageGrain, &access, 1) == 0
                && queue.copyUp(loanPages[k][0], loanStore + k * pageGrain, (size_t)pageGrain, stream) == 0
                && whole;
        }
        whole = waitStream() == 0 && whole;
        if(loanBase) cu.unreserve(loanBase, (size_t)loanSize);
        if(!whole) problem = "the borrowed prompt memory could not go back to the expert cache";
        int touched = -1;
        for(size_t i = 0; i <= loanSlots.size(); i++){
            int l = i < loanSlots.size() ? loanSlots[i][0] : -1;
            if(l != touched && touched >= 0) cu.copyUp(layers[touched].slots, layers[touched].slot.data(), expertCount * 4);
            touched = l;
            if(l < 0) break;
            int place = loanSlots[i][2] - fixedExperts;
            if(whole){
                layers[l].slot[loanSlots[i][1]] = loanSlots[i][2];
                continue;
            }
            if(place < 0 || place >= recentPerLayer) continue;
            recentHolds[(size_t)l * recentPerLayer + place] = -1;
            recentBusy[(size_t)l * recentPerLayer + place] = 1;
        }
        loanSaved.clear();
        loanPages.clear();
        loanSlots.clear();
        loanBase = 0;
        loanSize = 0;
        lent = false;
        liftHold = false;
        exchangeHold = false;
    }

    bool giveBack(uint64_t bytes) {
        if(lent) return false;
        uint64_t perExpert = 0;
        for(int l = firstLayer; l < endLayer; l++) perExpert += expertBytes(l);
        if(poolSpans.empty() || recentPerLayer == 0 || perExpert == 0) return false;
        uint64_t slack = (uint64_t)poolSpans.size() * pageGrain;
        int give = (int)((bytes + slack + perExpert - 1) / perExpert);
        return giveSlots(give);
    }

    bool giveSlots(int give) {
        if(lent || give <= 0) return false;
        if(givenBack + give > recentPerLayer || give > fixedExperts) return false;
        liftHold = true;
        exchangeHold = true;
        auto release = [&](bool fine){
            liftHold = false;
            exchangeHold = false;
            return fine;
        };
        CudaQueue& queue = cudaQueue();
        auto clock = std::chrono::steady_clock::now();
        while(liftPending > 0 && since(clock) < 20000.0){
            refresh();
            if(liftPending > 0) Sleep(1);
        }
        while((!moving.empty() || !leaving.empty()) && since(clock) < 20000.0){
            if(!moving.empty()) queue.eventSync(movedEvent);
            else queue.eventSync(leftEvent);
            exchange(0);
        }
        if(liftPending > 0 || !moving.empty() || !leaving.empty()) return release(false);
        waitStream();
        CudaLink& cu = cudaLink();
        int top = fixedExperts, low = fixedExperts - give;
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
            std::vector<int> owner(top, -1);
            for(int e = 0; e < expertCount; e++){
                if(L.slot[e] >= 0 && L.slot[e] < top) owner[L.slot[e]] = e;
            }
            std::vector<int> order(top);
            std::iota(order.begin(), order.end(), 0);
            const float* use = recent.data() + (size_t)l * expertCount;
            std::stable_sort(order.begin(), order.end(), [&](int x, int y){
                float a = owner[x] >= 0 ? use[owner[x]] : -1.0f, b = owner[y] >= 0 ? use[owner[y]] : -1.0f;
                return a < b || (a == b && x > y);
            });
            std::vector<char> leaves(top, 0);
            int homeless = 0;
            for(int i = 0; i < give; i++){
                int s = order[i];
                leaves[s] = 1;
                int e = owner[s];
                if(e >= 0 && (!L.warmHost[0] || L.warm[e] < 0)) homeless++;
            }
            int roomy = L.warmHost[0] ? std::min(homeless, (int)(ramFree[l].size() + spareFree[l].size())) : 0;
            uint8_t* block = nullptr;
            if(homeless > roomy){
                void* room = nullptr;
                if(cu.hostAlloc(&room, (size_t)((homeless - roomy) * expertBytes(l)), 0) == 0){
                    pinned.push_back(room);
                    block = (uint8_t*)room;
                }
            }
            std::vector<int> open;
            for(int s = 0; s < top; s++){
                if(!leaves[s]) continue;
                int e = owner[s];
                if(e >= 0){
                    keepAside(l, e, (uint64_t)s, block);
                    L.slot[e] = -1;
                }
                owner[s] = -1;
                if(s < low) open.push_back(s);
            }
            size_t next = 0;
            for(int s = low; s < top; s++){
                int e = owner[s];
                if(e < 0) continue;
                int to = open[next++];
                for(int p = 0; p < 3; p++){
                    uint64_t span = (uint64_t)parts[p]->rows * parts[p]->stride;
                    queue.copyAcross(L.resident[p] + (uint64_t)to * span, L.resident[p] + (uint64_t)s * span, (size_t)span, stream);
                }
                L.slot[e] = to;
            }
            for(int j = 0; j < recentPerLayer; j++){
                int e = recentHolds[(size_t)l * recentPerLayer + j];
                if(e < 0) continue;
                for(int p = 0; p < 3; p++){
                    uint64_t span = (uint64_t)parts[p]->rows * parts[p]->stride;
                    queue.copyAcross(L.resident[p] + (uint64_t)(low + j) * span, L.resident[p] + (uint64_t)(top + j) * span, (size_t)span, stream);
                }
                L.slot[e] = low + j;
            }
        }
        fixedExperts = low;
        waitStream();
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            if(cu.copyUp(L.slots, L.slot.data(), expertCount * 4) != 0 || cu.copyUp(L.warmSlots, L.warm.data(), expertCount * 4) != 0) return release(false);
        }
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
            for(int i = 0; i < 3; i++){
                uint64_t kept = (uint64_t)(fixedExperts + recentPerLayer) * parts[i]->rows * parts[i]->stride;
                for(PoolSpan& p : poolSpans){
                    if(p.base != L.resident[i]) continue;
                    for(size_t q = 0; q < p.pages.size(); q++){
                        uint64_t at = p.solid + q * pageGrain;
                        if(!p.pages[q] || at < kept) continue;
                        cu.unmap(p.base + at, (size_t)pageGrain);
                        // la pagina queda para el contexto porque soltarla y pedir otra hace que windows saque memoria de la gpu
                        freePages.push_back(p.pages[q]);
                        p.pages[q] = 0;
                    }
                }
            }
        }
        givenBack += give;
        return release(true);
    }

    bool growContext(int room) {
        uint64_t adding = 0;
        std::vector<char> need;
        for(const Stretch& s : stretches) adding += stretchNeed(s, room, need);
        NvmlLink& nv = nvmlLink();
        unsigned long long physical[3] = {0, 0, 0};
        uint64_t margin = growMargin;
        if(adding > 0 && nv.ready && nv.getMemory(nv.device, physical) == 0 && physical[0] > 0 && physical[1] < adding + margin && !giveBack(adding + margin - physical[1])){
            problem = "the context cannot grow to " + std::to_string(room) + " tokens: the gpu has " + std::to_string(physical[1] >> 20) + " MB free and it needs " + std::to_string(adding >> 20) + " MB more, close other programs that use the gpu or set a VRAM limit";
            return false;
        }
        for(Stretch& s : stretches){
            if(!mapStretch(s, room)) return false;
        }
        kvRoom = room;
        return true;
    }

    VramBudget vramBudget;
    ProcessVram processVram;
    uint64_t budgetAt = 0;
    uint64_t budgetBytes = 0;
    uint64_t usageBytes = 0;
    uint64_t dedicatedBytes = 0;
    uint64_t dedicatedPeak = 0;
    uint64_t movedBytes = 0;
    int movedSeen = 0;

    // cuando otro programa pide unos cientos de mb de vram windows mueve parte del modelo a la ram compartida
    // aunque quede presupuesto, y no la devuelve: cada paso queda casi el doble de lento (medido 82 a 48 tok/s)
    // lo notamos porque la vram dedicada del proceso baja sin que soltemos nada, y el master vuelve a cargar el modelo
    void readBudget() {
        uint64_t now = GetTickCount64();
        if(now - budgetAt < 1000) return;
        budgetAt = now;
        vramBudget.read(budgetBytes, usageBytes);
        if(!processVram.open() || !processVram.read(dedicatedBytes)) return;
        dedicatedPeak = std::max(dedicatedPeak, dedicatedBytes);
        bool dropped = dedicatedPeak > dedicatedBytes + (128ULL << 20);
        movedSeen = dropped ? movedSeen + 1 : 0;
        if(movedSeen >= 2) movedBytes = dedicatedPeak - dedicatedBytes;
    }

    bool reachContext(int tokens) {
        readBudget();
        if(stretches.empty() || tokens <= kvRoom) return true;
        return growContext(std::min(contextLimit, std::max(tokens + 4096, 2 * kvRoom)));
    }

    void stagedRows(const StagedSet& s, int inputStride, const PackedInput& a, const PackedInput& b, int rows, int cols) {
        uint64_t widest = 0;
        bool same = true;
        for(int i = 0; i < s.count; i++){
            widest = std::max(widest, s.rowBytes[i]);
            same = same && s.type[i] == s.type[0];
        }
        int four = (int)((4 * widest + 31) / 16 * 16), two = (int)((2 * widest + 31) / 16 * 16);
        int many = 8 * four <= 49152
            ? 4
            : 8 * two <= 49152
                ? 2
                : 1;
        int t = s.type[0];
        bool typed = same && (t == 7 || t == 8 || t == 12 || t == 13 || (t >= 16 && t <= 18) || (t >= 20 && t <= 23) || t == 42);
        std::string kind = !s.team[0]
            ? "rowsStagedSet"
            : s.input[0]
                ? "rowsTeam"
                : "rowsTeamDown";
        std::string name = many == 1 && !s.team[0]
            ? kind
            : many == 4 && typed
                ? kind + "4t" + std::to_string(t)
                : kind + std::to_string(many);
        int stage = many == 4
            ? four
            : many == 2
                ? two
                : (int)((widest + 31) / 16 * 16);
        goShared(8 * stage, name.c_str(), (unsigned int)((rows + 8 * many - 1) / (8 * many)), (unsigned int)(s.count * s.per), 256, 1, s, inputStride, a.values, a.scales, a.sums, b.values, b.scales, b.sums, rows, cols, stage, tables);
    }

    bool warmReady(const GpuLayer& L) {
        return fine && !exact && L.warmBase[0] != 0;
    }

    void route(GpuLayer& L, int l, int n) {
        bool perLayer = n == 1 || relay;
        uint64_t idsOut = perLayer
            ? mappedIds + 4ULL * stageOf(l) * expertsUsed * n
            : mappedIds;
        uint64_t xOut = perLayer
            ? mappedX + 4ULL * stageOf(l) * hidden * n
            : mappedX;
        bool copied = fine && fusedMix && hidden % 32 == 0;
        if(copied){
            PackedInput& a = packs[0];
            PackedInput& b = fines[0];
            go("quantizeCopy", (unsigned int)((n * hidden / 32 + 7) / 8), 1, 32, 8, x, xOut, n * hidden, a.values, a.scales, a.sums, b.values, b.scales, b.sums);
        } else pack(x, n * hidden, 0);
        if(fusedMix && n <= fewRows && L.router.type == 30 && L.sharedScalar.type == 30 && L.router.cols == L.sharedScalar.cols && hidden % 8 == 0){
            std::string pairName = "plainPair" + std::to_string(n);
            go(pairName.c_str(), (unsigned int)(expertCount + 1), 1, 256, 1, L.router.data, L.sharedScalar.data, L.router.stride, L.sharedScalar.stride, expertCount, expertCount + 1, x, hidden, routerOut, expertCount, scalar, 1, hidden, 0);
        } else {
            plain(L.router, x, hidden, routerOut, expertCount, n, 0);
            plain(L.sharedScalar, x, hidden, scalar, 1, n, 0);
        }
        uint64_t warmUse = warmReady(L) ? L.warmSlots : 0;
        uint64_t plan = lists + 8ULL * listSpan;
        int tail = tailActive() ? tailSkip : 0;
        uint64_t teamOn = teamed(n) ? teamLists : 0;
        uint64_t warmTeamOn = teamOn ? teamLists + teamSpan() : 0;
        bool ahead = relay && foresee && n <= relayLimit && l + 1 < endLayer;
        if(copied && n == 1 && !ahead && expertCount % 32 == 0 && expertCount <= 1024){
            uint64_t flag = relay ? mappedFlags + 64 : 0;
            go("routePlan", 1, 1, 32, 1, routerOut, routeIds, routeWeights, idsOut, L.slots, warmUse, lists, plan, plan + 8ULL * expertCount, plan + 12ULL * expertCount, plan + 16ULL * expertCount, spillCounts + 4ULL * l, warmShare, tail, flag, mappedFlags, l);
            if(timing && relay) go("stampTime", 1, 1, 1, 1, mappedTimeline + 32ULL * l);
            return;
        }
        if(copied && relay && n > 1 && n <= 16 && !ahead && expertCount % 32 == 0 && expertCount <= 1024){
            go("routePlanBatch", 1, 1, 32 * n, 1, routerOut, routeIds, routeWeights, idsOut, L.slots, warmUse, n, warmShare, lists, plan, plan + 4ULL * planWidth, plan + 8ULL * planWidth, plan + 12ULL * planWidth, plan + 16ULL * planWidth, plan + 20ULL * planWidth, spillCounts + 4ULL * l, tail, mappedFlags + 64, mappedFlags, l, teamOn, warmTeamOn);
            if(timing) go("stampTime", 1, 1, 1, 1, mappedTimeline + 32ULL * l);
            return;
        }
        go("routeTop", (unsigned int)n, 1, expertCount, 1, routerOut, routeIds, routeWeights);
        go("copyWords", (unsigned int)((n * expertsUsed + 255) / 256), 1, 256, 1, routeIds, idsOut, n * expertsUsed);
        if(!copied) go("copyWords", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, x, xOut, n * hidden);
        if(n == 1) go("planExperts", 1, 1, 32, 1, routeIds, L.slots, warmUse, lists, plan, plan + 8ULL * expertCount, plan + 12ULL * expertCount, plan + 16ULL * expertCount, spillCounts + 4ULL * l, warmShare, routeWeights, tail);
        if(n > 1 && relay) go("planBatch", 1, 1, 32, 1, routeIds, L.slots, warmUse, n, warmShare, lists, plan, plan + 4ULL * planWidth, plan + 8ULL * planWidth, plan + 12ULL * planWidth, plan + 16ULL * planWidth, plan + 20ULL * planWidth, spillCounts + 4ULL * l, routeWeights, tail, teamOn, warmTeamOn);
        if(relay && foresee && n <= relayLimit && l + 1 < endLayer){
            plain(layers[l + 1].router, x, hidden, routerOut, expertCount, n, 0);
            go("routeTop", (unsigned int)n, 1, expertCount, 1, routerOut, nextIds, nextWeights);
            go("copyWords", (unsigned int)((n * expertsUsed + 255) / 256), 1, 256, 1, nextIds, mappedNext + 4ULL * l * expertsUsed * n, n * expertsUsed);
        }
        if(relay) go("tellHost", 1, 1, 1, 1, mappedFlags + 64, mappedFlags, l);
        if(relay && timing) go("stampTime", 1, 1, 1, 1, mappedTimeline + 32ULL * l);
    }

    bool cpuFeeds(const GpuLayer& L, int n) const {
        return L.hostFfn && relay && hostDenseOut && n <= denseCpuRows;
    }

    void feedForward(GpuLayer& L, int l, int n) {
        if(cpuFeeds(L, n)){
            go("copyWords", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, x, mappedX, n * hidden);
            go("tellHost", 1, 1, 1, 1, mappedFlags + 64, mappedFlags, l);
            go("awaitHost", 1, 1, 1, 1, mappedFlags + 128, mappedFlags, l, spillCounts + 4ULL * l, mappedFlags + 192, mappedFlags + 160);
            go("copyWords", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, mappedDenseOut, shared, n * hidden);
            return;
        }
        if(L.hostFfn && denseStage && n > denseCpuRows){
            GpuMatrix gate = L.sharedGate, up = L.sharedUp, down = L.sharedDown;
            const Matrix* from[3] = {&L.hostGate, &L.hostUp, &L.hostDown};
            GpuMatrix* into[3] = {&gate, &up, &down};
            uint64_t at = denseStage;
            for(int i = 0; i < 3; i++){
                uint64_t bytes = (uint64_t)from[i]->rows * from[i]->stride;
                if(cudaQueue().copyUp(at, from[i]->data, (size_t)bytes, stream) != 0) failed = true;
                into[i]->data = at;
                at += (bytes + 255) / 256 * 256;
            }
            pack(x, n * hidden, 0);
            projectMany({{&gate, sharedGateOut}, {&up, sharedUpOut}}, x, 0, n);
            swigluPack(sharedGateOut, sharedUpOut, sharedHidden, n * expertWidth, 1);
            project(down, sharedHidden, 1, shared, n);
            return;
        }
        pack(x, n * hidden, 0);
        projectMany({{&L.sharedGate, sharedGateOut}, {&L.sharedUp, sharedUpOut}}, x, 0, n);
        swigluPack(sharedGateOut, sharedUpOut, sharedHidden, n * expertWidth, 1);
        project(L.sharedDown, sharedHidden, 1, shared, n);
    }

    void foreseeLayer(int l, const int* guess, int n) {
        GpuLayer& L = layers[l];
        const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
        std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
        int tail = tailActive() ? tailSkip : 0;
        for(int i = 0; i < n * expertsUsed; i++){
            int e = guess[i];
            if(e < 0 || e >= expertCount || (tail > 0 && i % expertsUsed >= tail) || L.slot[e] >= 0 || (L.warmHost[0] && L.warm[e] >= 0)) continue;
            uint32_t& mark = guessMark[(size_t)l * expertCount + e];
            if(mark == ticket) continue;
            mark = ticket;
            for(int p = 0; p < 3; p++){
                uint64_t span = (uint64_t)parts[p]->rows * parts[p]->stride;
                ranges.push_back({(PVOID)(parts[p]->data + (uint64_t)e * span), (SIZE_T)span});
            }
        }
        if(!ranges.empty()) PrefetchVirtualMemory(GetCurrentProcess(), ranges.size(), ranges.data(), 0);
        foreseen += ranges.size() / 3;
    }

    bool startLifting() {
        CudaQueue& queue = cudaQueue();
        uint64_t widest = 0;
        for(int l = firstLayer; l < endLayer; l++) widest = std::max(widest, expertBytes(l));
        liftRoom = 8 * (widest + 3 * 8192);
        void* table = nullptr;
        if(cudaLink().hostAlloc(&table, 2 * layers.size() * expertCount * 4, 0) != 0) return false;
        pinned.push_back(table);
        slotStage = (int*)table;
        warmStage = slotStage + layers.size() * expertCount;
        lifters.resize(liftCrew);
        for(Lifter& own : lifters){
            void* stage = nullptr;
            if(queue.streamCreate(&own.stream, 1) != 0 || queue.eventCreate(&own.event, 3) != 0 || cudaLink().hostAlloc(&stage, (size_t)liftRoom, 0) != 0) return false;
            pinned.push_back(stage);
            own.stage = (uint8_t*)stage;
        }
        for(int i = 0; i < liftCrew; i++) lifters[i].thread = std::thread([this, i]{ liftLoop(lifters[i]); });
        return true;
    }

    void liftLoop(Lifter& own) {
        CudaLink& cu = cudaLink();
        CudaQueue& queue = cudaQueue();
        cu.setCurrent(cu.context);
        std::vector<Lift> batch;
        own.files.assign(file.paths.size(), INVALID_HANDLE_VALUE);
        while(true){
            {
                std::unique_lock<std::mutex> guard(liftLock);
                liftWake.wait(guard, [&]{ return liftStop || !lifting.empty(); });
                if(liftStop) break;
                size_t take = std::min<size_t>(8, lifting.size());
                batch.assign(lifting.end() - take, lifting.end());
                lifting.resize(lifting.size() - take);
            }
            uint64_t used = 0;
            bool sound = true;
            for(const Lift& x : batch){
                GpuLayer& L = layers[x.layer];
                const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
                for(int p = 0; p < 3; p++){
                    uint64_t span = (uint64_t)parts[p]->rows * parts[p]->stride;
                    uint64_t slot = L.resident[p] + (uint64_t)(fixedExperts + x.place) * span;
                    if(x.down){
                        if(queue.copyDown(L.warmHost[p] + (uint64_t)x.room * span, slot, (size_t)span, own.stream) != 0) sound = false;
                        continue;
                    }
                    const uint8_t* source = x.from[p];
                    if(!x.pinnedCopy){
                        HANDLE& h = own.files[x.source[p]];
                        if(h == INVALID_HANDLE_VALUE) h = CreateFileA(file.paths[x.source[p]].c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
                        uint64_t start = x.at[p] & ~4095ULL;
                        uint64_t skip = x.at[p] - start;
                        uint64_t length = (skip + span + 4095) & ~4095ULL;
                        OVERLAPPED where = {};
                        where.Offset = (DWORD)start;
                        where.OffsetHigh = (DWORD)(start >> 32);
                        DWORD got = 0;
                        if(h == INVALID_HANDLE_VALUE || !ReadFile(h, own.stage + used, (DWORD)length, &got, &where) || got < skip + span) sound = false;
                        source = own.stage + used + skip;
                        used += length;
                    }
                    if(queue.copyUp(slot, source, (size_t)span, own.stream) != 0) sound = false;
                }
            }
            sound = queue.eventRecord(own.event, own.stream) == 0 && queue.eventSync(own.event) == 0 && sound;
            for(Lift& x : batch) x.landed = sound;
            {
                std::lock_guard<std::mutex> guard(liftLock);
                lifted.insert(lifted.end(), batch.begin(), batch.end());
            }
        }
        for(HANDLE h : own.files){
            if(h != INVALID_HANDLE_VALUE) CloseHandle(h);
        }
    }

    void touchRecent(int layer, int e) {
        int place = layers[layer].slot[e] - fixedExperts;
        if(place >= 0) recentUsed[(size_t)layer * recentPerLayer + place] = recentTick;
    }

    void wish(int layer, int e) {
        if(recentPerLayer > 0 && !arriving[(size_t)layer * expertCount + e]) wanted.push_back({layer, e});
    }

    void refresh() {
        if(recentPerLayer == 0) return;
        std::vector<Lift> done;
        {
            std::lock_guard<std::mutex> guard(liftLock);
            done.swap(lifted);
        }
        int first = (int)layers.size(), last = -1;
        auto dirty = [&](int layer){
            first = std::min(first, layer);
            last = std::max(last, layer);
        };
        for(const Lift& x : done){
            GpuLayer& L = layers[x.layer];
            size_t at = (size_t)x.layer * recentPerLayer + x.place;
            if(!x.down && !x.pinnedCopy) diskPending--;
            recentBusy[at] = 0;
            arriving[(size_t)x.layer * expertCount + x.expert] = 0;
            if(x.down && !x.landed) ramFree[x.layer].push_back(x.room);
            if(!x.down && x.room >= 0) ramComing[x.layer]--;
            if(!x.landed) continue;
            dirty(x.layer);
            if(x.down){
                L.warm[x.expert] = x.room;
                L.slot[x.expert] = -1;
                recentHolds[at] = -1;
                continue;
            }
            L.slot[x.expert] = fixedExperts + x.place;
            recentHolds[at] = x.expert;
            recentUsed[at] = recentTick;
            liftCount++;
            if(x.room < 0) continue;
            L.warm[x.expert] = -1;
            ramFree[x.layer].push_back(x.room);
        }
        liftPending -= (int)done.size();
        if(liftHold){
            wanted.clear();
            return;
        }
        std::vector<Lift> fresh;
        int fromDisk = 0;
        for(const auto& w : wanted){
            int layer = w.first, e = w.second;
            GpuLayer& L = layers[layer];
            if(L.slot[e] >= 0 || arriving[(size_t)layer * expertCount + e] || liftPending + (int)fresh.size() >= liftLimit) continue;
            bool pinnedCopy = L.warmHost[0] && L.warm[e] >= 0;
            bool held = !pinnedCopy && hostExpert(layer, 0, e, nullptr);
            if(!pinnedCopy && !held && (fromDisk >= diskLifts || diskPending + fromDisk >= diskLimit)) continue;
            int place = -1;
            for(int j = 0; j < recentPerLayer && place < 0; j++){
                size_t at = (size_t)layer * recentPerLayer + j;
                if(recentHolds[at] < 0 && !recentBusy[at]) place = j;
            }
            if(place < 0) continue;
            recentBusy[(size_t)layer * recentPerLayer + place] = 1;
            arriving[(size_t)layer * expertCount + e] = 1;
            const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
            Lift x = {layer, e, place, {}, {}, {}, pinnedCopy || held, false, pinnedCopy ? L.warm[e] : -1, false};
            if(pinnedCopy) ramComing[layer]++;
            else if(!held) fromDisk++;
            for(int p = 0; p < 3; p++){
                uint64_t span = (uint64_t)parts[p]->rows * parts[p]->stride;
                x.from[p] = pinnedCopy
                    ? L.warmHost[p] + (uint64_t)L.warm[e] * span
                    : hostExpert(layer, p, e, nullptr);
                x.at[p] = L.expertTensor[p]->offset + (uint64_t)e * span;
                x.source[p] = (int)L.expertTensor[p]->file;
            }
            fresh.push_back(x);
        }
        wanted.clear();
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            int empty = 0, leaving = 0;
            for(int j = 0; j < recentPerLayer; j++){
                size_t at = (size_t)l * recentPerLayer + j;
                if(recentHolds[at] < 0 && !recentBusy[at]) empty++;
                if(recentHolds[at] >= 0 && recentBusy[at]) leaving++;
            }
            while(empty + leaving < recentSpare){
                int place = -1;
                uint32_t oldest = recentTick;
                for(int j = 0; j < recentPerLayer; j++){
                    size_t at = (size_t)l * recentPerLayer + j;
                    if(recentHolds[at] < 0 || recentBusy[at] || recentUsed[at] >= oldest) continue;
                    oldest = recentUsed[at];
                    place = j;
                }
                if(place < 0) break;
                size_t at = (size_t)l * recentPerLayer + place;
                int e = recentHolds[at];
                bool homed = L.warm[e] >= 0 || (helper.ready && !L.help.empty() && L.help[e] >= 0);
                bool stuck = empty == 0 && leaving == 0 && ramComing[l] == 0;
                if(!homed && ramFree[l].empty() && !stuck) break;
                if(homed || ramFree[l].empty()){
                    L.slot[e] = -1;
                    recentHolds[at] = -1;
                    dirty(l);
                    empty++;
                    continue;
                }
                recentBusy[at] = 1;
                arriving[(size_t)l * expertCount + e] = 1;
                fresh.push_back({l, e, place, {}, {}, {}, false, false, ramFree[l].back(), true});
                ramFree[l].pop_back();
                leaving++;
            }
        }
        recentTick++;
        if(last >= first){
            size_t span = (size_t)(last - first + 1) * expertCount * 4;
            for(int l = first; l <= last; l++){
                memcpy(slotStage + (size_t)l * expertCount, layers[l].slot.data(), expertCount * 4);
                memcpy(warmStage + (size_t)l * expertCount, layers[l].warm.data(), expertCount * 4);
            }
            if(cudaQueue().copyUp(layers[first].slots, slotStage + (size_t)first * expertCount, span, stream) != 0) failed = true;
            if(cudaQueue().copyUp(layers[first].warmSlots, warmStage + (size_t)first * expertCount, span, stream) != 0) failed = true;
        }
        diskPending += fromDisk;
        if(fresh.empty()) return;
        liftPending += (int)fresh.size();
        {
            std::lock_guard<std::mutex> guard(liftLock);
            lifting.insert(lifting.end(), fresh.begin(), fresh.end());
        }
        liftWake.notify_all();
    }

    void gather(GpuLayer& L, int layer, int n) {
        int routes = n * expertsUsed;
        if(relayRows > 1 && n > 1){
            const int* chosen = hostIds + (size_t)stageOf(layer) * expertsUsed * n;
            batchX = hostX + (size_t)stageOf(layer) * hidden * n;
            spillOut = hostExpertOut + (size_t)stageOf(layer) * expertsUsed * n * hidden;
            cpuExperts.clear();
            cpuTokens.clear();
            int kept = 0;
            int tail = tailActive() ? tailSkip : 0;
            int misses = 0;
            for(int i = 0; i < routes; i++){
                int e = chosen[i];
                if(e >= 0 && e < expertCount && L.slot[e] < 0 && !(tail > 0 && i % expertsUsed >= tail)) misses++;
            }
            int wanted = misses > 1 ? (misses - 1) * 3 * warmShare / 32 : 0;
            int sent = 0;
            sides.assign(expertCount, -1);
            for(int t = 0; t < n; t++){
                const int* own = chosen + (size_t)t * expertsUsed;
                int room = wanted - sent;
                int rank = 0, taken = 0;
                for(int k = 0; k < expertsUsed; k++){
                    int e = own[k];
                    if(e < 0 || e >= expertCount){
                        failed = true;
                        continue;
                    }
                    expertHits[(size_t)layer * expertCount + e]++;
                    recent[(size_t)layer * expertCount + e] += 1.0f;
                    learned[(size_t)layer * expertCount + e]++;
                    if(L.slot[e] >= 0){
                        kept++;
                        touchRecent(layer, e);
                        continue;
                    }
                    if(tail > 0 && k >= tail){
                        skippedRoutes++;
                        continue;
                    }
                    wish(layer, e);
                    bool fresh = sides[e] < 0 && warmReady(L) && L.warm[e] >= 0;
                    bool warm = sides[e] == 1 || (fresh && rank < room);
                    rank += fresh ? 1 : 0;
                    sides[e] = warm ? 1 : 0;
                    if(warm){
                        taken++;
                        warmHits++;
                        continue;
                    }
                    cpuExperts.push_back(e);
                    cpuTokens.push_back(t);
                }
                sent += taken;
            }
            cpuRows = (int)cpuExperts.size();
            relayRoutes += routes;
            relayKept += kept;
            relayCpu += cpuRows;
            relayWarm += sent;
            std::set<int> distinct;
            for(int t = 0; t < n; t++){
                for(int k = 0; k < expertsUsed; k++){
                    int e = chosen[(size_t)t * expertsUsed + k];
                    if(e < 0 || e >= expertCount || L.slot[e] >= 0) continue;
                    if(tailActive() && k >= tailSkip) continue;
                    missedRoutes++;
                    distinct.insert(e);
                }
            }
            missedUnique += distinct.size();
            L.lastRouting.assign(chosen + (size_t)(n - 1) * expertsUsed, chosen + (size_t)n * expertsUsed);
            routedCount += routes;
            residentCount += kept;
            return;
        }
        int* assign = hostLists;
        int* input = hostLists + listSpan;
        int* groupSlot = input + listSpan;
        int* groupStart = groupSlot + expertCount;
        int* groupCount = groupStart + expertCount;

        if(n == 1){
            const int* chosen = hostIds + (size_t)layer * expertsUsed;
            spillX = hostX + (size_t)layer * hidden;
            spillOut = hostExpertOut + (size_t)layer * expertsUsed * hidden;
            groups = 0;
            cpuExperts.clear();
            int tail = tailActive() ? tailSkip : 0;
            int missed = 0;
            for(int k = 0; k < expertsUsed; k++){
                int e = chosen[k];
                if(e >= 0 && e < expertCount && L.slot[e] < 0 && !(tail > 0 && k >= tail)) missed++;
            }
            int allowed = missed * warmShare / 8;
            int taken = 0;
            for(int k = 0; k < expertsUsed; k++){
                int e = chosen[k];
                if(e < 0 || e >= expertCount){
                    failed = true;
                    continue;
                }
                expertHits[(size_t)layer * expertCount + e]++;
                recent[(size_t)layer * expertCount + e] += 1.0f;
                learned[(size_t)layer * expertCount + e]++;
                if(L.slot[e] >= 0){
                    groups++;
                    touchRecent(layer, e);
                    continue;
                }
                if(tail > 0 && k >= tail){
                    skippedRoutes++;
                    continue;
                }
                wish(layer, e);
                if(warmReady(L) && L.warm[e] >= 0 && taken < allowed){
                    taken++;
                    warmHits++;
                    continue;
                }
                expertTally[e] = 1;
                expertBegin[e] = 2 * expertsUsed + (int)cpuExperts.size() + 1;
                cpuExperts.push_back(e);
            }
            residentRows = groups;
            cpuBase = 2 * expertsUsed;
            cpuRows = (int)cpuExperts.size();
            L.lastRouting.assign(chosen, chosen + expertsUsed);
            routedCount += expertsUsed;
            residentCount += groups;
            decodeRouted += expertsUsed;
            decodeResident += groups;
            return;
        }

        std::fill(expertTally.begin(), expertTally.end(), 0);
        for(int i = 0; i < routes; i++){
            int e = hostIds[i];
            if(e < 0 || e >= expertCount){
                failed = true;
                e = 0;
                hostIds[i] = 0;
            }
            expertTally[e]++;
        }

        groups = 0;
        widest = 1;
        int cursor = 0;
        for(int e = 0; e < expertCount; e++){
            if(!expertTally[e] || L.slot[e] < 0) continue;
            expertBegin[e] = cursor;
            groupSlot[groups] = L.slot[e];
            groupStart[groups] = cursor;
            groupCount[groups] = expertTally[e];
            widest = std::max(widest, expertTally[e]);
            groups++;
            cursor += expertTally[e];
        }
        residentRows = cursor;
        for(int g = groups; g < expertsUsed; g++){
            groupSlot[g] = 0;
            groupStart[g] = 0;
            groupCount[g] = 0;
        }

        cpuBase = residentRows;
        cpuExperts.clear();
        for(int e = 0; e < expertCount; e++){
            if(!expertTally[e] || L.slot[e] >= 0) continue;
            expertBegin[e] = cursor;
            cpuExperts.push_back(e);
            cursor += expertTally[e];
        }
        cpuRows = cursor - cpuBase;

        for(int e = 0; e < expertCount; e++){
            expertHits[(size_t)layer * expertCount + e] += expertTally[e];
            if(n > verifyLimit) continue;
            recent[(size_t)layer * expertCount + e] += (float)expertTally[e];
            learned[(size_t)layer * expertCount + e] += expertTally[e];
        }
        L.lastRouting.assign(hostIds + (size_t)(n - 1) * expertsUsed, hostIds + (size_t)n * expertsUsed);
        routedCount += routes;
        residentCount += residentRows;

        for(int i = 0; i < routes; i++){
            int a = expertBegin[hostIds[i]]++;
            assign[i] = a;
            input[a] = i / expertsUsed;
        }
    }

    void resident(GpuLayer& L, int n) {
        StagedSet set = {};
        auto add = [&](int type, uint64_t weights, uint64_t rowBytes, uint64_t span, uint64_t slots, uint64_t used, uint64_t inputs, uint64_t out, int shift){
            int i = set.count++;
            set.type[i] = type;
            set.shift[i] = shift;
            set.weights[i] = weights;
            set.rowBytes[i] = rowBytes;
            set.expertBytes[i] = span;
            set.slot[i] = slots;
            set.used[i] = used;
            set.input[i] = inputs;
            set.output[i] = out;
        };
        projectMany({{&L.sharedGate, sharedGateOut}, {&L.sharedUp, sharedUpOut}}, x, 0, n);
        if(relay && n > 1){
            swigluPack(sharedGateOut, sharedUpOut, sharedHidden, n * expertWidth, 1);
            project(L.sharedDown, sharedHidden, 1, shared, n);
            int count = expertsUsed * n;
            uint64_t plan = lists + 8ULL * listSpan;
            uint64_t slotOnGpu = plan;
            uint64_t inputOnGpu = plan + 4ULL * planWidth;
            uint64_t countOnGpu = plan + 8ULL * planWidth;
            uint64_t warmSlotOnGpu = plan + 12ULL * planWidth;
            uint64_t warmCountOnGpu = plan + 16ULL * planWidth;
            uint64_t warmInputOnGpu = plan + 20ULL * planWidth;
            uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride;
            uint64_t upSpan = (uint64_t)L.expertUp.rows * L.expertUp.stride;
            uint64_t downSpan = (uint64_t)L.expertDown.rows * L.expertDown.stride;
            PackedInput& a = packs[0];
            PackedInput b = residual ? fines[0] : PackedInput();
            int stage = (int)((L.expertGate.stride + 31) / 16 * 16);
            uint64_t shift = 4ULL * count * expertWidth;
            if(stagedSets){
                set.per = count;
                add((int)L.expertGate.type, L.resident[0], L.expertGate.stride, gateSpan, slotOnGpu, countOnGpu, inputOnGpu, expertGateOut, 0);
                add((int)L.expertUp.type, L.resident[1], L.expertUp.stride, upSpan, slotOnGpu, countOnGpu, inputOnGpu, expertUpOut, 0);
                if(warmReady(L)){
                    add((int)L.expertGate.type, L.warmBase[0], L.expertGate.stride, gateSpan, warmSlotOnGpu, warmCountOnGpu, warmInputOnGpu, expertGateOut + shift, 0);
                    add((int)L.expertUp.type, L.warmBase[1], L.expertUp.stride, upSpan, warmSlotOnGpu, warmCountOnGpu, warmInputOnGpu, expertUpOut + shift, 0);
                }
                if(teamed(n)){
                    set.team[0] = set.team[1] = teamLists;
                    set.team[2] = set.team[3] = teamLists + teamSpan();
                }
                stagedRows(set, hidden, a, b, expertWidth, hidden);
            } else {
                goShared(8 * stage, "rowsRouted", expertWidth / 8, (unsigned int)count, 256, 1, (int)L.expertGate.type, L.resident[0], L.expertGate.stride, gateSpan, slotOnGpu, countOnGpu, inputOnGpu, a.values, a.scales, a.sums, b.values, b.scales, b.sums, expertGateOut, expertWidth, hidden, stage, tables);
                goShared(8 * stage, "rowsRouted", expertWidth / 8, (unsigned int)count, 256, 1, (int)L.expertUp.type, L.resident[1], L.expertUp.stride, upSpan, slotOnGpu, countOnGpu, inputOnGpu, a.values, a.scales, a.sums, b.values, b.scales, b.sums, expertUpOut, expertWidth, hidden, stage, tables);
            }
            if(warmReady(L) && !stagedSets){
                goShared(8 * stage, "rowsRouted", expertWidth / 8, (unsigned int)count, 256, 1, (int)L.expertGate.type, L.warmBase[0], L.expertGate.stride, gateSpan, warmSlotOnGpu, warmCountOnGpu, warmInputOnGpu, a.values, a.scales, a.sums, b.values, b.scales, b.sums, expertGateOut + shift, expertWidth, hidden, stage, tables);
                goShared(8 * stage, "rowsRouted", expertWidth / 8, (unsigned int)count, 256, 1, (int)L.expertUp.type, L.warmBase[1], L.expertUp.stride, upSpan, warmSlotOnGpu, warmCountOnGpu, warmInputOnGpu, a.values, a.scales, a.sums, b.values, b.scales, b.sums, expertUpOut + shift, expertWidth, hidden, stage, tables);
            }
            swigluPack(expertGateOut, expertUpOut, expertHidden, 2 * count * expertWidth, 1);
            PackedInput& c = packs[1];
            PackedInput d = residual ? fines[1] : PackedInput();
            int downStage = (int)((L.expertDown.stride + 31) / 16 * 16);
            if(stagedSets){
                set = {};
                set.per = count;
                add((int)L.expertDown.type, L.resident[2], L.expertDown.stride, downSpan, slotOnGpu, countOnGpu, 0, expertOut, 0);
                if(warmReady(L)) add((int)L.expertDown.type, L.warmBase[2], L.expertDown.stride, downSpan, warmSlotOnGpu, warmCountOnGpu, 0, expertOut + 4ULL * count * hidden, count);
                if(teamed(n)){
                    set.team[0] = teamLists;
                    set.team[1] = teamLists + teamSpan();
                }
                stagedRows(set, expertWidth, c, d, hidden, expertWidth);
                return;
            }
            goShared(8 * downStage, "rowsStaged", hidden / 8, (unsigned int)count, 256, 1, (int)L.expertDown.type, L.resident[2], L.expertDown.stride, downSpan, slotOnGpu, countOnGpu, expertWidth, c.values, c.scales, c.sums, d.values, d.scales, d.sums, expertOut, hidden, expertWidth, downStage, tables);
            if(warmReady(L)){
                uint64_t values = (uint64_t)count * expertWidth;
                uint64_t scales = 4ULL * count * (expertWidth / 32);
                goShared(8 * downStage, "rowsStaged", hidden / 8, (unsigned int)count, 256, 1, (int)L.expertDown.type, L.warmBase[2], L.expertDown.stride, downSpan, warmSlotOnGpu, warmCountOnGpu, expertWidth, c.values + values, c.scales + scales, c.sums + scales, d.values ? d.values + values : 0, d.scales ? d.scales + scales : 0, d.sums ? d.sums + scales : 0, expertOut + 4ULL * count * hidden, hidden, expertWidth, downStage, tables);
            }
            return;
        }
        int words = 2 * listSpan + 3 * expertCount;
        if(n > 1) go("copyWords", (unsigned int)((words + 255) / 256), 1, 256, 1, mappedLists, lists, words);
        swigluPack(sharedGateOut, sharedUpOut, sharedHidden, n * expertWidth, 1);
        project(L.sharedDown, sharedHidden, 1, shared, n);

        int count = n == 1 && fine && !exact
            ? expertsUsed
            : groups;
        int rows = n == 1
            ? 2 * expertsUsed
            : residentRows;
        if(count == 0) return;

        uint64_t inputOnGpu = lists + 4ULL * listSpan;
        uint64_t groupSlotOnGpu = inputOnGpu + 4ULL * listSpan;
        uint64_t groupStartOnGpu = groupSlotOnGpu + 4ULL * expertCount;
        uint64_t groupCountOnGpu = groupStartOnGpu + 4ULL * expertCount;
        uint64_t warmSlotOnGpu = groupSlotOnGpu + 12ULL * expertCount;
        uint64_t warmCountOnGpu = groupSlotOnGpu + 16ULL * expertCount;
        uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride;
        uint64_t upSpan = (uint64_t)L.expertUp.rows * L.expertUp.stride;
        uint64_t downSpan = (uint64_t)L.expertDown.rows * L.expertDown.stride;
        uint64_t none = 0;

        PackedInput& a = packs[0];
        PackedInput b = fine && residual ? fines[0] : PackedInput();
        bool together = stagedSets && n == 1 && fine && !exact;
        if(together){
            set.per = expertsUsed;
            add((int)L.expertGate.type, L.resident[0], L.expertGate.stride, gateSpan, groupSlotOnGpu, groupCountOnGpu, 0, expertGateOut, 0);
            add((int)L.expertUp.type, L.resident[1], L.expertUp.stride, upSpan, groupSlotOnGpu, groupCountOnGpu, 0, expertUpOut, 0);
            if(warmReady(L)){
                uint64_t shift = 4ULL * expertsUsed * expertWidth;
                add((int)L.expertGate.type, L.warmBase[0], L.expertGate.stride, gateSpan, warmSlotOnGpu, warmCountOnGpu, 0, expertGateOut + shift, 0);
                add((int)L.expertUp.type, L.warmBase[1], L.expertUp.stride, upSpan, warmSlotOnGpu, warmCountOnGpu, 0, expertUpOut + shift, 0);
            }
            stagedRows(set, 0, a, b, expertWidth, hidden);
        } else if(n == 1 && exact){
            if(!launchExperts(module, (int)L.expertGate.type, L.resident[0], L.expertGate.stride, gateSpan, groupSlotOnGpu, count, x, 0, expertGateOut, expertWidth, hidden, tables)) failed = true;
            if(!launchExperts(module, (int)L.expertUp.type, L.resident[1], L.expertUp.stride, upSpan, groupSlotOnGpu, count, x, 0, expertUpOut, expertWidth, hidden, tables)) failed = true;
        } else if(n == 1 && !fine){
            if(!launchExpertsQ8(module, (int)L.expertGate.type, L.resident[0], L.expertGate.stride, gateSpan, groupSlotOnGpu, count, x, a, 0, expertGateOut, expertWidth, hidden, tables)) failed = true;
            if(!launchExpertsQ8(module, (int)L.expertUp.type, L.resident[1], L.expertUp.stride, upSpan, groupSlotOnGpu, count, x, a, 0, expertUpOut, expertWidth, hidden, tables)) failed = true;
        } else if(n == 1){
            int stage = (int)((L.expertGate.stride + 31) / 16 * 16);
            goShared(8 * stage, "rowsStaged", expertWidth / 8, (unsigned int)count, 256, 1, (int)L.expertGate.type, L.resident[0], L.expertGate.stride, gateSpan, groupSlotOnGpu, groupCountOnGpu, 0, a.values, a.scales, a.sums, b.values, b.scales, b.sums, expertGateOut, expertWidth, hidden, stage, tables);
            goShared(8 * stage, "rowsStaged", expertWidth / 8, (unsigned int)count, 256, 1, (int)L.expertUp.type, L.resident[1], L.expertUp.stride, upSpan, groupSlotOnGpu, groupCountOnGpu, 0, a.values, a.scales, a.sums, b.values, b.scales, b.sums, expertUpOut, expertWidth, hidden, stage, tables);
            if(warmReady(L)){
                uint64_t shift = 4ULL * expertsUsed * expertWidth;
                goShared(8 * stage, "rowsStaged", expertWidth / 8, (unsigned int)expertsUsed, 256, 1, (int)L.expertGate.type, L.warmBase[0], L.expertGate.stride, gateSpan, warmSlotOnGpu, warmCountOnGpu, 0, a.values, a.scales, a.sums, b.values, b.scales, b.sums, expertGateOut + shift, expertWidth, hidden, stage, tables);
                goShared(8 * stage, "rowsStaged", expertWidth / 8, (unsigned int)expertsUsed, 256, 1, (int)L.expertUp.type, L.warmBase[1], L.expertUp.stride, upSpan, warmSlotOnGpu, warmCountOnGpu, 0, a.values, a.scales, a.sums, b.values, b.scales, b.sums, expertUpOut + shift, expertWidth, hidden, stage, tables);
            }
        } else {
            bool wide = rows >= wideLeast * count;
            int width = tileWidth(wide);
            int tiles = (widest + width - 1) / width;
            PackedInput e = batchFine ? b : PackedInput();
            go(tiled(L.expertGate.type, wide, e.values).c_str(), (expertWidth + rowTile(L.expertGate.type, wide, e.values) - 1) / rowTile(L.expertGate.type, wide, e.values), (unsigned int)(count * tiles), width == 128 ? 256 : 128, 1, (int)L.expertGate.type, L.resident[0], L.expertGate.stride, gateSpan, groupSlotOnGpu, groupStartOnGpu, groupCountOnGpu, inputOnGpu, a.values, a.scales, a.sums, e.values, e.scales, e.sums, expertGateOut, expertWidth, hidden, 0, tiles, none, none, none, tables);
            go(tiled(L.expertUp.type, wide, e.values).c_str(), (expertWidth + rowTile(L.expertUp.type, wide, e.values) - 1) / rowTile(L.expertUp.type, wide, e.values), (unsigned int)(count * tiles), width == 128 ? 256 : 128, 1, (int)L.expertUp.type, L.resident[1], L.expertUp.stride, upSpan, groupSlotOnGpu, groupStartOnGpu, groupCountOnGpu, inputOnGpu, a.values, a.scales, a.sums, e.values, e.scales, e.sums, expertUpOut, expertWidth, hidden, 0, tiles, none, none, none, tables);
        }
        swigluPack(expertGateOut, expertUpOut, expertHidden, rows * expertWidth, 1);

        PackedInput& c = packs[1];
        PackedInput d = fine && residual ? fines[1] : PackedInput();
        if(together){
            set = {};
            set.per = expertsUsed;
            add((int)L.expertDown.type, L.resident[2], L.expertDown.stride, downSpan, groupSlotOnGpu, groupCountOnGpu, 0, expertOut, 0);
            if(warmReady(L)) add((int)L.expertDown.type, L.warmBase[2], L.expertDown.stride, downSpan, warmSlotOnGpu, warmCountOnGpu, 0, expertOut + 4ULL * expertsUsed * hidden, expertsUsed);
            stagedRows(set, expertWidth, c, d, hidden, expertWidth);
        } else if(n == 1 && exact){
            if(!launchExperts(module, (int)L.expertDown.type, L.resident[2], L.expertDown.stride, downSpan, groupSlotOnGpu, count, expertHidden, expertWidth, expertOut, hidden, expertWidth, tables)) failed = true;
        } else if(n == 1 && !fine){
            if(!launchExpertsQ8(module, (int)L.expertDown.type, L.resident[2], L.expertDown.stride, downSpan, groupSlotOnGpu, count, expertHidden, c, expertWidth, expertOut, hidden, expertWidth, tables)) failed = true;
        } else if(n == 1){
            int stage = (int)((L.expertDown.stride + 31) / 16 * 16);
            goShared(8 * stage, "rowsStaged", hidden / 8, (unsigned int)count, 256, 1, (int)L.expertDown.type, L.resident[2], L.expertDown.stride, downSpan, groupSlotOnGpu, groupCountOnGpu, expertWidth, c.values, c.scales, c.sums, d.values, d.scales, d.sums, expertOut, hidden, expertWidth, stage, tables);
            if(warmReady(L)){
                uint64_t values = (uint64_t)expertsUsed * expertWidth;
                uint64_t scales = 4ULL * expertsUsed * (expertWidth / 32);
                goShared(8 * stage, "rowsStaged", hidden / 8, (unsigned int)expertsUsed, 256, 1, (int)L.expertDown.type, L.warmBase[2], L.expertDown.stride, downSpan, warmSlotOnGpu, warmCountOnGpu, expertWidth, c.values + values, c.scales + scales, c.sums + scales, d.values ? d.values + values : 0, d.scales ? d.scales + scales : 0, d.sums ? d.sums + scales : 0, expertOut + 4ULL * expertsUsed * hidden, hidden, expertWidth, stage, tables);
            }
        } else {
            bool wide = rows >= wideLeast * count;
            int width = tileWidth(wide);
            int tiles = (widest + width - 1) / width;
            go(tiled(L.expertDown.type, wide, batchFine ? d.values : 0).c_str(), (hidden + rowTile(L.expertDown.type, wide, batchFine ? d.values : 0) - 1) / rowTile(L.expertDown.type, wide, batchFine ? d.values : 0), (unsigned int)(count * tiles), width == 128 ? 256 : 128, 1, (int)L.expertDown.type, L.resident[2], L.expertDown.stride, downSpan, groupSlotOnGpu, groupStartOnGpu, groupCountOnGpu, none, c.values, c.scales, c.sums, batchFine ? d.values : none, batchFine ? d.scales : none, batchFine ? d.sums : none, expertOut, hidden, expertWidth, 0, tiles, none, none, none, tables);
        }
    }

    void prefetchExperts(GpuLayer& L) {
        std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
        const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
        size_t layer = (size_t)(&L - layers.data());
        int last = -1;
        for(int e : cpuExperts){
            if(e == last || (L.warmHost[0] && L.warm[e] >= 0) || (helper.ready && !L.help.empty() && L.help[e] >= 0) || hostExpert((int)layer, 0, e, nullptr)) continue;
            last = e;
            if(relay || relayRows > 0) foreseenUsed += guessMark.size() > layer * expertCount + e && guessMark[layer * expertCount + e] == ticket ? 1 : 0;
            for(int p = 0; p < 3; p++){
                uint64_t span = (uint64_t)parts[p]->rows * parts[p]->stride;
                ranges.push_back({(PVOID)(parts[p]->data + (uint64_t)e * span), (SIZE_T)span});
            }
        }
        if(!ranges.empty()) PrefetchVirtualMemory(GetCurrentProcess(), ranges.size(), ranges.data(), 0);
        coldExperts += ranges.size() / 3;
    }

    void spill(GpuLayer& L, int n) {
        if(cpuRows == 0) return;
        bool batch = relayRows > 1 && n > 1;
        int rows = 0;
        std::vector<char> aside;
        if(helper.ready && !L.help.empty() && (n == 1 || batch)){
            aside.assign(cpuRows, 0);
            for(int c = 0; c < cpuRows && rows < helper.capacity; c++){
                int s = L.help[cpuExperts[c]];
                if(s < 0) continue;
                const float* from = batch
                    ? batchX + (size_t)cpuTokens[c] * hidden
                    : spillX;
                memcpy(helper.inRows + (size_t)rows * hidden, from, sizeof(float) * hidden);
                helper.slotList[rows] = s;
                helper.where[rows] = c;
                aside[c] = 1;
                rows++;
            }
        }
        auto launchClock = std::chrono::steady_clock::now();
        bool launched = rows == 0 || helperLaunch(L, rows);
        if(rows > 0){
            helperLaunchMs += since(launchClock);
            helperCalls++;
        }
        if(!launched){
            helper.problem = helper.name + " stopped taking work, its experts went back to the cpu";
            helper.ready = false;
            rows = 0;
            aside.clear();
        }
        auto cpuClock = std::chrono::steady_clock::now();
        spillCpu(L, n, aside);
        if(rows > 0) helperCpuMs += since(cpuClock);
        if(rows == 0) return;

        CudaLink& cu = cudaLink();
        auto clock = std::chrono::steady_clock::now();
        cu.setCurrent(helper.context);
        int code = 0;
        bool late = false;
        while(helper.done){
            int state = cudaQueue().eventQuery(helper.done);
            if(state != 600){
                code = state;
                break;
            }
            if(since(clock) > helperLimit){
                late = true;
                break;
            }
            _mm_pause();
        }
        if(!helper.done) code = cu.sync();
        cu.setCurrent(cu.context);
        helperWaitMs += since(clock);
        if(late){
            helper.problem = helper.name + " took more than " + std::to_string((int)helperLimit) + " ms on one layer, its experts went back to the cpu";
            helper.ready = false;
            std::vector<char> rest(cpuRows, 1);
            for(int i = 0; i < rows; i++) rest[helper.where[i]] = 0;
            spillCpu(L, n, rest);
            return;
        }
        if(code == 0 && (profile || timeHelper) && helper.marks[4]){
            cu.setCurrent(helper.context);
            for(int i = 0; i < 4; i++){
                float part = 0.0f;
                if(cudaQueue().eventElapsed(&part, helper.marks[i], helper.marks[i + 1]) == 0) helper.partMs[i] += part;
            }
            cu.setCurrent(cu.context);
        }
        if(code != 0){
            helper.problem = helper.name + " failed with cuda error " + std::to_string(code) + ", its experts went back to the cpu";
            helper.ready = false;
            std::vector<char> rest(cpuRows, 1);
            for(int i = 0; i < rows; i++) rest[helper.where[i]] = 0;
            spillCpu(L, n, rest);
            return;
        }
        for(int i = 0; i < rows; i++) memcpy(spillOut + (size_t)helper.where[i] * hidden, helper.outRows + (size_t)i * hidden, sizeof(float) * hidden);
        helper.rows += rows;
        expertMs += since(clock);
    }

    void spillCpu(GpuLayer& L, int n, const std::vector<char>& aside) {
        if(cpuRows == 0) return;
        auto clock = std::chrono::steady_clock::now();
        int layer = (int)(&L - layers.data());
        bool skipping = !aside.empty();
        prefetchExperts(L);
        if(relayRows > 1 && n > 1){
            for(int c = 0; c < (int)cpuExperts.size(); c++){
                if(!skipping || !aside[c]) touch(layer, cpuExperts[c]);
            }
            uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride;
            uint64_t upSpan = (uint64_t)L.expertUp.rows * L.expertUp.stride;
            uint64_t downSpan = (uint64_t)L.expertDown.rows * L.expertDown.stride;
            std::vector<int> order;
            for(int c = 0; c < cpuRows; c++){
                if(!skipping || !aside[c]) order.push_back(c);
            }
            std::stable_sort(order.begin(), order.end(), [&](int a, int b){ return cpuExperts[a] < cpuExperts[b]; });
            int rows = (int)order.size();
            cpuInput.resize((size_t)rows * hidden);
            for(int i = 0; i < rows; i++) memcpy(cpuInput.data() + (size_t)i * hidden, batchX + (size_t)cpuTokens[order[i]] * hidden, sizeof(float) * hidden);
            cpuGate.resize((size_t)rows * expertWidth);
            cpuUp.resize((size_t)rows * expertWidth);
            hostInput.fill(cpuInput.data(), rows * hidden);
            std::vector<HostProduct> work;
            for(int i = 0; i < rows;){
                int e = cpuExperts[order[i]];
                int k = 1;
                while(i + k < rows && cpuExperts[order[i + k]] == e) k++;
                bool copied = L.warmHost[0] && L.warm[e] >= 0;
                Matrix g = L.expertGate, up = L.expertUp;
                g.data = copied ? L.warmHost[0] + (uint64_t)L.warm[e] * gateSpan : hostExpert(layer, 0, e, g.data + (uint64_t)e * gateSpan);
                up.data = copied ? L.warmHost[1] + (uint64_t)L.warm[e] * upSpan : hostExpert(layer, 1, e, up.data + (uint64_t)e * upSpan);
                work.push_back({g, &hostInput, i, cpuGate.data() + (size_t)i * expertWidth, k});
                work.push_back({up, &hostInput, i, cpuUp.data() + (size_t)i * expertWidth, k});
                i += k;
            }
            hostMultiply(pool, work, residual);
            hostSwiglu(cpuGate.data(), cpuUp.data(), cpuGate.size());
            hostHidden.fill(cpuGate.data(), rows * expertWidth);
            work.clear();
            for(int i = 0; i < rows;){
                int e = cpuExperts[order[i]];
                int k = 1;
                while(i + k < rows && cpuExperts[order[i + k]] == e) k++;
                bool copied = L.warmHost[0] && L.warm[e] >= 0;
                Matrix d = L.expertDown;
                d.data = copied ? L.warmHost[2] + (uint64_t)L.warm[e] * downSpan : hostExpert(layer, 2, e, d.data + (uint64_t)e * downSpan);
                work.push_back({d, &hostHidden, i, cpuInput.data() + (size_t)i * hidden, k});
                i += k;
            }
            hostMultiply(pool, work, residual);
            for(int i = 0; i < rows; i++) memcpy(spillOut + (size_t)order[i] * hidden, cpuInput.data() + (size_t)i * hidden, sizeof(float) * hidden);
            expertMs += since(clock);
            spilledRows += cpuRows;
            return;
        }
        for(int c = 0; c < (int)cpuExperts.size(); c++){
            if(!skipping || !aside[c]) touch(layer, cpuExperts[c]);
        }
        bool single = n == 1;
        float* target = single ? spillOut : hostExpertOut;
        int* input = hostLists + listSpan;
        cpuGate.resize((size_t)cpuRows * expertWidth);
        cpuUp.resize((size_t)cpuRows * expertWidth);

        uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride;
        uint64_t upSpan = (uint64_t)L.expertUp.rows * L.expertUp.stride;
        uint64_t downSpan = (uint64_t)L.expertDown.rows * L.expertDown.stride;
        if(single && hostInt8 && hostDotType(L.expertGate.type) && hostDotType(L.expertDown.type)){
            hostInput.fill(spillX, hidden);
            std::vector<HostProduct> work;
            for(int c = 0; c < cpuRows; c++){
                if(skipping && aside[c]) continue;
                int e = cpuExperts[c];
                bool copied = L.warmHost[0] && L.warm[e] >= 0;
                Matrix g = L.expertGate, up = L.expertUp;
                g.data = copied ? L.warmHost[0] + (uint64_t)L.warm[e] * gateSpan : hostExpert(layer, 0, e, g.data + (uint64_t)e * gateSpan);
                up.data = copied ? L.warmHost[1] + (uint64_t)L.warm[e] * upSpan : hostExpert(layer, 1, e, up.data + (uint64_t)e * upSpan);
                work.push_back({g, &hostInput, 0, cpuGate.data() + (size_t)c * expertWidth, 1});
                work.push_back({up, &hostInput, 0, cpuUp.data() + (size_t)c * expertWidth, 1});
            }
            hostMultiply(pool, work, residual);
            hostSwiglu(cpuGate.data(), cpuUp.data(), cpuGate.size());
            hostHidden.fill(cpuGate.data(), cpuRows * expertWidth);
            work.clear();
            for(int c = 0; c < cpuRows; c++){
                if(skipping && aside[c]) continue;
                int e = cpuExperts[c];
                bool copied = L.warmHost[0] && L.warm[e] >= 0;
                Matrix d = L.expertDown;
                d.data = copied ? L.warmHost[2] + (uint64_t)L.warm[e] * downSpan : hostExpert(layer, 2, e, d.data + (uint64_t)e * downSpan);
                work.push_back({d, &hostHidden, c, target + (size_t)c * hidden, 1});
            }
            hostMultiply(pool, work, residual);
            expertMs += since(clock);
            return;
        }

        cpuInput.resize((size_t)cpuRows * hidden);
        for(int a = 0; a < cpuRows; a++){
            const float* row = single
                ? spillX
                : hostX + (size_t)input[cpuBase + a] * hidden;
            memcpy(cpuInput.data() + (size_t)a * hidden, row, sizeof(float) * hidden);
        }

        if(hostInt8 && hostDotType(L.expertGate.type) && hostDotType(L.expertUp.type) && hostDotType(L.expertDown.type)){
            hostInput.fill(cpuInput.data(), cpuRows * hidden);
            std::vector<HostProduct> work;
            for(int e : cpuExperts){
                int start = expertBegin[e] - expertTally[e] - cpuBase;
                if(skipping && aside[start]) continue;
                bool copied = L.warmHost[0] && L.warm[e] >= 0;
                Matrix g = L.expertGate, up = L.expertUp;
                g.data = copied ? L.warmHost[0] + (uint64_t)L.warm[e] * gateSpan : hostExpert(layer, 0, e, g.data + (uint64_t)e * gateSpan);
                up.data = copied ? L.warmHost[1] + (uint64_t)L.warm[e] * upSpan : hostExpert(layer, 1, e, up.data + (uint64_t)e * upSpan);
                work.push_back({g, &hostInput, start, cpuGate.data() + (size_t)start * expertWidth, expertTally[e]});
                work.push_back({up, &hostInput, start, cpuUp.data() + (size_t)start * expertWidth, expertTally[e]});
            }
            hostMultiply(pool, work, residual);
            hostSwiglu(cpuGate.data(), cpuUp.data(), cpuGate.size());
            hostHidden.fill(cpuGate.data(), cpuRows * expertWidth);
            work.clear();
            for(int e : cpuExperts){
                int start = expertBegin[e] - expertTally[e] - cpuBase;
                if(skipping && aside[start]) continue;
                bool copied = L.warmHost[0] && L.warm[e] >= 0;
                Matrix d = L.expertDown;
                d.data = copied ? L.warmHost[2] + (uint64_t)L.warm[e] * downSpan : hostExpert(layer, 2, e, d.data + (uint64_t)e * downSpan);
                work.push_back({d, &hostHidden, start, target + (size_t)start * hidden, expertTally[e]});
            }
            hostMultiply(pool, work, residual);
            expertMs += since(clock);
            return;
        }

        std::vector<Product> list;
        for(int e : cpuExperts){
            int start = expertBegin[e] - expertTally[e] - cpuBase;
            if(skipping && aside[start]) continue;
            Matrix g = L.expertGate, up = L.expertUp;
            bool copied = L.warmHost[0] && L.warm[e] >= 0;
            g.data = copied ? L.warmHost[0] + (uint64_t)L.warm[e] * gateSpan : hostExpert(layer, 0, e, g.data + (uint64_t)e * gateSpan);
            up.data = copied ? L.warmHost[1] + (uint64_t)L.warm[e] * upSpan : hostExpert(layer, 1, e, up.data + (uint64_t)e * upSpan);
            list.push_back({g, cpuInput.data() + (size_t)start * hidden, cpuGate.data() + (size_t)start * expertWidth, expertTally[e]});
            list.push_back({up, cpuInput.data() + (size_t)start * hidden, cpuUp.data() + (size_t)start * expertWidth, expertTally[e]});
        }
        multiply(pool, list);

        hostSwiglu(cpuGate.data(), cpuUp.data(), cpuGate.size());

        list.clear();
        for(int e : cpuExperts){
            int start = expertBegin[e] - expertTally[e] - cpuBase;
            if(skipping && aside[start]) continue;
            Matrix d = L.expertDown;
            bool copied = L.warmHost[0] && L.warm[e] >= 0;
            d.data = copied ? L.warmHost[2] + (uint64_t)L.warm[e] * downSpan : hostExpert(layer, 2, e, d.data + (uint64_t)e * downSpan);
            list.push_back({d, cpuGate.data() + (size_t)start * expertWidth, target + (size_t)start * hidden, expertTally[e]});
        }
        multiply(pool, list);
        expertMs += since(clock);
    }

    void finish(int n, int l) {
        if(dense){
            go("writeBack", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, stack, shared, inj, n);
            return;
        }
        bool perLayer = n == 1 || relay;
        int spillBase = perLayer ? 2 * expertsUsed * n : 1 << 30;
        uint64_t spillFrom = perLayer
            ? mappedExpertOut + 4ULL * stageOf(l) * expertsUsed * n * hidden
            : mappedExpertOut;
        if(!perLayer && cpuRows > 0) go("copyWords", (unsigned int)((cpuRows * hidden + 255) / 256), 1, 256, 1, mappedExpertOut, expertOut + 4ULL * cpuBase * hidden, cpuRows * hidden);
        if(relay && timing) go("stampTime", 1, 1, 1, 1, mappedTimeline + 32ULL * l + 8);
        if(relay) go("awaitHost", 1, 1, 1, 1, mappedFlags + 128, mappedFlags, l, spillCounts + 4ULL * l, mappedFlags + 192, mappedFlags + 160);
        if(relay && timing) go("stampTime", 1, 1, 1, 1, mappedTimeline + 32ULL * l + 16);
        go("writeExperts", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, stack, shared, scalar, expertOut, spillFrom, spillBase, routeWeights, lists, inj, n);
    }

    void front(GpuLayer& L, int l, int n, int top) {
        if(l == firstLayer){
            go("copyWords", (unsigned int)((n * hidden + 255) / 256), 1, 256, 1, mappedEmbed, embedRows, n * hidden);
            go("copyWords", (unsigned int)((n + 255) / 256), 1, 256, 1, mappedPositions, positionsOnGpu, n);
            if(l == 0) go("spreadRows", (unsigned int)((n * hcWidth + 255) / 256), 1, 256, 1, embedRows, stack, n * hcWidth);
        } else if(!dense) finish(n, l - 1);
        if(l == 1 && usePle) perLayer(n);

        mix(L.hcNorm[0], L.hcDown[0], L.hcUp[0], &L.hcInject[0], stack, n, x, inj, dense && l != firstLayer ? shared : 0);
        if(L.attention) attend(L, n, top);
        else deltaNet(L, n);
        mix(L.hcNorm[1], L.hcDown[1], L.hcUp[1], &L.hcInject[1], stack, n, x, inj, y);
        if(dense) feedForward(L, l, n);
        else route(L, l, n);
    }

    struct Batch {
        int first = 0;
        int groups = 0;
        int rowStart = 0;
        int rowEnd = 0;
        int widest = 1;
    };

    void crest(GpuLayer& L, int l, int n, int top) {
        phase = "ple";
        if(l == 1 && usePle) perLayer(n);
        phase = "hc";
        mix(L.hcNorm[0], L.hcDown[0], L.hcUp[0], &L.hcInject[0], stack, n, x, inj);
        phase = L.attention ? "attention" : "gdn";
        if(L.attention) attend(L, n, top);
        else deltaNet(L, n);
        phase = "hc";
        mix(L.hcNorm[1], L.hcDown[1], L.hcUp[1], &L.hcInject[1], stack, n, x, inj, y);
        if(wideHead) return;
        phase = "router";
        if(!halvedAll({&L.sharedGate, &L.sharedUp})) pack(x, n * hidden, 0);
        plain(L.router, x, hidden, routerOut, expertCount, n, 0);
        plain(L.sharedScalar, x, hidden, scalar, 1, n, 0);
        go("routeTop", (unsigned int)n, 1, expertCount, 1, routerOut, routeIds, routeWeights);
        phase = "shared";
        project(L.sharedGate, x, 0, sharedGateOut, n);
        project(L.sharedUp, x, 0, sharedUpOut, n);
        go("swiglu", (unsigned int)((n * expertWidth + 255) / 256), 1, 256, 1, sharedGateOut, sharedUpOut, sharedHidden, n * expertWidth);
        if(!halvedAll({&L.sharedDown})) pack(sharedHidden, n * expertWidth, 1);
        project(L.sharedDown, sharedHidden, 1, shared, n);
        phase = "experts";
    }

    void routeAll(GpuLayer& L, int m) {
        uint64_t room = ((uint64_t)m * hidden * 2 + 255) / 256 * 256;
        uint64_t keptRows = wideRows, keptRoom = wideRoom;
        wideRows = moeOut;
        wideRoom = room;
        uint64_t picks = moeOut + 2 * room;
        phase = "router";
        if(!halvedAll({&L.sharedGate, &L.sharedUp})) pack(x, m * hidden, 0);
        plain(L.router, x, hidden, picks, expertCount, m, 0);
        plain(L.sharedScalar, x, hidden, scalar, 1, m, 0, L.router.cols != L.sharedScalar.cols);
        if(expertCount % 32 == 0 && expertCount <= 1024) go("routeWarp", (unsigned int)((m + 7) / 8), 1, 256, 1, picks, routeIds, routeWeights, m);
        else go("routeTop", (unsigned int)m, 1, expertCount, 1, picks, routeIds, routeWeights);
        wideRows = keptRows;
        wideRoom = keptRoom;
        phase = "experts";
    }

    void sharedAll(GpuLayer& L, int m) {
        uint64_t keptRows = wideRows, keptRoom = wideRoom;
        wideRows = moeOut;
        wideRoom = ((uint64_t)m * hidden * 2 + 255) / 256 * 256;
        phase = "shared";
        projectMany({{&L.sharedGate, expertGateOut}, {&L.sharedUp, expertUpOut}}, x, 0, m);
        go("swiglu", (unsigned int)(((uint64_t)m * expertWidth + 255) / 256), 1, 256, 1, expertGateOut, expertUpOut, expertHidden, m * expertWidth);
        pack(x, m * hidden, 0);
        if(!halvedAll({&L.sharedDown})) pack(expertHidden, m * expertWidth, 1);
        project(L.sharedDown, expertHidden, 1, x, m);
        wideRows = keptRows;
        wideRoom = keptRoom;
        phase = "experts";
    }

    void runBatch(GpuLayer& L, uint64_t gate, uint64_t up, uint64_t down, uint64_t gateBytes, uint64_t upBytes, uint64_t downBytes, uint64_t slotOnGpu, uint64_t startOnGpu, uint64_t countOnGpu, uint64_t inputOnGpu, uint64_t weightOnGpu, const Batch& b) {
        int rows = b.rowEnd - b.rowStart;
        bool wide = rows >= wideLeast * b.groups;
        int width = tileWidth(wide);
        unsigned int threads = width == 128 ? 256 : 128;
        int tiles = (b.widest + width - 1) / width;
        uint64_t shift = 4ULL * b.first;
        uint64_t rowShift = 4ULL * b.rowStart * expertWidth;
        uint64_t none = 0;
        PackedInput& a = packs[0];
        PackedInput f = fine && residual && batchFine ? fines[0] : PackedInput();
        go(tiled(L.expertGate.type, wide, f.values).c_str(), (expertWidth + rowTile(L.expertGate.type, wide, f.values) - 1) / rowTile(L.expertGate.type, wide, f.values), (unsigned int)(b.groups * tiles), threads, 1, (int)L.expertGate.type, gate, L.expertGate.stride, gateBytes, slotOnGpu + shift, startOnGpu + shift, countOnGpu + shift, inputOnGpu, a.values, a.scales, a.sums, f.values, f.scales, f.sums, expertGateOut - rowShift, expertWidth, hidden, 0, tiles, none, none, none, tables);
        go(tiled(L.expertUp.type, wide, f.values).c_str(), (expertWidth + rowTile(L.expertUp.type, wide, f.values) - 1) / rowTile(L.expertUp.type, wide, f.values), (unsigned int)(b.groups * tiles), threads, 1, (int)L.expertUp.type, up, L.expertUp.stride, upBytes, slotOnGpu + shift, startOnGpu + shift, countOnGpu + shift, inputOnGpu, a.values, a.scales, a.sums, f.values, f.scales, f.sums, expertUpOut - rowShift, expertWidth, hidden, 0, tiles, none, none, none, tables);
        swigluPack(expertGateOut, expertUpOut, expertHidden, rows * expertWidth, 1);

        PackedInput& c = packs[1];
        PackedInput d = fine && residual && batchFine ? fines[1] : PackedInput();
        uint64_t values = (uint64_t)b.rowStart * expertWidth;
        uint64_t scales = 4ULL * b.rowStart * (expertWidth / 32);
        uint64_t fineValues = d.values ? d.values - values : 0;
        uint64_t fineScales = d.scales ? d.scales - scales : 0;
        uint64_t fineSums = d.sums ? d.sums - scales : 0;
        go(tiled(L.expertDown.type, wide, fineValues).c_str(), (hidden + rowTile(L.expertDown.type, wide, fineValues) - 1) / rowTile(L.expertDown.type, wide, fineValues), (unsigned int)(b.groups * tiles), threads, 1, (int)L.expertDown.type, down, L.expertDown.stride, downBytes, slotOnGpu + shift, startOnGpu + shift, countOnGpu + shift, none, c.values - values, c.scales - scales, c.sums - scales, fineValues, fineScales, fineSums, moeOut, hidden, expertWidth, 0, tiles, inputOnGpu, weightOnGpu, moeOut, tables);
    }

    void issuePiece() {
        CudaQueue& queue = cudaQueue();
        const Piece& piece = streamPieces[streamIssued];
        int side = (int)(streamIssued % stageSides);
        int layer = piece.layer;
        GpuLayer& L = layers[layer];
        const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
        auto clock = std::chrono::steady_clock::now();
        std::vector<std::pair<int, int>> wanted;
        std::vector<char*> targets;
        std::vector<int> cached;
        std::vector<char> direct(piece.count, 0);
        for(int j = 0; j < piece.count; j++){
            int e = streamPlan[layer][piece.first + j];
            if(directStream && L.warmHost[0] && L.warm[e] >= 0){
                direct[j] = 1;
                continue;
            }
            bool warm = !packed || warmClock[(size_t)layer * expertCount + e] != 0 || (L.warmHost[0] && L.warm[e] >= 0);
            if(warm){
                cached.push_back(j);
                continue;
            }
            wanted.push_back({layer, e});
            targets.push_back((char*)ring[side] + (uint64_t)j * L.slice);
        }
        if(!cached.empty() || !wanted.empty()){
            if(queue.eventSync(moved[side]) != 0) failed = true;
            auto copied = std::chrono::steady_clock::now();
            pool.run((int)cached.size() * 3, [&](int job){
                int j = cached[job / 3], i = job % 3;
                int e = streamPlan[layer][piece.first + j];
                uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride;
                const uint8_t* source = L.warmHost[0] && L.warm[e] >= 0
                    ? L.warmHost[i] + (uint64_t)L.warm[e] * span
                    : parts[i]->data + (uint64_t)e * span;
                memcpy(ring[side] + (uint64_t)j * L.slice + L.piece[i], source, (size_t)span);
            });
            copyMs += since(copied);
            auto read = std::chrono::steady_clock::now();
            if(!wanted.empty() && !expertPack.readMany(wanted, targets)){
                problem = "reading the expert pack failed";
                failed = true;
            }
            readMs += since(read);
        }
        queue.streamWait(copyStream, stageFree[side], 0);
        for(int j = 0; j < piece.count; j++){
            if(direct[j]){
                int e = streamPlan[layer][piece.first + j];
                int run = 1;
                while(queue.copyPlane && j + run < piece.count && direct[j + run] && L.warm[streamPlan[layer][piece.first + j + run]] == L.warm[e] + run) run++;
                for(int i = 0; i < 3; i++){
                    uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride;
                    uint64_t to = stage[side] + (uint64_t)j * L.slice + L.piece[i];
                    const uint8_t* from = L.warmHost[i] + (uint64_t)L.warm[e] * span;
                    int code = run > 1
                        ? queue.planeUp(to, L.slice, from, span, span, run, copyStream)
                        : queue.copyUp(to, from, (size_t)span, copyStream);
                    if(code != 0) failed = true;
                }
                j += run - 1;
                continue;
            }
            int end = j + 1;
            while(end < piece.count && !direct[end]) end++;
            if(queue.copyUp(stage[side] + (uint64_t)j * L.slice, ring[side] + (uint64_t)j * L.slice, (size_t)(end - j) * L.slice, copyStream) != 0) failed = true;
            j = end - 1;
        }
        queue.eventRecord(moved[side], copyStream);
        if(flushQueue) queue.query(copyStream);
        streamIssued++;
        streamMs += since(clock);
        if(!readAheadCold || streamIssued >= streamPieces.size()) return;
        const Piece& next = streamPieces[streamIssued];
        GpuLayer& N = layers[next.layer];
        const Matrix* nextParts[3] = {&N.expertGate, &N.expertUp, &N.expertDown};
        std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
        for(int j = 0; j < next.count; j++){
            int e = streamPlan[next.layer][next.first + j];
            if(warmClock[(size_t)next.layer * expertCount + e] != 0 || (N.warmHost[0] && N.warm[e] >= 0)) continue;
            for(int i = 0; i < 3; i++){
                uint64_t span = (uint64_t)nextParts[i]->rows * nextParts[i]->stride;
                ranges.push_back({(PVOID)(nextParts[i]->data + (uint64_t)e * span), (SIZE_T)span});
            }
        }
        if(!ranges.empty()) PrefetchVirtualMemory(GetCurrentProcess(), ranges.size(), ranges.data(), 0);
    }

    std::vector<int> coldPick(int l, int m) {
        GpuLayer& L = layers[l];
        std::vector<std::pair<double, int>> order;
        double total = 0.0;
        for(int e = 0; e < expertCount; e++){
            size_t id = (size_t)l * expertCount + e;
            double h = 1.0 + (id < heat.size() ? (double)heat[id] : 0.0) + (id < learned.size() ? (double)learned[id] : 0.0);
            total += h;
            if(L.slot[e] < 0 && L.warm[e] >= 0) order.push_back({h, e});
        }
        std::sort(order.begin(), order.end());
        std::vector<int> picked;
        double expected = 0.0, scale = (double)m * expertsUsed / total;
        for(const auto& o : order){
            double routes = o.first * scale;
            if(expected + routes > (double)coldRoutes) break;
            expected += routes;
            picked.push_back(o.second);
        }
        std::sort(picked.begin(), picked.end());
        return picked;
    }

    void coldCompute(GpuLayer& L, int layer, const std::vector<int>& expertOf, const std::vector<int>& tokenOf, const float* rowsIn, float* rowsOut) {
        int rows = (int)expertOf.size();
        if(rows == 0) return;
        uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride;
        uint64_t upSpan = (uint64_t)L.expertUp.rows * L.expertUp.stride;
        uint64_t downSpan = (uint64_t)L.expertDown.rows * L.expertDown.stride;
        cpuInput.resize((size_t)rows * hidden);
        for(int i = 0; i < rows; i++) memcpy(cpuInput.data() + (size_t)i * hidden, rowsIn + (size_t)tokenOf[i] * hidden, sizeof(float) * hidden);
        cpuGate.resize((size_t)rows * expertWidth);
        cpuUp.resize((size_t)rows * expertWidth);
        hostInput.fill(cpuInput.data(), rows * hidden);
        std::vector<HostProduct> work;
        for(int i = 0; i < rows;){
            int e = expertOf[i];
            int k = 1;
            while(i + k < rows && expertOf[i + k] == e) k++;
            Matrix g = L.expertGate, up = L.expertUp;
            g.data = L.warmHost[0] + (uint64_t)L.warm[e] * gateSpan;
            up.data = L.warmHost[1] + (uint64_t)L.warm[e] * upSpan;
            work.push_back({g, &hostInput, i, cpuGate.data() + (size_t)i * expertWidth, k});
            work.push_back({up, &hostInput, i, cpuUp.data() + (size_t)i * expertWidth, k});
            i += k;
        }
        hostMultiply(pool, work, residual);
        hostSwiglu(cpuGate.data(), cpuUp.data(), cpuGate.size());
        hostHidden.fill(cpuGate.data(), rows * expertWidth);
        work.clear();
        for(int i = 0; i < rows;){
            int e = expertOf[i];
            int k = 1;
            while(i + k < rows && expertOf[i + k] == e) k++;
            Matrix d = L.expertDown;
            d.data = L.warmHost[2] + (uint64_t)L.warm[e] * downSpan;
            work.push_back({d, &hostHidden, i, rowsOut + (size_t)i * hidden, k});
            i += k;
        }
        hostMultiply(pool, work, residual);
        coldRouted += rows;
    }

    void moePass(GpuLayer& L, int layer, int m) {
        CudaQueue& queue = cudaQueue();
        int routes = m * expertsUsed;
        bool coldOn = layer < (int)coldPlan.size() && !coldPlan[layer].empty();
        if(coldOn && queue.copyDown(hostExpertOut, x, (size_t)m * hidden * 4, stream) != 0) failed = true;
        go("copyWords", (unsigned int)((routes + 255) / 256), 1, 256, 1, routeIds, mappedIds, routes);
        go("copyWords", (unsigned int)((routes + 255) / 256), 1, 256, 1, routeWeights, mappedWeights, routes);
        if(queue.eventRecord(routed, stream) != 0) failed = true;
        if(wideHead){
            sharedAll(L, m);
            go("clearWords", (unsigned int)(((exactMix ? 2 : 1) * m * hidden + 255) / 256), 1, 256, 1, moeOut, (exactMix ? 2 : 1) * m * hidden);
        }
        auto waited = std::chrono::steady_clock::now();
        if(queue.eventSync(routed) != 0) failed = true;
        waitMs += since(waited);

        int* input = hostLists;
        float* weight = (float*)(hostLists + listSpan);
        int* groupSlot = hostLists + 2 * listSpan;
        int* groupStart = groupSlot + expertCount;
        int* groupCount = groupStart + expertCount;
        int* streamSlot = groupCount + expertCount;
        int* streamStart = streamSlot + expertCount;
        int* streamCount = streamStart + expertCount;

        std::fill(expertTally.begin(), expertTally.end(), 0);
        for(int i = 0; i < routes; i++){
            int e = hostIds[i];
            if(e < 0 || e >= expertCount){
                failed = true;
                e = 0;
                hostIds[i] = 0;
            }
            expertTally[e]++;
        }

        std::vector<char> inPlan(expertCount, 0);
        for(int e : streamPlan[layer]) inPlan[e] = 1;
        std::vector<char> inCold(expertCount, 0);
        if(coldOn) for(int e : coldPlan[layer]) inCold[e] = 1;
        std::vector<int> kept, fetched;
        int cursor = 0, streamed = 0;
        for(int e = 0; e < expertCount; e++){
            if(!expertTally[e] || L.slot[e] < 0 || inPlan[e] || inCold[e]) continue;
            expertBegin[e] = cursor;
            kept.push_back(e);
            cursor += expertTally[e];
        }
        int keptRows = cursor;
        for(int e = 0; e < expertCount; e++){
            if(expertTally[e] && L.slot[e] < 0 && !inPlan[e] && !inCold[e]){
                problem = "an expert outside the stream plan was routed";
                failed = true;
            }
            if(!inPlan[e]) continue;
            expertBegin[e] = cursor;
            fetched.push_back(e);
            cursor += expertTally[e];
            streamed += expertTally[e] > 0;
        }
        int coldStart = cursor;
        std::vector<int> coldExpert;
        if(coldOn){
            for(int e : coldPlan[layer]){
                expertBegin[e] = cursor;
                cursor += expertTally[e];
                coldExpert.insert(coldExpert.end(), expertTally[e], e);
            }
        }
        for(int i = 0; i < routes; i++){
            int a = expertBegin[hostIds[i]]++;
            input[a] = i / expertsUsed;
            weight[a] = hostWeights[i];
        }
        for(int e = 0; e < expertCount; e++) expertHits[(size_t)layer * expertCount + e] += expertTally[e];
        L.lastRouting.assign(hostIds + (size_t)(m - 1) * expertsUsed, hostIds + (size_t)m * expertsUsed);
        routedCount += routes;
        residentCount += keptRows;
        streamedCount += streamed;

        std::vector<Batch> keptBatches;
        for(size_t g = 0; g < kept.size(); g++){
            int e = kept[g];
            int start = expertBegin[e] - expertTally[e];
            groupSlot[g] = L.slot[e];
            groupStart[g] = start;
            groupCount[g] = expertTally[e];
            if(keptBatches.empty() || keptBatches.back().rowEnd - keptBatches.back().rowStart + expertTally[e] > batchRows){
                Batch b;
                b.first = (int)g;
                b.rowStart = start;
                b.rowEnd = start;
                keptBatches.push_back(b);
            }
            Batch& b = keptBatches.back();
            b.groups++;
            b.rowEnd += expertTally[e];
            b.widest = std::max(b.widest, expertTally[e]);
        }
        for(size_t k = streamDone; k < streamPieces.size() && streamPieces[k].layer == layer; k++){
            const Piece& piece = streamPieces[k];
            for(int j = 0; j < piece.count; j++){
                int g = piece.first + j, e = fetched[g];
                streamSlot[g] = j;
                streamStart[g] = expertBegin[e] - expertTally[e];
                streamCount[g] = expertTally[e];
            }
        }

        int words = 2 * listSpan + 6 * expertCount;
        go("copyWords", (unsigned int)((words + 255) / 256), 1, 256, 1, mappedLists, lists, words);
        if(!wideHead){
            go("clearWords", (unsigned int)(((exactMix ? 2 : 1) * m * hidden + 255) / 256), 1, 256, 1, moeOut, (exactMix ? 2 : 1) * m * hidden);
            pack(x, m * hidden, 0);
        }

        uint64_t inputOnGpu = lists;
        uint64_t weightOnGpu = lists + 4ULL * listSpan;
        uint64_t slotOnGpu = lists + 8ULL * listSpan;
        uint64_t startOnGpu = slotOnGpu + 4ULL * expertCount;
        uint64_t countOnGpu = startOnGpu + 4ULL * expertCount;
        uint64_t streamSlotOnGpu = countOnGpu + 4ULL * expertCount;
        uint64_t streamStartOnGpu = streamSlotOnGpu + 4ULL * expertCount;
        uint64_t streamCountOnGpu = streamStartOnGpu + 4ULL * expertCount;
        uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride;
        uint64_t upSpan = (uint64_t)L.expertUp.rows * L.expertUp.stride;
        uint64_t downSpan = (uint64_t)L.expertDown.rows * L.expertDown.stride;

        bool keptDone = false;
        auto runKept = [&]{
            for(const Batch& b : keptBatches) runBatch(L, L.resident[0], L.resident[1], L.resident[2], gateSpan, upSpan, downSpan, slotOnGpu, startOnGpu, countOnGpu, inputOnGpu, weightOnGpu, b);
            if(flushQueue) queue.query(stream);
            keptDone = true;
        };
        size_t layerStart = streamDone;
        while(streamDone < streamPieces.size() && streamPieces[streamDone].layer == layer){
            if(!keptDone && streamDone - layerStart == (size_t)stageSides) runKept();
            const Piece& piece = streamPieces[streamDone];
            int side = (int)(streamDone % stageSides);
            queue.streamWait(stream, moved[side], 0);
            Batch b;
            bool open = false;
            for(int j = 0; j <= piece.count; j++){
                int g = piece.first + j;
                int tally = j < piece.count ? expertTally[fetched[g]] : 0;
                if(open && (j == piece.count || b.rowEnd - b.rowStart + tally > batchRows)){
                    if(b.rowEnd > b.rowStart) runBatch(L, stage[side] + L.piece[0], stage[side] + L.piece[1], stage[side] + L.piece[2], L.slice, L.slice, L.slice, streamSlotOnGpu, streamStartOnGpu, streamCountOnGpu, inputOnGpu, weightOnGpu, b);
                    open = false;
                }
                if(j == piece.count) break;
                if(!open){
                    b = Batch();
                    b.first = g;
                    b.rowStart = streamStart[g];
                    b.rowEnd = streamStart[g];
                    open = true;
                }
                b.groups++;
                b.rowEnd += tally;
                b.widest = std::max(b.widest, tally);
            }
            queue.eventRecord(stageFree[side], stream);
            if(flushQueue) queue.query(stream);
            streamDone++;
            if(streamIssued < streamPieces.size()) issuePiece();
        }
        if(!keptDone) runKept();
        if(coldOn && !coldExpert.empty()){
            if(flushQueue) queue.query(stream);
            auto coldClock = std::chrono::steady_clock::now();
            int total = (int)coldExpert.size();
            int room = (int)std::min<uint64_t>(hostExpertRows - (uint64_t)m, hostXRows * hidden / 2);
            float* rowsOut = hostExpertOut + (size_t)m * hidden;
            int* tokenOut = (int*)hostX;
            float* weightOut = hostX + room;
            for(int done = 0; done < total && room > 0 && !failed; done += room){
                int part = std::min(room, total - done);
                if(done > 0 && (queue.eventRecord(routed, stream) != 0 || queue.eventSync(routed) != 0)) failed = true;
                std::vector<int> expertOf(coldExpert.begin() + done, coldExpert.begin() + done + part), tokenOf(part);
                for(int i = 0; i < part; i++){
                    int a = coldStart + done + i;
                    tokenOf[i] = input[a];
                    tokenOut[i] = input[a];
                    weightOut[i] = weight[a];
                }
                coldCompute(L, layer, expertOf, tokenOf, hostExpertOut, rowsOut);
                go("addRows", (unsigned int)((hidden + 255) / 256), (unsigned int)part, 256, 1, moeOut, mappedExpertOut + 4ULL * m * hidden, mappedX, mappedX + 4ULL * room, part);
            }
            coldMs += since(coldClock);
        }

        go("writeMixed", (unsigned int)((m * hidden + 255) / 256), 1, 256, 1, stack, wideHead ? x : shared, scalar, moeOut, inj, m);
    }

    void superChunk(const std::vector<int>& ids, const std::vector<int>& positions, size_t at, int m) {
        size_t first = tokens.size();
        tokens.insert(tokens.end(), ids.begin() + at, ids.begin() + at + m);
        auto clock = std::chrono::steady_clock::now();
        pool.run(m, [&](int t){
            embedCell(ids[at + t], hostEmbed + (size_t)t * hidden);
        });
        for(int t = 0; t < m; t++) hostPositions[t] = positions[at + t];
        embedMs += since(clock);
        if(usePle) pleRows(m, first);
        go("copyWords", (unsigned int)((m * hidden + 255) / 256), 1, 256, 1, mappedEmbed, embedRows, m * hidden);
        go("copyWords", (unsigned int)((m + 255) / 256), 1, 256, 1, mappedPositions, positionsOnGpu, m);
        if(firstLayer == 0) go("spreadRows", (unsigned int)((m * hcWidth + 255) / 256), 1, 256, 1, embedRows, stack, m * hcWidth);
        else if(frontAhead ? !takeAhead(m) : !pullFront(m, false, ids.data() + at, false)) return;
        if(pipeFront && at + m < ids.size() && !sendAhead(ids, positions, at + m, (int)std::min(ids.size() - at - m, (size_t)superLimit))) return;
        for(int k = 0; k < 2; k++){
            int row = cutRows[k] - (int)at;
            crestCuts[k] = row > 0 && row < m && cutSlots[k] >= 0 && snapshotRoom(cutSlots[k]) ? row : -1;
        }
        superLayers(m);
        for(int k = 0; k < 2; k++){
            if(failed) break;
            if(crestCuts[k] > 0){
                savedTokens[cutSlots[k]] = first + crestCuts[k];
                cutKept[k] = true;
            } else if(cutSlots[k] >= 0 && cutRows[k] == (int)(at + m) && moveState(cutSlots[k], true, false)){
                cutKept[k] = true;
            }
            crestCuts[k] = -1;
        }
    }

    bool superRows(int n, const int* positions, const float* embeds, const float* ples, float* out) {
        for(int at = 0; at < n && !failed; at += superLimit){
            int m = std::min(superLimit, n - at);
            memcpy(hostEmbed, embeds + (size_t)at * hidden, (size_t)m * hidden * 4);
            memcpy(hostPle, ples + (size_t)at * hidden, (size_t)m * hidden * 4);
            for(int t = 0; t < m; t++) hostPositions[t] = positions[at + t];
            go("copyWords", (unsigned int)((m * hidden + 255) / 256), 1, 256, 1, mappedEmbed, embedRows, m * hidden);
            go("copyWords", (unsigned int)((m + 255) / 256), 1, 256, 1, mappedPositions, positionsOnGpu, m);
            go("spreadRows", (unsigned int)((m * hcWidth + 255) / 256), 1, 256, 1, embedRows, stack, m * hcWidth);
            superLayers(m);
            if(cudaLink().sync() != 0) failed = true;
            if(!failed && cudaLink().copyDown(out + (size_t)at * hcWidth, stack, (size_t)m * hcWidth * 4) != 0) failed = true;
            if(!failed && rowsReady && !rowsReady(at, m)) failed = true;
        }
        return !failed;
    }

    void superLayers(int m) {
        bool lifts = liftHold, swaps = exchangeHold;
        liftHold = true;
        exchangeHold = true;
        streamPlan.assign(layers.size(), {});
        streamPieces.clear();
        streamIssued = 0;
        streamDone = 0;
        coldPlan.assign(layers.size(), {});
        bool cold = coldRoutes > 0 && m > chunk && m <= coldRows && !remote && hostX && hostExpertOut && (uint64_t)m < hostExpertRows;
        for(int l = firstLayer; l < endLayer && ring[0]; l++){
            GpuLayer& L = layers[l];
            if(cold && L.warmHost[0] && L.warmHost[1] && L.warmHost[2]) coldPlan[l] = coldPick(l, m);
            std::vector<char> skip(expertCount, 0);
            for(int e : coldPlan[l]) skip[e] = 1;
            for(int e = 0; e < expertCount; e++){
                if(L.slot[e] < 0 && !skip[e]) streamPlan[l].push_back(e);
            }
            for(int at = 0; at < (int)streamPlan[l].size(); at += stageSlots) streamPieces.push_back({l, at, std::min(stageSlots, (int)streamPlan[l].size() - at)});
        }
        while(streamIssued < streamPieces.size() && streamIssued < (size_t)stageSides) issuePiece();
        wideHead = blas != nullptr && (uint64_t)m * (2 * hidden * 2 + expertCount * 4) <= (uint64_t)superLimit * hidden * 8;
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            auto clock = std::chrono::steady_clock::now();
            phase = "half";
            halve(L);
            for(int c0 = 0; c0 < m;){
                int cn = std::min(crestRows, m - c0);
                for(int k = 0; k < 2; k++){
                    if(crestCuts[k] > c0 && crestCuts[k] < c0 + cn) cn = crestCuts[k] - c0;
                }
                int top = 0;
                for(int t = c0; t < c0 + cn; t++) top = std::max(top, hostPositions[t]);
                uint64_t keep[9] = {stack, x, inj, shared, scalar, routeIds, routeWeights, positionsOnGpu, mappedPle};
                stack += 4ULL * c0 * hcWidth;
                x += 4ULL * c0 * hidden;
                inj += 4ULL * c0 * hcCount;
                shared += 4ULL * c0 * hidden;
                scalar += 4ULL * c0;
                routeIds += 4ULL * c0 * expertsUsed;
                routeWeights += 4ULL * c0 * expertsUsed;
                positionsOnGpu += 4ULL * c0;
                mappedPle += 4ULL * c0 * hidden;
                crest(L, l, cn, top);
                stack = keep[0];
                x = keep[1];
                inj = keep[2];
                shared = keep[3];
                scalar = keep[4];
                routeIds = keep[5];
                routeWeights = keep[6];
                positionsOnGpu = keep[7];
                mappedPle = keep[8];
                if(flushQueue) cudaQueue().query(stream);
                c0 += cn;
                for(int k = 0; k < 2; k++){
                    if(crestCuts[k] == c0) keepLayer(cutSlots[k], l);
                }
            }
            if(wideHead) routeAll(L, m);
            moePass(L, l, m);
            layerMs[l] += since(clock);
        }
        halved.clear();
        streamPieces.clear();
        coldPlan.clear();
        wideHead = false;
        liftHold = lifts;
        exchangeHold = swaps;
    }

    bool pullFront(int n, bool record, const int* ids, bool leave) {
        if(!remote) return true;
        if(frontStack.size() < (size_t)n * hcWidth) frontStack.resize((size_t)n * hcWidth);
        auto clock = std::chrono::steady_clock::now();
        // las filas salen de tablas cuantizadas: van los bloques tal cual (4 veces menos bytes) y el nodo los abre igual
        // una fila de imagen no esta en la tabla y manda todo en floats
        bool packed = canPack(ids, n) && (!usePle || pleRowIds.size() >= (size_t)n * pleHeads);
        uint32_t flags = askFlags(n) | (leave && (remote->features & ROWS_LEAVE) ? ROWS_LEAVE : 0);
        bool ok = false;
        if(packed){
            packRows(ids, hostPositions, n, record, flags, pleRowIds.data());
            ok = remote->exchange(n, frontStack.data(), hcWidth);
        } else ok = remote->rows(n, record, hostPositions, hostEmbed, hostPle, hidden, frontStack.data(), hcWidth, flags);
        frontMs += since(clock);
        if(!ok){
            problem = remote->problem;
            failed = true;
            return false;
        }
        if(cudaLink().copyUp(stack, frontStack.data(), (size_t)n * hcWidth * 4) != 0){
            failed = true;
            return false;
        }
        return true;
    }

    uint32_t askFlags(int n) {
        return ((remote->features & ROWS_HALF) && !exact ? ROWS_HALF : 0) | ((remote->features & ROWS_PIECES) && n >= piecesFrom ? ROWS_PIECES : 0);
    }

    bool canPack(const int* ids, int n) {
        bool packed = remote && (remote->features & ROWS_PACKED) && ids && embed.data && embed.cols == hidden && (!usePle || pleTable.data);
        for(int t = 0; t < n && packed; t++) packed = ids[t] >= 0;
        return packed;
    }

    void packRows(const int* ids, const int* positions, int n, bool record, uint32_t flags, const uint64_t* pleIds) {
        int embedBytes = (int)embed.stride, pleBytes = usePle ? (int)pleTable.stride : 0;
        char* at = remote->begin(PACKET_ROWS, flags | ROWS_PACKED, packedSize(n, embedBytes, pleBytes, pleHeads));
        PackedHead head = {n, record ? 1 : 0, (int32_t)embed.type, embedBytes, usePle ? (int32_t)pleTable.type : -1, pleBytes};
        memcpy(at, &head, sizeof(head));
        at += sizeof(head);
        memcpy(at, positions, (size_t)n * 4);
        at += (size_t)n * 4;
        for(int t = 0; t < n; t++, at += embedBytes) memcpy(at, embed.data + (uint64_t)(ids[t] < embed.rows ? ids[t] : 0) * embed.stride, (size_t)embedBytes);
        for(int t = 0; t < n && usePle; t++){
            for(int h = 0; h < pleHeads; h++, at += pleBytes) memcpy(at, pleTable.data + pleIds[(size_t)t * pleHeads + h] * pleTable.stride, (size_t)pleBytes);
        }
    }

    // en un prompt largo la parte que sigue sale hacia el nodo antes de que esta pc haga sus capas de la actual:
    // el nodo la calcula y la manda mientras tanto y un hilo la recibe, asi la red y las dos tarjetas trabajan a la vez
    bool sendAhead(const std::vector<int>& ids, const std::vector<int>& positions, size_t at, int n) {
        size_t first = tokens.size();
        if(usePle){
            if(aheadPle.size() < (size_t)n * pleHeads) aheadPle.resize((size_t)n * pleHeads);
            pool.run(n * pleHeads, [&](int job){
                int t = job / pleHeads, h = job % pleHeads;
                aheadPle[job] = pleRowOf(first + t, h, [&](size_t s){ return s < first ? tokens[s] : ids[at + (s - first)]; });
            });
        }
        packRows(ids.data() + at, positions.data() + at, n, false, askFlags(n), aheadPle.data());
        if(frontStack.size() < (size_t)n * hcWidth) frontStack.resize((size_t)n * hcWidth);
        if(!remote->post()){
            problem = remote->problem;
            failed = true;
            return false;
        }
        FrontLink* link = remote;
        float* out = frontStack.data();
        frontPending = std::async(std::launch::async, [link, n, out]{ return link->collect(n, out, hcWidth); });
        frontAhead = true;
        aheadRows = n;
        return true;
    }

    bool takeAhead(int n) {
        frontAhead = false;
        auto clock = std::chrono::steady_clock::now();
        bool ok = frontPending.get();
        frontMs += since(clock);
        if(!ok || n != aheadRows){
            problem = !remote->problem.empty() ? remote->problem : "the pool node answered another part of the prompt";
            failed = true;
            return false;
        }
        if(cudaLink().copyUp(stack, frontStack.data(), (size_t)n * hcWidth * 4) != 0){
            failed = true;
            return false;
        }
        return true;
    }

    bool forwardRows(int n, bool record, bool leave, const int* positions, const float* embeds, const float* ples, float* out) {
        if(!ready || n <= 0 || tail() || !useCuda()){
            problem = "this node does not hold the leading layers";
            return false;
        }
        // las posiciones llegan por la red desde el master, ninguna puede salirse del contexto
        int highest = 0;
        for(int t = 0; t < n; t++){
            if(positions[t] < 0 || positions[t] >= contextLimit){
                problem = "position " + std::to_string(positions[t]) + " is past the context limit of " + std::to_string(contextLimit);
                return false;
            }
            highest = std::max(highest, positions[t]);
        }
        // la memoria del contexto crece igual que en el master: sin esto una conversacion mas larga que el espacio
        // del principio (16384 posiciones) escribia fuera de la memoria y el nodo quedaba muerto con cuda error 700
        if(!reachContext(highest + 1)){
            if(problem.empty()) problem = "no room in video memory for " + std::to_string(highest + 1) + " tokens of context";
            return false;
        }
        CudaQueue& queue = cudaQueue();
        failed = false;
        replayRows = 0;
        settle(n > relayLimit);
        // el master pide dejar el estado como esta en la pasada que verifica borradores: se graba cada fila y se avanza cuando confirma
        // si esta pasada no puede grabar se guarda una copia del estado y confirmar la restaura como antes
        settling = leave && record && n > 1 && n <= relayLimit && n <= chunk && fine && !exact && !profile && useGraphs && replayStore;
        if(settling) savedTokens[1] = tokens.size();
        else if(leave && !saveState(1)){
            problem = "could not keep the state before checking the drafts";
            return false;
        }
        layerMs.assign(layers.size(), 0.0);
        waitMs = 0;
        spilledRows = 0;
        missedRoutes = 0;
        missedUnique = 0;
        bool streamed = streamPrefill && fine && !exact && n > chunk;
        if(streamed) superRows(n, positions, embeds, ples, out);
        for(int at = 0; at < n && !failed && !streamed; at += chunk){
            int cn = std::min(chunk, n - at);
            memcpy(hostEmbed, embeds + (size_t)at * hidden, (size_t)cn * hidden * 4);
            memcpy(hostPle, ples + (size_t)at * hidden, (size_t)cn * hidden * 4);
            int top = 0;
            for(int t = 0; t < cn; t++){
                hostPositions[t] = positions[at + t];
                top = std::max(top, hostPositions[t]);
            }
            if(cn <= relayLimit && fine && !exact && !profile && useGraphs){
                // la pasada que graba para confirmar tiene su propio grafo: uno capturado sin grabar no copia lo que confirmar repite
                recording = record && cn > 1;
                leaveState = recording && settling;
                everyRow = recording;
                relayToken(top, cn);
                replayRows = recording ? cn : 0;
                recording = false;
                leaveState = false;
                everyRow = false;
            } else {
                fixed = false;
                for(int l = firstLayer; l < endLayer && !failed; l++){
                    GpuLayer& L = layers[l];
                    auto clock = std::chrono::steady_clock::now();
                    front(L, l, cn, top);
                    if(!dense){
                        if(queue.eventRecord(routed, stream) != 0) failed = true;
                        if(queue.eventSync(routed) != 0) failed = true;
                        gather(L, l, cn);
                        resident(L, cn);
                        spill(L, cn);
                    }
                    layerMs[l] += since(clock);
                }
                finish(cn, endLayer - 1);
            }
            fixed = false;
            if(cudaLink().sync() != 0) failed = true;
            if(!failed && cudaLink().copyDown(out + (size_t)at * hcWidth, stack, (size_t)cn * hcWidth * 4) != 0) failed = true;
            if(!failed && rowsReady && !rowsReady(at, cn)) failed = true;
        }
        bool alarmed = hostFlags && hostFlags[48] != 0;
        if(alarmed) hostFlags[48] = 0;
        if(failed || alarmed){
            problem = alarmed
                ? "a gpu wait for the cpu experts timed out"
                : "gpu forward failed";
            failed = false;
            return false;
        }
        sinceAdapt += n;
        if(!exchangeReady() && adaptEvery > 0 && sinceAdapt >= adaptEvery){
            sinceAdapt = 0;
            adapt(std::max(adaptSwaps, std::min(1024, n / 2)));
        }
        heatClock += n;
        if(heatClock >= 2048){
            heatClock = 0;
            saveHeat();
        }
        return true;
    }

    void finale(int n, bool settled = false) {
        if(!settled) finish(n, (int)layers.size() - 1);
        int rows = everyRow ? n : 1;
        uint64_t from = everyRow
            ? stack
            : stack + 4ULL * (n - 1) * hcWidth;
        uint64_t into = rows == 1 ? fin : y;
        mix(outNorm, outDown, outUp, nullptr, from, rows, into, 0);
        pack(into, rows * hidden, 0);
        project(head, into, 0, vocabOut, rows);
        if(shortlist > 0) go("shortRows", (unsigned int)rows, 1, 512, 1, vocabOut, head.rows, shortlist, mappedLogits);
        else go("copyWords", (unsigned int)((rows * head.rows + 255) / 256), 1, 256, 1, vocabOut, mappedLogits, rows * head.rows);
    }

    bool replay(void*& exec, const std::function<void()>& work) {
        CudaQueue& queue = cudaQueue();
        if(!useGraphs){
            work();
            return true;
        }
        if(!exec){
            void* graph = nullptr;
            int begun = queue.beginCapture(stream, 2);
            int ended = -1, made = -1;
            bool broke = false;
            if(begun == 0){
                work();
                broke = failed;
                ended = queue.endCapture(stream, &graph);
                made = ended == 0 && graph ? queue.instantiate(&exec, graph, 0) : -1;
                if(graph) queue.graphDestroy(graph);
            }
            if(begun != 0 || ended != 0 || made != 0){
                graphProblem = "begin " + std::to_string(begun) + ", end " + std::to_string(ended) + ", instantiate " + std::to_string(made) + (broke ? ", a launch failed while capturing" : "");
                exec = nullptr;
                failed = false;
                return false;
            }
        }
        if(queue.graphLaunch(exec, stream) != 0) failed = true;
        return true;
    }

    void relayToken(int top, int n = 1) {
        CudaQueue& queue = cudaQueue();
        if(hostPlace.Mask && pinnedThread != GetCurrentThreadId()){
            SetThreadGroupAffinity(GetCurrentThread(), &hostPlace, nullptr);
            pinnedThread = GetCurrentThreadId();
        }
        span = top + 1 <= budget
            ? 0
            : top < 8192
                ? 8192
                : top < 32768
                    ? 32768
                    : top < 131072
                        ? 131072
                        : contextLimit;
        span = std::min(span, contextLimit);
        if(graphBudget != budget){
            for(auto& g : tokenGraphs){
                if(g.second) queue.execDestroy(g.second);
            }
            tokenGraphs.clear();
            graphless.clear();
            graphBudget = budget;
        }
        if(queue.eventRecord(routed, stream) != 0 || queue.eventSync(routed) != 0) failed = true;
        ticket++;
        hostFlags[0] = ticket;
        relayRows = n;
        relay = true;
        fixed = true;
        int key = (((span * 32 + n * 2 + (everyRow ? 1 : 0)) * 4 + reachStep(top)) * 2 + (shortlist > 0 ? 1 : 0)) * 2 + (leaveState ? 1 : 0);
        auto work = [&]{
            for(int l = firstLayer; l < endLayer; l++){
                front(layers[l], l, n, top);
                if(!dense) resident(layers[l], n);
            }
            if(tail()) finale(n);
            else finish(n, endLayer - 1);
            if(drafting && everyRow && n > 1) draftLayer(stack, embedRows + 4ULL * hidden, n - 1, top - 1, false);
        };
        bool recorded = n <= graphRows && stageRing == 0 && !graphless.count(key);
        if(recorded && !replay(tokenGraphs[key], work)){
            graphless.insert(key);
            recorded = false;
        }
        if(!recorded && dense) work();
        if(dense && hostDenseOut && n <= denseCpuRows){
            // windows junta el trabajo de la gpu en el driver: sin pedirlo el final del paso no sale y la cpu espera para siempre
            queue.query(stream);
            for(int l = firstLayer; l < endLayer; l++){
                if(layers[l].hostFfn && !serveDense(l, n)) break;
            }
        }
        if(!recorded && !dense){
            for(int l = firstLayer; l < endLayer; l++){
                front(layers[l], l, n, top);
                resident(layers[l], n);
                queue.query(stream);
                if(!serve(l, n)){
                    relay = false;
                    relayRows = 0;
                    return;
                }
            }
            if(tail()) finale(n);
            else finish(n, endLayer - 1);
            if(drafting && everyRow && n > 1) draftLayer(stack, embedRows + 4ULL * hidden, n - 1, top - 1, false);
        }
        relay = false;
        queue.query(stream);
        if(dense){
            if(queue.eventRecord(routed, stream) != 0 || queue.eventSync(routed) != 0) failed = true;
            relayRows = 0;
            return;
        }
        if(!recorded){
            relayRows = 0;
            return;
        }

        for(int l = firstLayer; l < endLayer; l++){
            if(!serve(l, n)) break;
        }
        relayRows = 0;
    }

    bool serveDense(int l, int n) {
        GpuLayer& L = layers[l];
        uint32_t want = ticket * 64u + (uint32_t)l + 1u;
        auto clock = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        while(hostFlags[16] < want){
            _mm_pause();
            if(++spins % 4096 == 0) cudaQueue().query(stream);
            if(since(clock) > 20000.0){
                if(!failed) launchProblem = "the gpu never reached layer " + std::to_string(l) + " (its last signal was " + std::to_string((long long)hostFlags[16] - (long long)ticket * 64 - 1) + ", ticket " + std::to_string(ticket) + ", rows " + std::to_string(n) + ")";
                failed = true;
                _mm_sfence();
                hostFlags[32] = ticket * 64u + 64u;
                return false;
            }
        }
        waitMs += since(clock);
        denseWait += since(clock);
        hostFlags[40] = want;
        _mm_sfence();
        auto started = std::chrono::steady_clock::now();
        hostInput.fill(hostX, n * hidden);
        cpuGate.resize((size_t)n * expertWidth);
        cpuUp.resize((size_t)n * expertWidth);
        std::vector<HostProduct> work = {{L.hostGate, &hostInput, 0, cpuGate.data(), n}, {L.hostUp, &hostInput, 0, cpuUp.data(), n}};
        hostMultiply(pool, work, residual);
        hostSwiglu(cpuGate.data(), cpuUp.data(), cpuGate.size());
        hostHidden.fill(cpuGate.data(), n * expertWidth);
        work = {{L.hostDown, &hostHidden, 0, hostDenseOut, n}};
        hostMultiply(pool, work, residual);
        expertMs += since(started);
        denseWork += since(started);
        _mm_sfence();
        hostFlags[32] = want;
        return true;
    }

    bool serve(int l, int n) {
        {
            GpuLayer& L = layers[l];
            uint32_t want = ticket * 64u + (uint32_t)l + 1u;
            auto clock = std::chrono::steady_clock::now();
            while(hostFlags[16] < want){
                _mm_pause();
                if(since(clock) > 20000.0){
                    if(!failed) launchProblem = "the gpu never reached layer " + std::to_string(l);
                    failed = true;
                    break;
                }
            }
            waitMs += since(clock);
            serveWait += since(clock);
            if(timing) cpuSaw[l] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
            if(failed){
                _mm_sfence();
                hostFlags[32] = ticket * 64u + 64u;
                return false;
            }
            hostFlags[40] = want;
            _mm_sfence();
            if(hostDelay > 0){
                Sleep(hostDelay);
                hostDelay = 0;
            }
            auto started = std::chrono::steady_clock::now();
            double launchBefore = helperLaunchMs, waitBefore = helperWaitMs;
            PROCESS_MEMORY_COUNTERS memoryBefore = {};
            GetProcessMemoryInfo(GetCurrentProcess(), &memoryBefore, sizeof(memoryBefore));
            gather(L, l, n);
            double gathered = since(started);
            if(foresee && l + 1 < endLayer) foreseeLayer(l + 1, hostNext + (size_t)l * expertsUsed * n, n);
            spill(L, n);
            double spent = since(started);
            serveWork += spent;
            _mm_sfence();
            hostFlags[32] = want;
            if(timing){
                cpuDone[l] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
                cpuGather[l] = gathered;
                cpuSeen[l] = cpuRows;
            }
            layerMs[l] += since(clock);
            if(spent > 1000.0 && launchProblem.empty()){
                bool fast = hostInt8 && hostDotType(L.expertGate.type) && hostDotType(L.expertUp.type) && hostDotType(L.expertDown.type);
                PROCESS_MEMORY_COUNTERS memoryAfter = {};
                GetProcessMemoryInfo(GetCurrentProcess(), &memoryAfter, sizeof(memoryAfter));
                double helperPart = (helperLaunchMs - launchBefore) + (helperWaitMs - waitBefore);
                launchProblem = "layer " + std::to_string(l) + " took " + std::to_string((int)spent) + " ms on the cpu (gather " + std::to_string((int)gathered) + " ms, cpu experts " + std::to_string((int)(spent - gathered - helperPart)) + " ms, second gpu " + std::to_string((int)helperPart) + " ms, " + std::to_string(memoryAfter.PageFaultCount - memoryBefore.PageFaultCount) + " page faults, " + std::to_string(cpuRows) + " rows, " + (fast ? "int8" : "float") + " path, types " + std::to_string(L.expertGate.type) + "/" + std::to_string(L.expertUp.type) + "/" + std::to_string(L.expertDown.type) + ")";
            }
        }
        return true;
    }

    bool snapshotRoom(int slot) {
        uint64_t& snapshot = snapshots[slot];
        void*& spare = hostSnapshots[slot];
        if(snapshot || spare) return true;
        uint64_t stateBytes = (uint64_t)valueHeads * ssmSize * ssmSize * 4;
        uint64_t convBytes = (uint64_t)3 * convWidth * 4;
        uint64_t total = (uint64_t)pleTaps * hcWidth * 4;
        for(int l = firstLayer; l < endLayer; l++){
            if(!layers[l].attention) total += stateBytes + convBytes;
        }
        // si la vram ya esta llena windows no niega la memoria sino que saca otras nuestras a la ram compartida y todo queda lento hasta cerrar
        size_t freeNow = 0, totalNow = 0;
        bool roomy = cudaLink().info(&freeNow, &totalNow) == 0 && (uint64_t)freeNow > total + (512ULL << 20);
        if(roomy && cudaLink().alloc(&snapshot, (size_t)total) == 0) allocations.push_back(snapshot);
        else if(cudaLink().hostAlloc(&spare, (size_t)total, 0) == 0){
            snapshot = 0;
            pinned.push_back(spare);
        } else {
            snapshot = 0;
            spare = nullptr;
            problem = "no memory left for a state snapshot";
            return false;
        }
        return true;
    }

    void keepPart(int slot, uint64_t device, uint64_t at, uint64_t bytes) {
        if(snapshots[slot]){
            if(cudaQueue().copyAcross(snapshots[slot] + at, device, (size_t)bytes, stream) != 0) failed = true;
            return;
        }
        if(cudaQueue().copyDown((uint8_t*)hostSnapshots[slot] + at, device, (size_t)bytes, stream) != 0) failed = true;
    }

    void keepLayer(int slot, int layer) {
        uint64_t stateBytes = (uint64_t)valueHeads * ssmSize * ssmSize * 4;
        uint64_t convBytes = (uint64_t)3 * convWidth * 4;
        uint64_t at = 0;
        for(int l = firstLayer; l < endLayer; l++){
            if(layers[l].attention) continue;
            if(l == layer){
                keepPart(slot, layers[l].state, at, stateBytes);
                keepPart(slot, layers[l].convHistory, at + stateBytes, convBytes);
            }
            at += stateBytes + convBytes;
        }
        if(layer == 1 && usePle && pleTaps > 0) keepPart(slot, pleHistory, at, (uint64_t)pleTaps * hcWidth * 4);
    }

    bool canMark(size_t rows) const {
        return ready && !gemma && !hy4 && !dense && !remote && firstLayer == 0 && streamPrefill && fine && !exact && rows > (size_t)chunk;
    }

    bool moveState(int slot, bool keep, bool tell = true) {
        if(gemma) return moveGemma(slot, keep);
        if(hy4) return moveHy4(slot, keep);
        if(!ready || slot < 0 || slot >= 4 || !useCuda()) return false;
        settleTail();
        if(!keep && !snapshots[slot] && !hostSnapshots[slot]) return false;
        if(remote && tell && !remote->state(keep ? STATE_SAVE : STATE_RESTORE, slot)) return false;
        uint64_t& snapshot = snapshots[slot];
        void*& spare = hostSnapshots[slot];
        uint64_t stateBytes = (uint64_t)valueHeads * ssmSize * ssmSize * 4;
        uint64_t convBytes = (uint64_t)3 * convWidth * 4;
        uint64_t pleBytes = (uint64_t)pleTaps * hcWidth * 4;
        if(!snapshot && !spare){
            if(!keep) return false;
            if(!snapshotRoom(slot)) return false;
        }
        CudaQueue& queue = cudaQueue();
        auto move = [&](uint64_t device, uint64_t at, uint64_t bytes) {
            if(snapshot){
                return (keep
                    ? queue.copyAcross(snapshot + at, device, (size_t)bytes, stream)
                    : queue.copyAcross(device, snapshot + at, (size_t)bytes, stream)) == 0;
            }
            uint8_t* host = (uint8_t*)spare + at;
            return (keep
                ? queue.copyDown(host, device, (size_t)bytes, stream)
                : queue.copyUp(device, host, (size_t)bytes, stream)) == 0;
        };
        uint64_t at = 0;
        bool ok = true;
        for(int l = firstLayer; l < endLayer; l++){
            GpuLayer& L = layers[l];
            if(L.attention) continue;
            ok = ok && move(L.state, at, stateBytes);
            at += stateBytes;
            ok = ok && move(L.convHistory, at, convBytes);
            at += convBytes;
        }
        ok = ok && move(pleHistory, at, pleBytes);
        ok = ok && waitStream() == 0;
        if(keep) savedTokens[slot] = tokens.size();
        else tokens.resize(std::min(tokens.size(), savedTokens[slot]));
        if(!keep){
            draftPendingAt = -1;
            draftConsumed = true;
        }
        return ok;
    }

    bool saveState(int slot = 0) {
        return moveState(slot, true);
    }

    bool restoreState(int slot = 0) {
        return moveState(slot, false);
    }

    std::vector<float> forwardAll(const std::vector<int>& ids, const std::vector<int>& positions) {
        if(ids.size() > (size_t)verifyLimit){
            problem = "at most " + std::to_string(verifyLimit) + " tokens can be verified per call";
            return {};
        }
        everyRow = true;
        std::vector<float> all = forward(ids, positions);
        everyRow = false;
        return all;
    }

    std::vector<float> forward(const std::vector<int>& ids, const std::vector<int>& positions) {
        if(gemma) return forwardGemma(ids, positions);
        if(hy4) return forwardHy4(ids, positions);
        settleTail();
        std::vector<float> logits;
        if(!ready || ids.empty() || ids.size() != positions.size()) return logits;
        int highest = 0;
        for(int p : positions){
            if(p < 0 || p >= contextLimit){
                problem = "position " + std::to_string(p) + " is past the context limit of " + std::to_string(contextLimit);
                return logits;
            }
            highest = std::max(highest, p);
        }
        if(indexHeads > 0 && (budget < 4 || budget > 4096)){
            problem = "indexer budget must be between 4 and 4096 cells";
            return logits;
        }
        if(!useCuda()){
            problem = "cuda context is not available";
            return logits;
        }
        if(!reachContext(highest + 1)) return logits;
        if(firstLayer > 0 && !remote){
            problem = "the pool node that holds the first layers is not connected";
            return logits;
        }

        CudaQueue& queue = cudaQueue();
        failed = false;
        replayRows = 0;
        cutKept[0] = cutKept[1] = false;
        if(remote) remote->problem.clear();
        if(drafting && !draftConsumed && draftPendingAt >= 0 && draftPendingAt == positions[0] - 1){
            int next = ids[0];
            int place = draftPendingAt;
            draftCells(draftPending, &next, &place, 1, nullptr);
            draftConsumed = true;
        }
        settle(ids.size() > (size_t)relayLimit);
        placePictures(ids, positions);
        layerMs.assign(layers.size(), 0.0);
        embedMs = 0;
        headMs = 0;
        pleMs = 0;
        expertMs = 0;
        waitMs = 0;
        spilledRows = 0;
        missedRoutes = 0;
        missedUnique = 0;

        int n = 0;
        bool relayed = false;
        bool settled = streamPrefill && fine && !exact && ids.size() > (size_t)chunk;
        bool lending = settled && !remote && firstLayer == 0 && ids.size() >= (size_t)loanFrom && lend((int)ids.size());
        struct Payback {
            GpuModel* model;
            bool on;
            ~Payback() { if(on) model->repay(); }
        } payback{this, lending};
        bool tailLater = deferTail && drafting && !everyRow && !remote && ids.size() > (size_t)relayLimit;
        if(settled){
            pipeFront = firstLayer > 0 && ids.size() > (size_t)superLimit && canPack(ids.data(), (int)ids.size());
            for(size_t at = 0; at < ids.size() && !failed && !stopGenerating; at += superLimit){
                n = (int)std::min(ids.size() - at, (size_t)superLimit);
                superChunk(ids, positions, at, n);
                if(!drafting || failed) continue;
                if(tailLater && at + n >= ids.size()){
                    feedIds = ids;
                    feedPositions = positions;
                    feedAt = at;
                    feedRows = n;
                    feedWaiting = true;
                    continue;
                }
                draftFeed(ids, positions, at, n);
            }
            // si se corto con una parte en camino se recibe igual, la conexion no puede quedar con una respuesta de mas,
            // y como el nodo ya avanzo con esa parte y esta pc no, los dos empiezan de cero para quedar iguales
            if(frontAhead){
                frontAhead = false;
                if(!frontPending.get() && problem.empty()) problem = remote->problem;
                reset();
            }
            pipeFront = false;
        }
        bool relaying = fine && !exact && ((!profile && useGraphs) || relayAlways);
        bool quick = relaying && !everyRow && !dense && !foresee && !remote && !settled && ids.size() > (size_t)relayLimit && ids.size() <= (size_t)promptRelay && 2 * ids.size() <= (size_t)chunk && (uint64_t)ids.size() * expertsUsed <= planWidth;
        int step = quick
            ? (int)ids.size()
            : !everyRow && ids.size() <= (size_t)shortRead && fine && !exact && !profile && useGraphs
                ? std::min(relayLimit, graphRows)
                : chunk;
        for(size_t at = settled ? ids.size() : 0; at < ids.size() && !failed && !stopGenerating; at += step){
            n = (int)std::min(ids.size() - at, (size_t)step);
            size_t first = tokens.size();
            tokens.insert(tokens.end(), ids.begin() + at, ids.begin() + at + n);

            auto clock = std::chrono::steady_clock::now();
            pool.run(n, [&](int t){
                embedCell(ids[at + t], hostEmbed + (size_t)t * hidden);
            });
            int top = 0;
            for(int t = 0; t < n; t++){
                hostPositions[t] = positions[at + t];
                top = std::max(top, hostPositions[t]);
            }
            embedMs += since(clock);
            if(usePle) pleRows(n, first);
            if(remote && !pullFront(n, everyRow && n > 1, ids.data() + at, everyRow && n > 1 && settling)) break;

            if((n <= relayLimit || quick) && relaying){
                recording = everyRow && n > 1;
                leaveState = recording && settling;
                stageRing = n > relayLimit ? 2 : 0;
                relayToken(top, n);
                stageRing = 0;
                replayRows = recording ? n : 0;
                recording = false;
                leaveState = false;
                relayed = true;
                if(drafting && !everyRow && at + n < ids.size()) draftFeed(ids, positions, at, n);
                continue;
            }

            fixed = false;
            for(int l = firstLayer; l < endLayer; l++){
                GpuLayer& L = layers[l];
                clock = std::chrono::steady_clock::now();
                front(L, l, n, top);
                if(!dense){
                    if(queue.eventRecord(routed, stream) != 0) failed = true;
                    auto waited = std::chrono::steady_clock::now();
                    if(queue.eventSync(routed) != 0) failed = true;
                    waitMs += since(waited);
                    gather(L, l, n);
                    resident(L, n);
                    spill(L, n);
                }
                layerMs[l] += since(clock);
            }
            if(tiedDraft && at + n < ids.size()){
                finish(n, endLayer - 1);
                draftFeed(ids, positions, at, n);
            }
        }

        if(stopGenerating && !failed){
            feedWaiting = false;
            cudaLink().sync();
            problem = "stopped";
            return logits;
        }
        auto clock = std::chrono::steady_clock::now();
        if(failed){
            feedWaiting = false;
            int code = cudaLink().sync();
            problem = remote && !remote->problem.empty()
                ? remote->problem
                : "gpu forward failed" + (code != 0 ? ", cuda error " + std::to_string(code) : "") + (launchProblem.empty() ? "" : ", " + launchProblem);
            launchProblem.clear();
            if(remote) reset();
            return logits;
        }
        if(!relayed) finale(n, settled);
        fixed = false;
        if(drafting && !settled && relayed && everyRow && n > 1){
            cudaQueue().copyAcross(draftPending, stack + 4ULL * (n - 1) * hcWidth, (size_t)hcWidth * 4, stream);
            draftPendingAt = positions[ids.size() - 1];
            draftConsumed = false;
        } else if(drafting && !settled && tailLater){
            feedIds = ids;
            feedPositions = positions;
            feedAt = ids.size() - n;
            feedRows = n;
            feedWaiting = true;
        } else if(drafting && !settled) draftFeed(ids, positions, ids.size() - n, n);
        int syncCode = waitStream();
        uint32_t alarm = hostFlags ? hostFlags[48] : 0;
        bool alarmed = alarm != 0;
        if(alarmed) hostFlags[48] = 0;
        if(syncCode != 0 || failed || alarmed){
            feedWaiting = false;
            problem = (alarmed ? "a gpu wait for the cpu experts timed out at layer " + std::to_string((int)(alarm % 64u) - 1) : std::string("gpu forward failed"))
                + (syncCode != 0 ? ", cuda error " + std::to_string(syncCode) : "")
                + (launchProblem.empty() ? "" : ", " + launchProblem);
            launchProblem.clear();
            return logits;
        }
        logits.assign(hostLogits, hostLogits + (size_t)(everyRow ? n : 1) * (shortlist > 0 ? 2 * shortlist : head.rows));
        if(lending && feedWaiting){
            payback.on = false;
            repayWaiting = true;
        }
        residualRows = stack;
        residualCount = n;
        residualStart = (int)ids.size() - n;
        headMs = since(clock);
        sinceAdapt += (int)ids.size();
        refresh();
        if(exchangeReady()){
            auto begun = std::chrono::steady_clock::now();
            if(ids.size() <= (size_t)verifyLimit) exchange((int)ids.size());
            exchangeMs += since(begun);
        } else if(adaptEvery > 0 && sinceAdapt >= adaptEvery){
            sinceAdapt = 0;
            adapt(std::max(adaptSwaps, std::min(1024, (int)ids.size() / 2)));
        }
        heatClock += ids.size();
        if(heatClock >= 2048){
            heatClock = 0;
            saveHeat();
        }
        for(size_t i = 0; i < markUsed; i++){
            float ms = 0, idle = 0;
            cudaQueue().eventElapsed(&ms, marks[i].second.first, marks[i].second.second);
            kernelMs[marks[i].first] += ms;
            kernelCalls[marks[i].first]++;
            if(i == 0 || cudaQueue().eventElapsed(&idle, marks[i - 1].second.second, marks[i].second.first) != 0 || idle < 0.02f) continue;
            kernelMs["idle before " + marks[i].first] += idle;
            kernelCalls["idle before " + marks[i].first]++;
        }
        markUsed = 0;
        return logits;
    }
};

#include "gemma4run.h"
#include "hy4run.h"
#include "hy4draft.h"

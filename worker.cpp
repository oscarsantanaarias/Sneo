#include "src/net.h"
#include "src/json.h"
#include "src/residency.h"
#include "src/gpumodel.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <atomic>
#include <random>
#include <deque>
#include <condition_variable>
#ifdef _WIN32
#include <shellapi.h>
#include <psapi.h>

#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "psapi.lib")
#endif

namespace fs = std::filesystem;

struct WorkerState {
    std::mutex lock;
    std::string cacheRoot;
    std::string modelId;
    std::string planText;
    int firstLayer = -1;
    int lastLayer = -1;
    uint64_t storedBytes = 0;
    std::atomic<bool> busy{false};
    Residency store;
    NodeStats specs = {};
    NodeStats latest = {};
    std::mutex latestLock;
    int tcpPort = 9090;
    int pagePort = 9096;
    std::string memberFile = "membership.json";
    std::string poolKey;
    std::string token;
    std::string requestId;
    std::string host;
    int doorPort = 9097;
    std::string status = "idle";
    std::string problem;
    int round = 0;
    SOCKET wire = INVALID_SOCKET;
    std::string master;
    std::mutex modelLock;
    std::shared_ptr<GpuModel> model;
    std::string modelPlan;
    std::string modelState = "none";
    std::string modelProblem;
    std::atomic<bool> computing{false};
    SOCKET computeWire = INVALID_SOCKET;
    std::atomic<bool> leaving{false};
    std::atomic<bool> told{false};
};

void dropModel(WorkerState& state);

const char* page = R"page(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sneo worker</title>
<link rel="icon" href="data:,">
<style>
:root{--bg:#131316;--panel:#1a1a1e;--panel2:#212127;--line:#2c2c33;--edge:#34343d;--text:#eae8e4;--dim:#9a99a2;--faint:#6b6b74;--amber:#f0a63c;--amberhi:#ffc061;--amberdim:rgba(240,166,60,.14);--bad:#ef6a4c;--good:#6fbf73}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--text)}
body{min-height:100vh;display:flex;justify-content:center;padding:48px 16px;font:14px/1.5 "Segoe UI",system-ui,sans-serif}
main{width:100%;max-width:480px}
header{display:flex;align-items:center;gap:12px;margin-bottom:20px}
.mark{width:40px;height:40px;flex:none;display:grid;place-items:center;color:var(--amber);background:var(--amberdim);border:1px solid rgba(240,166,60,.4);border-radius:11px;font-weight:700}
h1{margin:0;font-size:18px;font-weight:600}
header p{margin:0;color:var(--dim);font-size:12.5px}
section{margin-bottom:12px;padding:16px;background:var(--panel);border:1px solid var(--line);border-radius:12px}
h2{margin:0 0 12px;color:var(--dim);font-size:11px;font-weight:700;letter-spacing:1.1px;text-transform:uppercase}
.specs{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.spec{min-width:0;padding:9px 11px;background:var(--panel2);border-radius:8px}
.spec span{display:block;color:var(--faint);font-size:11px;letter-spacing:.8px;text-transform:uppercase}
.spec strong{display:block;overflow:hidden;font-size:13px;font-weight:600;text-overflow:ellipsis;white-space:nowrap}
.wide{grid-column:1/-1}
.row{display:grid;grid-template-columns:1fr 92px;gap:8px;margin-bottom:12px}
label{display:flex;flex-direction:column;gap:6px;color:var(--dim);font-size:12px;font-weight:600}
input{width:100%;padding:9px 11px;color:var(--text);background:#0e0e11;border:1px solid var(--edge);border-radius:8px;outline:none;font:13px/1.4 Consolas,"Cascadia Code",monospace}
input:focus{border-color:var(--amber)}
input:disabled{opacity:.5}
button{width:100%;padding:10px 14px;border:0;border-radius:8px;cursor:pointer;font:600 13.5px/1.2 "Segoe UI",system-ui,sans-serif}
button:disabled{opacity:.4;cursor:not-allowed}
.go{color:#1a1200;background:var(--amber)}
.go:hover:not(:disabled){background:var(--amberhi)}
.out{margin-top:8px;color:var(--dim);background:transparent;border:1px solid var(--edge)}
.out:hover:not(:disabled){color:var(--bad);border-color:var(--bad)}
.status{display:flex;align-items:flex-start;gap:10px;margin:14px 0 2px;padding:10px 12px;background:var(--panel2);border-radius:8px}
.dot{width:9px;height:9px;flex:none;margin-top:6px;background:var(--faint);border-radius:50%}
.wait .dot{background:var(--amber);animation:beat 1.4s ease-in-out infinite}
.good .dot{background:var(--good)}
.bad .dot{background:var(--bad)}
.status strong{display:block;font-size:13.5px;font-weight:600}
.status em{display:block;color:var(--dim);font-size:12px;font-style:normal}
footer{color:var(--faint);font-size:12px;text-align:center}
@keyframes beat{50%{opacity:.25}}
</style>
</head>
<body>
<main>
<header>
<div class="mark">M</div>
<div>
<h1>Sneo worker</h1>
<p id="name">Reading this machine...</p>
</div>
</header>
<section>
<h2>This machine</h2>
<div class="specs">
<div class="spec wide"><span>GPU</span><strong id="gpu">-</strong></div>
<div class="spec"><span>VRAM</span><strong id="vram">-</strong></div>
<div class="spec"><span>RAM</span><strong id="ram">-</strong></div>
<div class="spec wide"><span>CPU</span><strong id="cpu">-</strong></div>
</div>
</section>
<section>
<h2>Pool</h2>
<div class="row">
<label>Pool host IPv4<input id="host" placeholder="192.168.1.20" autocomplete="off" spellcheck="false"></label>
<label>Port<input id="door" value="9097" inputmode="numeric"></label>
</div>
<button id="ask" class="go" type="button">Request to join this pool</button>
<div id="status" class="status"><i class="dot"></i><div><strong id="line">Not in a pool</strong><em id="hint"></em></div></div>
<button id="leave" class="out" type="button" disabled>Leave pool</button>
</section>
<footer id="port"></footer>
</main>
<script>
const field = (id) => document.getElementById(id);
const gb = (bytes) => `${Math.round(Number(bytes || 0) / 1073741824)} GB`;
let typed = false;
let sending = false;
let notice = null;

const lines = {
  idle: () => ['Not in a pool', 'Type the IPv4 address of the PC that runs the pool and send a request.', ''],
  sending: () => ['Sending the request…', '', 'wait'],
  waiting: (s) => ['Waiting for the pool owner to accept…', `Request sent to ${s.host}:${s.doorPort}`, 'wait'],
  joined: (s) => [`Joined pool at ${s.host}`, s.linked ? 'Connected, the pool can use this machine.' : 'Waiting for the pool to connect.', 'good'],
  declined: () => ['Request declined', 'The pool owner said no. You can send another request.', 'bad'],
  unreachable: (s) => ['Pool host unreachable', `Nothing answers on ${s.host}:${s.doorPort}. Check the address and that the pool takes requests.`, 'bad'],
  expired: () => ['Request expired', 'The pool no longer has this request. Send it again.', 'bad'],
  failed: (s) => ['Request refused', s.problem || 'The pool host refused the request.', 'bad']
};

const paint = (s) => {
  const pick = lines[s.status] || lines.idle;
  const [line, hint, tone] = notice || pick(s);
  const joined = s.status === 'joined';

  field('name').textContent = s.name;
  field('gpu').textContent = s.gpu || 'No GPU';
  field('vram').textContent = s.gpu ? gb(s.vramTotal) : '-';
  field('ram').textContent = gb(s.ramTotal);
  field('cpu').textContent = `${String(s.cpu || 'CPU').trim()} · ${s.cores}C / ${s.threads}T`;
  field('port').textContent = `The pool connects to this machine on TCP port ${s.tcpPort}`;
  field('line').textContent = line;
  field('hint').textContent = hint;
  field('status').className = `status ${tone}`;
  field('ask').disabled = joined || sending || s.status === 'sending';
  field('host').disabled = joined;
  field('door').disabled = joined;
  field('leave').disabled = s.status === 'idle';

  if(!typed && s.host){
    field('host').value = s.host;
    field('door').value = s.doorPort;
  }
};

const refresh = async () => {
  try {
    const req = await fetch('/state', {cache: 'no-store'});
    paint(await req.json());
  } catch(e) {
    field('line').textContent = 'worker.exe is not running';
    field('hint').textContent = 'Start worker.exe again and reload this page.';
    field('status').className = 'status bad';
  }
};

const send = async (path, body) => {
  const req = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {})
  });
  const res = await req.json();

  if(!req.ok){
    notice = [res.error || 'Something went wrong', '', 'bad'];
    return refresh();
  }

  paint(res);
};

const edited = () => {
  typed = true;
  notice = null;
};

field('host').addEventListener('input', edited);
field('door').addEventListener('input', edited);

field('ask').addEventListener('click', async () => {
  const host = field('host').value.trim();
  const door = Number(field('door').value);
  notice = null;

  if(!/^(\d{1,3}\.){3}\d{1,3}$/.test(host)){
    notice = ['Type the IPv4 address of the pool host', 'It looks like 192.168.1.20. The pool owner finds it in the Pool view, under Network.', 'bad'];
    return refresh();
  }

  if(!Number.isInteger(door) || door < 1 || door > 65535){
    notice = ['The port must be a number from 1 to 65535', 'Leave 9097 unless the pool owner changed it.', 'bad'];
    return refresh();
  }

  sending = true;
  field('ask').disabled = true;
  field('line').textContent = 'Sending the request…';
  field('hint').textContent = '';
  field('status').className = 'status wait';

  try {
    await send('/join', {host, doorPort: door});
  } catch(e) {
    notice = ['worker.exe did not answer', 'Start worker.exe again and reload this page.', 'bad'];
  }

  sending = false;
  refresh();
});

field('leave').addEventListener('click', async () => {
  notice = null;

  try {
    await send('/leave');
  } catch(e) {
    refresh();
  }
});

refresh();
setInterval(refresh, 1500);
</script>
</body>
</html>
)page";

void saveMembership(const WorkerState& state) {
    if(state.poolKey.empty()){
        std::error_code code;
        fs::remove(state.memberFile, code);
        return;
    }
    std::ofstream(state.memberFile, std::ios::trunc) << "{\"host\":" << quoteJson(state.host) << ",\"doorPort\":" << state.doorPort << ",\"key\":" << quoteJson(state.poolKey) << "}";
}

int askDoor(const std::string& host, int port, const std::string& method, const std::string& path, const std::string& body, std::string& reply) {
    SOCKET s = connectTcp(host, port, 3000);
    if(s == INVALID_SOCKET) return 0;

    DWORD timeout = 6000;
    socketWait(s, SO_RCVTIMEO, timeout);

    std::string out = method + " " + path + " HTTP/1.1\r\n"
                      "Host: " + host + ":" + std::to_string(port) + "\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: " + std::to_string(body.size()) + "\r\n"
                      "Connection: close\r\n\r\n" + body;

    if(!sendAll(s, out.data(), out.size())){
        closesocket(s);
        return 0;
    }

    std::string raw;
    char buffer[4096];
    while(raw.size() < 65536){
        int got = recv(s, buffer, sizeof(buffer), 0);
        if(got <= 0) break;
        raw.append(buffer, got);
    }
    closesocket(s);

    size_t headerEnd = raw.find("\r\n\r\n");
    if(raw.rfind("HTTP/1.", 0) != 0 || headerEnd == std::string::npos) return 0;
    reply = raw.substr(headerEnd + 4);
    return atoi(raw.c_str() + 9);
}

void followRequest(WorkerState& state, int round, std::string host, int doorPort, std::string id) {
    int misses = 0;

    while(true){
        std::this_thread::sleep_for(std::chrono::seconds(2));
        {
            std::lock_guard<std::mutex> guard(state.lock);
            if(state.round != round || state.status == "joined") return;
        }

        std::string reply;
        int code = askDoor(host, doorPort, "GET", "/join/" + id, "", reply);
        std::string answer = parseJson(reply)["status"].str();

        std::lock_guard<std::mutex> guard(state.lock);
        if(state.round != round || state.status == "joined") return;

        if(code == 0){
            misses++;
            if(misses >= 5) state.status = "unreachable";
            continue;
        }

        misses = 0;
        if(state.status == "unreachable") state.status = "waiting";

        if(code == 404){
            state.token.clear();
            state.status = "expired";
            return;
        }

        if(answer == "declined"){
            state.token.clear();
            state.status = "declined";
            std::cout << "the pool at " << host << " declined the request" << std::endl;
            return;
        }

        if(answer == "accepted"){
            state.poolKey = state.token;
            state.token.clear();
            state.status = "joined";
            saveMembership(state);
            return;
        }
    }
}

void startJoin(WorkerState& state, std::string host, int doorPort) {
    std::random_device seed;
    const char* digits = "0123456789abcdef";
    std::string token;
    for(int i = 0; i < 32; i++) token.push_back(digits[seed() % 16]);

    int round = 0;
    {
        std::lock_guard<std::mutex> guard(state.lock);
        round = ++state.round;
        state.token = token;
        state.requestId.clear();
        state.host = host;
        state.doorPort = doorPort;
        state.status = "sending";
        state.problem.clear();
    }

    std::ostringstream out;
    out << "{\"name\":" << quoteJson(state.specs.host.nodeName) << ",\"tcpPort\":" << state.tcpPort << ",\"token\":" << quoteJson(token)
        << ",\"gpu\":" << quoteJson(state.specs.gpu.present ? state.specs.gpu.name : "") << ",\"vramTotal\":" << state.specs.gpu.vramTotal
        << ",\"ramTotal\":" << state.specs.host.ramTotal << ",\"cpu\":" << quoteJson(state.specs.host.cpuName) << ",\"version\":" << protocolVersion << "}";

    std::cout << "asking the pool at " << host << ":" << doorPort << " to let this machine in" << std::endl;
    std::string reply;
    int code = askDoor(host, doorPort, "POST", "/join", out.str(), reply);
    Json answer = parseJson(reply);

    std::lock_guard<std::mutex> guard(state.lock);
    if(state.round != round) return;

    if(code == 200 && !answer["id"].str().empty()){
        state.requestId = answer["id"].str();
        if(state.status == "sending") state.status = "waiting";
        std::thread(followRequest, std::ref(state), round, host, doorPort, state.requestId).detach();
        return;
    }

    state.token.clear();
    state.status = code == 0 ? "unreachable" : "failed";
    state.problem = answer["error"].str("the pool host answered " + std::to_string(code));
}

std::string stateJson(WorkerState& state) {
    std::lock_guard<std::mutex> guard(state.lock);
    std::ostringstream out;
    out << "{\"name\":" << quoteJson(state.specs.host.nodeName) << ",\"gpu\":" << quoteJson(state.specs.gpu.present ? state.specs.gpu.name : "")
        << ",\"vramTotal\":" << state.specs.gpu.vramTotal << ",\"ramTotal\":" << state.specs.host.ramTotal << ",\"cpu\":" << quoteJson(state.specs.host.cpuName)
        << ",\"cores\":" << state.specs.host.cores << ",\"threads\":" << state.specs.host.threads << ",\"tcpPort\":" << state.tcpPort
        << ",\"status\":" << quoteJson(state.status) << ",\"problem\":" << quoteJson(state.problem) << ",\"host\":" << quoteJson(state.host)
        << ",\"doorPort\":" << state.doorPort << ",\"linked\":" << (state.master.empty() ? "false" : "true") << "}";
    return out.str();
}

void servePage(SOCKET client, WorkerState& state) {
    DWORD timeout = 5000;
    socketWait(client, SO_RCVTIMEO, timeout);

    std::string raw;
    std::string head;
    char buffer[4096];
    size_t headerEnd = std::string::npos;
    size_t wanted = 0;

    while(raw.size() < 16384){
        int got = recv(client, buffer, sizeof(buffer), 0);
        if(got <= 0) break;
        raw.append(buffer, got);
        if(headerEnd == std::string::npos){
            headerEnd = raw.find("\r\n\r\n");
            if(headerEnd == std::string::npos) continue;
            head = raw.substr(0, headerEnd);
            std::transform(head.begin(), head.end(), head.begin(), ::tolower);
            size_t at = head.find("content-length:");
            wanted = at == std::string::npos ? 0 : (size_t)atoll(head.c_str() + at + 15);
        }
        if(raw.size() >= headerEnd + 4 + wanted) break;
    }

    auto reply = [&](int status, const char* type, const std::string& body) {
        std::string out = "HTTP/1.1 " + std::to_string(status) + (status == 200 ? " OK" : " Error") + "\r\n"
                          "Content-Type: " + type + "\r\n"
                          "Cache-Control: no-store\r\n"
                          "X-Frame-Options: DENY\r\n"
                          "Content-Length: " + std::to_string(body.size()) + "\r\n"
                          "Connection: close\r\n\r\n" + body;
        sendAll(client, out.data(), out.size());
        shutdown(client, SD_SEND);
        closesocket(client);
    };

    if(headerEnd == std::string::npos){
        closesocket(client);
        return;
    }

    std::string self = std::to_string(state.pagePort);
    size_t hostAt = head.find("\r\nhost:");
    size_t originAt = head.find("\r\norigin:");
    std::string hostHeader = hostAt == std::string::npos ? "" : head.substr(hostAt + 7, head.find("\r\n", hostAt + 2) - hostAt - 7);
    std::string origin = originAt == std::string::npos ? "" : head.substr(originAt + 9, head.find("\r\n", originAt + 2) - originAt - 9);
    hostHeader.erase(0, hostHeader.find_first_not_of(' '));
    origin.erase(0, origin.find_first_not_of(' '));

    bool ownHost = hostHeader == "127.0.0.1:" + self || hostHeader == "localhost:" + self;
    bool ownOrigin = origin.empty() || origin == "http://127.0.0.1:" + self || origin == "http://localhost:" + self;
    if(!ownHost || !ownOrigin){
        reply(403, "application/json", "{\"error\":\"this page only answers to this PC\"}");
        return;
    }

    size_t space = raw.find(' ');
    size_t second = raw.find(' ', space + 1);
    std::string route = space == std::string::npos || second == std::string::npos ? "" : raw.substr(0, space) + " " + raw.substr(space + 1, second - space - 1);
    Json body = parseJson(raw.substr(headerEnd + 4));

    if(route == "GET /" || route == "GET /index.html"){
        reply(200, "text/html; charset=utf-8", page);
        return;
    }

    if(route == "GET /state"){
        reply(200, "application/json", stateJson(state));
        return;
    }

    if(route == "POST /join"){
        std::string poolHost = body["host"].str();
        double door = body["doorPort"].num(9097);
        in_addr probe = {};

        if(inet_pton(AF_INET, poolHost.c_str(), &probe) != 1 || door < 1 || door > 65535 || door != floor(door)){
            reply(400, "application/json", "{\"error\":\"Type the IPv4 address of the pool host and a port from 1 to 65535\"}");
            return;
        }

        bool joined = false;
        {
            std::lock_guard<std::mutex> guard(state.lock);
            joined = state.status == "joined";
        }

        if(joined){
            reply(409, "application/json", "{\"error\":\"Leave the current pool first\"}");
            return;
        }

        startJoin(state, poolHost, (int)door);
        reply(200, "application/json", stateJson(state));
        return;
    }

    if(route == "POST /leave"){
        bool linked = false;
        {
            std::lock_guard<std::mutex> guard(state.lock);
            linked = state.wire != INVALID_SOCKET;
        }
        state.told = false;
        state.leaving = linked;
        for(int i = 0; i < 30 && linked && !state.told; i++) Sleep(100);
        state.leaving = false;
        {
            std::lock_guard<std::mutex> guard(state.lock);
            state.round++;
            state.poolKey.clear();
            state.token.clear();
            state.requestId.clear();
            state.status = "idle";
            state.problem.clear();
            state.master.clear();
            saveMembership(state);
            if(state.wire != INVALID_SOCKET) shutdown(state.wire, SD_BOTH);
            if(state.computeWire != INVALID_SOCKET) shutdown(state.computeWire, SD_BOTH);
        }
        state.store.generation++;
        releaseAll(state.store);
        dropModel(state);
        std::cout << "left the pool" << (state.told ? ", the pool host was told," : "") << " memory released" << std::endl;
        reply(200, "application/json", stateJson(state));
        return;
    }

    if(route == "POST /quit"){
        {
            std::lock_guard<std::mutex> guard(state.lock);
            state.round++;
            if(state.wire != INVALID_SOCKET) shutdown(state.wire, SD_BOTH);
            if(state.computeWire != INVALID_SOCKET) shutdown(state.computeWire, SD_BOTH);
        }
        state.store.generation++;
        releaseAll(state.store);
        dropModel(state);
        std::cout << "closing, memory released" << std::endl;
        reply(200, "application/json", "{\"closing\":true}");
        std::thread([]{
            Sleep(300);
            ExitProcess(0);
        }).detach();
        return;
    }

    reply(404, "application/json", "{\"error\":\"not found\"}");
}

std::string cacheFile(const WorkerState& state, const std::string& tensor);

void dropModel(WorkerState& state) {
    std::lock_guard<std::mutex> guard(state.modelLock);
    state.model.reset();
    state.modelPlan.clear();
    state.modelState = "none";
    state.modelProblem.clear();
}

void loadFront(WorkerState& state, std::string planText) {
    {
        std::lock_guard<std::mutex> guard(state.modelLock);
        if(state.modelState == "ready" && state.modelPlan == planText && state.model) return;
        state.model.reset();
        state.modelPlan.clear();
        state.modelState = "loading";
        state.modelProblem.clear();
    }

    Json plan = parseJson(planText);
    const Json& front = plan["front"];
    ModelFile file;
    file.arch = front["arch"].str("qwen4exp");
    file.layers = (uint32_t)front["layers"].num(48);
    file.experts = (uint32_t)front["experts"].num(512);
    file.expertsUsed = (uint32_t)front["expertsUsed"].num(10);
    file.hidden = (uint32_t)front["hidden"].num(2560);
    MetaValue value;
    value.whole = (int64_t)front["interval"].num(4);
    file.meta[file.arch + ".full_attention_interval"] = value;
    value.whole = (int64_t)front["topK"].num(2048);
    file.meta[file.arch + ".attention.indexer.top_k"] = value;
    value.whole = 0;
    value.real = front["ropeBase"].num(1e7);
    file.meta[file.arch + ".rope.freq_base"] = value;
    for(const auto& entry : front["meta"].fields){
        MetaValue shape;
        shape.whole = !entry.second.items.empty() ? (int64_t)entry.second.items[0].num() : 0;
        shape.real = entry.second.items.size() > 1 ? entry.second.items[1].num() : 0;
        file.meta[entry.first] = shape;
    }

    for(const Json& t : plan["tensors"].items){
        TensorInfo info;
        info.name = t["name"].str();
        info.type = (uint32_t)t["type"].num();
        info.bytes = (uint64_t)t["bytes"].num();
        info.dimCount = (uint32_t)std::min<size_t>(4, t["dims"].items.size());
        info.elements = 1;
        for(uint32_t d = 0; d < info.dimCount; d++){
            info.dims[d] = (uint64_t)t["dims"].items[d].num(1);
            info.elements *= info.dims[d];
        }
        info.file = (uint32_t)file.paths.size();
        info.offset = 0;
        size_t block = info.name.find("blk.");
        info.layer = block == std::string::npos ? -1 : atoi(info.name.c_str() + block + 4);
        info.expert = info.name.find("_exps") != std::string::npos;
        file.paths.push_back(cacheFile(state, info.name));
        file.lookup[info.name] = file.tensors.size();
        file.totalBytes += info.bytes;
        file.tensors.push_back(info);
    }

    auto next = std::make_shared<GpuModel>();
    next->firstLayer = 0;
    next->endLayer = (int)plan["lastLayer"].num(-1) + 1;
    next->contextLimit = (int)front["context"].num(32768);
    next->relayLimit = (int)front["relayLimit"].num(4);
    next->tailSkip = (int)front["tailSkip"].num(0);
    next->foresee = front["foresee"].truthy(false);
    next->vramCap = (uint64_t)front["vramCap"].num(0);
    next->hostBudget = (uint64_t)front["ramCap"].num(0);
    next->streamPrefill = true;
    next->superLimit = 2048;
    next->heatTokens = (uint64_t)front["heatTokens"].num(0);
    const Json& heat = front["heat"];
    if(!heat.items.empty()){
        next->heat.assign((size_t)file.layers * file.experts, 0);
        for(size_t l = 0; l < heat.items.size() && l < file.layers; l++){
            for(size_t e = 0; e < heat.items[l].items.size() && e < file.experts; e++) next->heat[l * file.experts + e] = (uint64_t)heat.items[l].items[e].num();
        }
    }

    uint64_t start = nowMs();
    std::cout << "loading layers 0.." << next->endLayer - 1 << " on the gpu..." << std::endl;
    int threads = (int)std::max<uint32_t>(4, state.specs.host.cores);
    bool ok = next->loadFile(file, file.paths.empty() ? "" : file.paths[0], threads);

    std::lock_guard<std::mutex> guard(state.modelLock);
    if(!ok){
        state.modelState = "failed";
        state.modelProblem = next->problem.empty() ? "the layers did not load" : next->problem;
        std::cout << "layers failed to load: " << state.modelProblem << std::endl;
        return;
    }
    state.model = next;
    state.modelPlan = planText;
    state.modelState = "ready";
    std::cout << "layers 0.." << next->endLayer - 1 << " ready in " << (nowMs() - start) / 1000.0 << " s: " << next->residentExperts << " experts per layer in vram, " << next->warmPerLayer << " pinned in ram" << std::endl;
}

// los nombres que manda el master solo pueden ser nombres de archivo dentro de la carpeta del cache
bool cacheName(const std::string& name) {
    if(name.empty() || name.size() > 200 || name[0] == '.' || name.find("..") != std::string::npos) return false;
    for(unsigned char c : name){
        if(!isalnum(c) && c != '.' && c != '_' && c != '-') return false;
    }
    return true;
}

void loadFromCache(WorkerState& state, std::string planText) {
    Json plan = parseJson(planText);
    if(plan["role"].str() == "front"){
        state.store.generation++;
        releaseAll(state.store);
        loadFront(state, planText);
        return;
    }
    dropModel(state);
    std::vector<TensorSource> sources;
    std::vector<LayerShare> layers;
    if(!cacheName(plan["modelId"].str())){
        std::cout << "the plan names a model folder outside the cache, it was ignored" << std::endl;
        return;
    }

    for(const Json& t : plan["tensors"].items){
        TensorSource s;
        s.name = t["name"].str();
        if(!cacheName(s.name)){
            std::cout << "the plan names a file outside the cache, it was ignored" << std::endl;
            return;
        }
        s.path = (fs::path(state.cacheRoot) / plan["modelId"].str() / (s.name + ".bin")).string();
        s.bytes = (uint64_t)t["bytes"].num();
        size_t block = s.name.find("blk.");
        s.layer = block == std::string::npos ? -1 : atoi(s.name.c_str() + block + 4);
        s.expert = s.name.find("_exps") != std::string::npos;
        sources.push_back(s);
    }

    for(const Json& l : plan["layers"].items){
        LayerShare share;
        share.layer = (int)l["layer"].num();
        share.denseTier = (int)l["dense"].num();
        share.gpuExperts = (int)l["gpuExperts"].num();
        share.ramExperts = (int)l["ramExperts"].num();
        share.ssdExperts = (int)l["ssdExperts"].num();
        layers.push_back(share);
    }

    uint64_t start = nowMs();
    loadShare(state.store, sources, layers, false, SLOT_RAM, SLOT_RAM, (uint32_t)plan["expertCount"].num());
    std::lock_guard<std::mutex> guard(state.store.lock);
    std::cout << "share " << state.store.state << ": " << (double)state.store.gpuBytes / (1 << 30) << " GB vram, " << (double)state.store.ramBytes / (1 << 30) << " GB ram in " << (nowMs() - start) / 1000.0 << " s" << (state.store.problem.empty() ? "" : " | " + state.store.problem) << std::endl;
}

std::string cacheFile(const WorkerState& state, const std::string& tensor) {
    return (fs::path(state.cacheRoot) / state.modelId / (tensor + ".bin")).string();
}

// lo que el nodo manda por partes sale en orden desde un hilo aparte
struct Outbox {
    std::mutex lock;
    std::condition_variable moved;
    std::deque<std::vector<char>> queue;
    bool closed = false;
    std::atomic<bool> broken{false};

    void push(std::vector<char> data) {
        std::lock_guard<std::mutex> guard(lock);
        queue.push_back(std::move(data));
        moved.notify_one();
    }

    void close() {
        std::lock_guard<std::mutex> guard(lock);
        closed = true;
        moved.notify_one();
    }

    void run(SOCKET wire) {
        while(true){
            std::vector<char> next;
            {
                std::unique_lock<std::mutex> guard(lock);
                moved.wait(guard, [&]{ return !queue.empty() || closed; });
                if(queue.empty()) return;
                next = std::move(queue.front());
                queue.pop_front();
            }
            if(!broken && !sendAll(wire, next.data(), next.size())) broken = true;
        }
    }
};

void serveCompute(SOCKET master, WorkerState& state, const std::string& address, HelloBody hello) {
    tuneSocket(master);
    hello.key[sizeof(hello.key) - 1] = 0;
    std::string key = hello.key;
    bool member = false;
    {
        std::lock_guard<std::mutex> guard(state.lock);
        member = !key.empty() && key == state.poolKey;
    }
    auto refuse = [&](const std::string& why) {
        sendPacket(master, PACKET_REFUSED, why.data(), why.size());
        closesocket(master);
    };
    if(hello.version != protocolVersion){
        refuse("protocol mismatch, update worker.exe");
        return;
    }
    if(!member){
        refuse("this machine is not in your pool");
        return;
    }
    if(state.computing.exchange(true)){
        refuse("this machine already computes layers for a pool");
        return;
    }

    uint64_t start = nowMs();
    std::shared_ptr<GpuModel> model;
    std::string problem;
    while(true){
        {
            // el master puede abrir esta conexion apenas mando el plan, antes de que la carga empiece: se espera un poco
            std::lock_guard<std::mutex> guard(state.modelLock);
            if(state.modelState == "ready") model = state.model;
            if(state.modelState == "failed") problem = state.modelProblem;
            if(state.modelState == "none" && nowMs() - start > 30000) problem = "no layers were sent to this machine";
        }
        if(model || !problem.empty()) break;
        if(nowMs() - start > 14ULL * 60 * 1000){
            problem = "the layers took too long to load";
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if(!model){
        refuse(problem);
        state.computing = false;
        return;
    }

    // las capas que tiene, cuantas filas puede verificar por pasada esta tarjeta y lo que sabe hacer con las filas
    // un master viejo lee solo los tres primeros
    int32_t range[4] = {model->firstLayer, model->endLayer, model->relayLimit, (int32_t)rowsFeatures};
    DWORD forever = 0;
    socketWait(master, SO_RCVTIMEO, forever);
    if(!sendPacket(master, PACKET_READY, range, sizeof(range))){
        closesocket(master);
        state.computing = false;
        return;
    }
    {
        std::lock_guard<std::mutex> guard(state.lock);
        state.computeWire = master;
    }
    std::cout << "computing layers " << range[0] << ".." << range[1] - 1 << " for the pool at " << address << std::endl;

    PacketHead head;
    std::vector<char> body;
    std::vector<float> out;
    std::vector<char> reply;
    std::vector<int> openPositions;
    std::vector<float> openEmbeds, openPles;
    std::string sticky;
    uint64_t rows = 0;
    std::string reason = "the pool said goodbye";
    while(true){
        if(!readPacket(master, head, body)){
            reason = "the connection dropped (socket error " + std::to_string(WSAGetLastError()) + ")";
            break;
        }
        if(head.kind == PACKET_STATE){
            if(body.size() < 8){
                reason = "a broken state message";
                break;
            }
            int32_t step[2];
            memcpy(step, body.data(), 8);
            bool ok = step[0] == STATE_SAVE
                ? model->saveState(step[1])
                : step[0] == STATE_RESTORE
                    ? model->restoreState(step[1])
                    : step[0] == STATE_COMMIT
                        ? model->commitRows(std::vector<int>((size_t)std::min(std::max(0, step[1]), 4096), 0))
                        : (model->reset(), true);
            if(!ok && sticky.empty()) sticky = "state step " + std::to_string(step[0]) + " failed" + (model->problem.empty() ? "" : ": " + model->problem);
            continue;
        }

        if(head.kind == PACKET_ROWS){
            int n = 0;
            bool record = false, sized = false;
            std::string why;
            const int* positions = nullptr;
            const float* embeds = nullptr;
            const float* ples = nullptr;
            if(head.flags & ROWS_PACKED){
                sized = unpackRows(body.data(), body.size(), hidden, pleHeads, pleRowWidth, openPositions, openEmbeds, openPles, n, record, why);
                positions = openPositions.data();
                embeds = openEmbeds.data();
                ples = openPles.data();
            } else {
                int32_t top[2] = {};
                if(body.size() >= 8) memcpy(top, body.data(), 8);
                n = top[0];
                record = top[1] != 0;
                uint64_t rowBytes = (uint64_t)hidden * 4;
                sized = n > 0 && body.size() == 8 + (uint64_t)n * 4 + 2ULL * n * rowBytes;
                if(!sized) why = "rows of the wrong size";
                positions = (const int*)(body.data() + 8);
                embeds = (const float*)(body.data() + 8 + (size_t)n * 4);
                ples = (const float*)(body.data() + 8 + (size_t)n * 4 + (size_t)(n * rowBytes));
            }
            out.resize(sized ? (size_t)n * hcWidth : 0);
            // por partes: un hilo manda cada parte apenas sale de la tarjeta mientras ella calcula la siguiente,
            // cada una en fp16 si el master la acepta y cabe
            bool pieces = sized && sticky.empty() && (head.flags & ROWS_PIECES) != 0;
            bool askedHalf = (head.flags & ROWS_HALF) != 0;
            bool opened = false;
            Outbox box;
            std::thread sender;
            if(pieces){
                sender = std::thread([&box, master]{ box.run(master); });
                model->rowsReady = [&](int at, int m){
                    if(box.broken) return false;
                    uint64_t cells = (uint64_t)m * hcWidth;
                    const float* part = out.data() + (size_t)at * hcWidth;
                    bool half = askedHalf && fitsHalf(part, cells);
                    size_t lead = opened ? 0 : sizeof(PacketHead) + 8;
                    std::vector<char> piece(lead + 8 + (size_t)(cells * (half ? 2 : 4)));
                    if(!opened){
                        PacketHead start = {PACKET_ROWS_DONE, (uint32_t)ROWS_PIECES, 8};
                        int32_t status[2] = {1, n};
                        memcpy(piece.data(), &start, sizeof(start));
                        memcpy(piece.data() + sizeof(start), status, 8);
                        opened = true;
                    }
                    int32_t top[2] = {m, half ? 1 : 0};
                    memcpy(piece.data() + lead, top, 8);
                    if(half) toHalf(part, (uint16_t*)(piece.data() + lead + 8), cells);
                    else memcpy(piece.data() + lead + 8, part, (size_t)cells * 4);
                    box.push(std::move(piece));
                    return true;
                };
            }
            uint64_t began = nowMs();
            bool ok = sized && sticky.empty() && model->forwardRows(n, record, (head.flags & ROWS_LEAVE) != 0, positions, embeds, ples, out.data());
            model->rowsReady = nullptr;
            if(pieces){
                if(!ok && opened){
                    why = model->problem;
                    std::vector<char> failure(8 + why.size());
                    int32_t top[2] = {-1, (int32_t)why.size()};
                    memcpy(failure.data(), top, 8);
                    memcpy(failure.data() + 8, why.data(), why.size());
                    box.push(std::move(failure));
                }
                box.close();
                sender.join();
                if(box.broken){
                    reason = "could not send " + std::to_string(n) + " rows back (socket error " + std::to_string(WSAGetLastError()) + ")";
                    break;
                }
            }
            if(ok && n >= 256) std::cout << n << " prompt rows through layers " << range[0] << ".." << range[1] - 1 << " in " << nowMs() - began << " ms" << std::endl;
            if(pieces && opened){
                if(ok) rows += n;
                else std::cout << "rows failed: " << why << std::endl;
                continue;
            }
            if(!ok){
                if(sized) why = !sticky.empty() ? sticky : model->problem;
                sticky.clear();
                int32_t status[2] = {0, 0};
                PacketHead failure = {PACKET_ROWS_DONE, 0, 8 + why.size()};
                reply.resize(sizeof(failure) + 8 + why.size());
                memcpy(reply.data(), &failure, sizeof(failure));
                memcpy(reply.data() + sizeof(failure), status, 8);
                memcpy(reply.data() + sizeof(failure) + 8, why.data(), why.size());
                std::cout << "rows failed: " << why << std::endl;
                if(!sendAll(master, reply.data(), reply.size())){
                    reason = "could not send the failure back (socket error " + std::to_string(WSAGetLastError()) + ")";
                    break;
                }
                continue;
            }
            // la pila vuelve en fp16 si el master la acepta y todo cabe, cabecera y filas en un solo envio
            uint64_t cells = (uint64_t)n * hcWidth;
            bool half = (head.flags & ROWS_HALF) && fitsHalf(out.data(), cells);
            uint64_t outBytes = cells * (half ? 2 : 4);
            int32_t status[2] = {1, n};
            PacketHead done = {PACKET_ROWS_DONE, half ? (uint32_t)ROWS_HALF : 0u, 8 + outBytes};
            reply.resize(sizeof(done) + 8 + (size_t)outBytes);
            memcpy(reply.data(), &done, sizeof(done));
            memcpy(reply.data() + sizeof(done), status, 8);
            if(half) toHalf(out.data(), (uint16_t*)(reply.data() + sizeof(done) + 8), cells);
            else memcpy(reply.data() + sizeof(done) + 8, out.data(), (size_t)outBytes);
            if(!sendAll(master, reply.data(), reply.size())){
                reason = "could not send " + std::to_string(n) + " rows back (socket error " + std::to_string(WSAGetLastError()) + ")";
                break;
            }
            rows += n;
            continue;
        }

        if(head.kind == PACKET_BYE) break;
    }

    {
        std::lock_guard<std::mutex> guard(state.lock);
        if(state.computeWire == master) state.computeWire = INVALID_SOCKET;
    }
    closesocket(master);
    state.computing = false;
    std::cout << "the pool at " << address << " stopped using layers " << range[0] << ".." << range[1] - 1 << " after " << rows << " rows: " << reason << std::endl;
}

void serveMaster(SOCKET master, WorkerState& state, const std::string& address, HelloBody hello) {
    tuneSocket(master);
    {
        std::lock_guard<std::mutex> guard(state.lock);
        state.wire = master;
    }

    PacketHead head;
    std::vector<char> body;
    std::ofstream writing;
    std::string writingName;
    hello.key[sizeof(hello.key) - 1] = 0;
    std::string key = hello.key;
    bool member = false;
    if(hello.version == protocolVersion){
        std::lock_guard<std::mutex> guard(state.lock);
        member = !key.empty() && (key == state.poolKey || key == state.token);
        if(member && key == state.token){
            state.poolKey = key;
            state.token.clear();
            state.status = "joined";
            saveMembership(state);
        }
        if(member) state.master = address;
    }

    if(!member){
        std::string why = hello.version != protocolVersion
            ? "protocol mismatch, update worker.exe"
            : "this machine did not ask to join your pool";
        sendPacket(master, PACKET_REFUSED, why.data(), why.size());
        if(hello.version == protocolVersion) std::cout << "refused " << address << ": no join request matches its key" << std::endl;
        {
            std::lock_guard<std::mutex> guard(state.lock);
            state.wire = INVALID_SOCKET;
        }
        closesocket(master);
        return;
    }

    DWORD forever = 0;
    socketWait(master, SO_RCVTIMEO, forever);
    std::cout << "pool at " << address << " connected" << std::endl;
    NodeStats greeting = readNode(state.cacheRoot);
    sendPacket(master, PACKET_NODE_INFO, &greeting, sizeof(greeting));
    bool trusted = true;

    while(readPacket(master, head, body)){
        if(head.kind == PACKET_STATS_ASK && state.leaving){
            const char left[] = "left";
            sendPacket(master, PACKET_BYE, left, 4);
            state.told = true;
            break;
        }

        if(head.kind == PACKET_STATS_ASK){
            NodeStats stats;
            {
                std::lock_guard<std::mutex> guard(state.latestLock);
                stats = state.latest;
            }
            uint64_t heldRam = stats.host.heldRam;
            uint64_t heldGpu = stats.host.heldGpu;
            residencyStats(state.store, stats.host);
            stats.host.heldRam = std::max(stats.host.heldRam, heldRam);
            stats.host.heldGpu = std::max(stats.host.heldGpu, heldGpu);
            sendPacket(master, PACKET_STATS, &stats, sizeof(stats));
            continue;
        }

        if(head.kind == PACKET_RESET){
            state.store.generation++;
            releaseAll(state.store);
            dropModel(state);
            std::cout << "memory released" << std::endl;
            continue;
        }

        if(head.kind == PACKET_PLAN){
            state.planText.assign(body.begin(), body.end());
            Json plan = parseJson(state.planText);
            bool named = cacheName(plan["modelId"].str("unknown"));
            for(const Json& t : plan["tensors"].items) named = named && cacheName(t["name"].str());
            if(!named){
                std::cout << "the master sent file names outside the cache, closing the connection" << std::endl;
                break;
            }
            state.modelId = plan["modelId"].str("unknown");
            state.firstLayer = (int)plan["firstLayer"].num(-1);
            state.lastLayer = (int)plan["lastLayer"].num(-1);
            fs::create_directories(fs::path(state.cacheRoot) / state.modelId);

            Json missing;
            missing.kind = JSON_LIST;
            uint64_t have = 0;
            std::map<std::string, uint64_t> wanted;
            for(const Json& t : plan["tensors"].items){
                std::string name = t["name"].str();
                uint64_t bytes = (uint64_t)t["bytes"].num();
                std::error_code code;
                uint64_t size = fs::file_size(cacheFile(state, name), code);
                if(!code && size == bytes){
                    have += bytes;
                    continue;
                }
                wanted[name] = bytes;
                Json item;
                item.kind = JSON_TEXT;
                item.text = name;
                missing.items.push_back(item);
            }

            if(!missing.items.empty()){
                std::vector<std::string> folders = {"modelos"};
                std::stringstream text;
                {
                    std::ifstream file("pool.json");
                    text << file.rdbuf();
                }
                std::string saved = parseJson(text.str())["folder"].str();
                if(!saved.empty()) folders.push_back(saved);

                ModelFile local;
                bool found = false;
                for(size_t f = 0; f < folders.size() && !found; f++){
                    std::error_code walk;
                    for(auto it = fs::recursive_directory_iterator(folders[f], fs::directory_options::skip_permission_denied, walk); it != fs::recursive_directory_iterator() && !found; it.increment(walk)){
                        if(walk) break;
                        std::string name = it->path().filename().string();
                        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
                        bool shard = name.find("-of-") != std::string::npos;
                        if(!it->is_regular_file(walk) || name.size() < 5 || name.substr(name.size() - 5) != ".gguf" || name.rfind("mmproj", 0) == 0 || name.rfind("mtp", 0) == 0) continue;
                        if(shard && name.find("-00001-of-") == std::string::npos) continue;
                        ModelFile candidate;
                        std::string error;
                        if(!openModel(it->path().string(), candidate, error) || modelIdOf(candidate) != state.modelId) continue;
                        local = std::move(candidate);
                        found = true;
                    }
                }

                if(found){
                    std::cout << "this pc has the same model (" << local.paths[0] << "), taking " << missing.items.size() << " tensors from it instead of the network" << std::endl;
                    Json still;
                    still.kind = JSON_LIST;
                    std::vector<char> buffer(8 << 20);
                    for(const Json& item : missing.items){
                        const TensorInfo* t = findTensor(local, item.text);
                        bool copied = t && t->bytes == wanted[item.text];
                        std::string target = cacheFile(state, item.text);
                        if(copied){
                            std::ifstream in(local.paths[t->file], std::ios::binary);
                            std::ofstream out(target + ".part", std::ios::binary | std::ios::trunc);
                            in.seekg((std::streamoff)t->offset);
                            uint64_t left = t->bytes;
                            while(copied && left > 0){
                                size_t step = (size_t)std::min<uint64_t>(left, buffer.size());
                                copied = (bool)in.read(buffer.data(), step) && (bool)out.write(buffer.data(), step);
                                left -= copied ? step : 0;
                            }
                            out.close();
                            std::error_code moved;
                            if(copied) fs::rename(target + ".part", target, moved);
                            copied = copied && !moved;
                        }
                        if(!copied){
                            still.items.push_back(item);
                            continue;
                        }
                        have += t->bytes;
                    }
                    missing = still;
                }
            }

            state.storedBytes = have;
            std::cout << "plan received: layers " << state.firstLayer << ".." << state.lastLayer << ", " << missing.items.size() << " tensors to download, " << (double)have / (1 << 30) << " GB already cached" << std::endl;
            std::string reply = writeJson(missing);
            sendPacket(master, PACKET_PLAN_ACK, reply.data(), reply.size());
            continue;
        }

        if(head.kind == PACKET_WEIGHTS_CHUNK){
            if(body.size() < 20) break;
            uint32_t nameLength = 0;
            uint64_t offset = 0, total = 0;
            memcpy(&nameLength, body.data(), 4);
            if(body.size() < 20ULL + nameLength) break;
            std::string name(body.data() + 4, nameLength);
            if(!cacheName(name)) break;
            memcpy(&offset, body.data() + 4 + nameLength, 8);
            memcpy(&total, body.data() + 12 + nameLength, 8);
            const char* data = body.data() + 20 + nameLength;
            uint64_t size = body.size() - 20 - nameLength;

            if(name != writingName){
                if(writing.is_open()) writing.close();
                writingName = name;
                writing.open(cacheFile(state, name) + ".part", std::ios::binary | std::ios::trunc);
            }
            writing.write(data, (std::streamsize)size);
            state.storedBytes += size;

            if(offset + size >= total){
                writing.close();
                writingName.clear();
                std::error_code code;
                fs::rename(cacheFile(state, name) + ".part", cacheFile(state, name), code);
                if(code) fs::remove(cacheFile(state, name) + ".part", code);
            }
            continue;
        }

        if(head.kind == PACKET_WEIGHTS_DONE){
            std::cout << "weights stored: " << (double)state.storedBytes / (1 << 30) << " GB in " << (fs::path(state.cacheRoot) / state.modelId).string() << std::endl;
            sendPacket(master, PACKET_WEIGHTS_DONE, nullptr, 0);
            std::thread(loadFromCache, std::ref(state), state.planText).detach();
            continue;
        }

        if(head.kind == PACKET_FORWARD){
            std::string why = "layer compute is not built into this worker yet";
            sendPacket(master, PACKET_REFUSED, why.data(), why.size());
            continue;
        }

        if(head.kind == PACKET_BYE) break;
    }

    if(writing.is_open()) writing.close();
    {
        std::lock_guard<std::mutex> guard(state.lock);
        state.wire = INVALID_SOCKET;
        state.master.clear();
        if(state.computeWire != INVALID_SOCKET) shutdown(state.computeWire, SD_BOTH);
    }
    closesocket(master);
    for(int i = 0; i < 100 && state.computing; i++) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    state.store.generation++;
    releaseAll(state.store);
    dropModel(state);
    if(trusted) std::cout << "pool at " << address << " disconnected, its layers left the GPU and RAM of this PC" << std::endl;
}

int main(int argc, char* argv[]) {
    keepFullSpeed();
    std::string missing = missingCpuFeatures();
    if(!missing.empty()){
        printf("this processor lacks%s, which the engine needs (any Intel since Haswell or AMD since Ryzen has them)\n", missing.c_str());
        return 1;
    }
    WorkerState state;
    state.cacheRoot = "cache";
    bool openPage = true;
    std::vector<std::string> plain;

    for(int i = 1; i < argc; i++){
        std::string arg = argv[i];
        if(arg == "--page" && i + 1 < argc){
            state.pagePort = std::stoi(argv[++i]);
            continue;
        }
        if(arg == "--no-browser"){
            openPage = false;
            continue;
        }
        plain.push_back(arg);
    }

    if(plain.size() > 0) state.tcpPort = std::stoi(plain[0]);
    if(plain.size() > 1) state.cacheRoot = plain[1];

    fs::create_directories(state.cacheRoot);
    state.cacheRoot = fs::absolute(state.cacheRoot).string();
    state.specs = readNode(state.cacheRoot);
    state.latest = state.specs;
    std::thread([&state](){
        while(true){
            std::this_thread::sleep_for(std::chrono::seconds(1));
            NodeStats fresh = readNode(state.cacheRoot);
            PROCESS_MEMORY_COUNTERS memory = {};
            if(GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory))) fresh.host.heldRam = memory.WorkingSetSize;
            {
                std::lock_guard<std::mutex> guard(state.modelLock);
                fresh.host.heldGpu = state.model ? state.model->vramHeld : 0;
            }
            std::lock_guard<std::mutex> guard(state.latestLock);
            state.latest = fresh;
        }
    }).detach();

    std::stringstream text;
    {
        std::ifstream file(state.memberFile);
        text << file.rdbuf();
    }
    Json saved = parseJson(text.str());
    state.poolKey = saved["key"].str();
    state.host = saved["host"].str();
    state.doorPort = (int)saved["doorPort"].num(9097);
    if(!state.poolKey.empty()) state.status = "joined";

    std::cout << "Sneo worker " << state.specs.host.nodeName << " on port " << state.tcpPort << std::endl;
    std::cout << "gpu: " << state.specs.gpu.name << " | vram free " << (double)state.specs.gpu.vramFree / (1 << 30) << " GB | ram free " << (double)state.specs.host.ramFree / (1 << 30) << " GB" << std::endl;
    std::cout << "cache: " << state.cacheRoot << std::endl;
    std::cout << (state.poolKey.empty() ? "not in a pool yet" : "member of the pool at " + state.host + ", waiting for it to connect") << std::endl;

    // un worker que solo escucha en esta pc (para probar el pool con el master al lado) no necesita el firewall
    char hostText[64] = {};
    std::string listenHost = GetEnvironmentVariableA("SNEO_WORKER_HOST", hostText, sizeof(hostText)) > 0 ? hostText : "0.0.0.0";
    bool thisPcOnly = listenHost.rfind("127.", 0) == 0;
#ifdef _WIN32
    bool ruleMissing = !thisPcOnly && system("netsh advfirewall firewall show rule name=\"Sneo worker\" >nul 2>&1") != 0;
    bool publicNetwork = !thisPcOnly && system("powershell -NoProfile -Command \"if(Get-NetConnectionProfile | Where-Object { $_.NetworkCategory -eq 'Public' }){ exit 1 }\" >nul 2>&1") != 0;
    fs::path setup = fs::path(runtimeFolder()).parent_path() / "tools" / "pool_setup.ps1";
    if((ruleMissing || publicNetwork) && fs::exists(setup)){
        std::cout << (ruleMissing ? "the firewall does not let the pool reach this PC yet" : "this network is marked public, so the pool port stays closed") << ", Windows will ask for permission to fix it" << std::endl;
        std::string args = "-NoProfile -ExecutionPolicy Bypass -File \"" + setup.string() + "\"";
        ShellExecuteA(nullptr, "runas", "powershell.exe", args.c_str(), nullptr, SW_SHOWNORMAL);
    }
#else
    // en linux el firewall lo maneja cada distro: solo se avisa del puerto que tiene que estar abierto
    if(!thisPcOnly) std::cout << "if this PC has a firewall, let tcp port " << state.tcpPort << " in so the pool can reach it" << std::endl;
#endif

    SOCKET server = listenTcp(state.tcpPort, listenHost);
    if(server == INVALID_SOCKET){
        std::cout << "cannot listen on port " << state.tcpPort << std::endl;
        return 1;
    }

    std::string url = "http://127.0.0.1:" + std::to_string(state.pagePort) + "/";
    SOCKET site = listenTcp(state.pagePort, "127.0.0.1");
    if(site == INVALID_SOCKET) std::cout << "cannot open the worker page on " << url << ", another worker.exe may be using it" << std::endl;

    if(site != INVALID_SOCKET){
        std::thread([site, &state](){
            while(true){
                SOCKET client = accept(site, nullptr, nullptr);
                if(client == INVALID_SOCKET) continue;
                std::thread(servePage, client, std::ref(state)).detach();
            }
        }).detach();
        std::cout << "open " << url << " to ask a pool to let this machine in" << std::endl;
#ifdef _WIN32
        if(openPage) ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
        if(openPage && system(("xdg-open '" + url + "' >/dev/null 2>&1 &").c_str()) != 0) std::cout << "open the page by hand, xdg-open is missing" << std::endl;
#endif
    }

    while(true){
        sockaddr_in from = {};
        socklen_t length = sizeof(from);
        SOCKET master = accept(server, (sockaddr*)&from, &length);
        if(master == INVALID_SOCKET) continue;

        char address[64] = {};
        inet_ntop(AF_INET, &from.sin_addr, address, sizeof(address));

        std::thread([master, &state, ip = std::string(address)](){
            DWORD timeout = 10000;
            socketWait(master, SO_RCVTIMEO, timeout);
            PacketHead head;
            std::vector<char> body;
            if(!readPacket(master, head, body, sizeof(HelloBody)) || head.kind != PACKET_HELLO || body.size() < sizeof(HelloBody)){
                closesocket(master);
                return;
            }
            HelloBody hello;
            memcpy(&hello, body.data(), sizeof(hello));
            if(hello.pad == 1){
                serveCompute(master, state, ip, hello);
                return;
            }
            if(state.busy.exchange(true)){
                std::string why = "this node already serves a master";
                sendPacket(master, PACKET_REFUSED, why.data(), why.size());
                closesocket(master);
                return;
            }
            serveMaster(master, state, ip, hello);
            state.busy = false;
        }).detach();
    }
}

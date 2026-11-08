// tests/core/native_dense_nextn_test.cpp - a qwen4exp file that carries its NextN (MTP) layer in-file (block_count
// 49, nextn_predict_layers 1: ThinkingCap's layout) serves the 48 trunk layers' projections and never block 48's.
// The pack has no rows for block 48 (iq_pack leaves them out), so a served or uploaded blk.48 tensor would fail the
// load on "tensor absent from canonical table".
//
//   names (no device): served_names keeps blocks 0 and 47 and leaves out exactly the declared NextN blocks; an
//          undeclared block past them still counts; an invalid count is refused; a later shard that copies the
//          architecture does not reset the tail; a file without the key is unchanged.
//   load (default; needs a CUDA device, exits 77 without one): the same fixture loads with and without stage bounds,
//          the global layer range still applies, an undeclared blk.49 and a truncated blk.48 payload are still
//          refused, and weight_bytes_for prices exactly the matrices load allocates.
// Synthetic two-shard files at small dimensions (Q8_0 {256, 4}: 1088 bytes a matrix).
#include "gguf_fixture.hpp"

#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::NativeDense;
using strata::core::WeightTable;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-nextn-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

const std::string NEXTN = "qwen4exp.nextn_predict_layers";
const std::string Q0 = "blk.0.attn_q.weight", Q47 = "blk.47.attn_q.weight", Q48 = "blk.48.attn_q.weight",
                  K48 = "blk.48.attn_k.weight", EH48 = "blk.48.nextn.eh_proj.weight", Q49 = "blk.49.attn_q.weight",
                  Q50 = "blk.50.attn_q.weight";
constexpr uint64_t NE0 = 256, NE1 = 4, MATRIX = NE0 / 32 * 34 * NE1;   // a Q8_0 {256, 4}: 1088 bytes
static_assert(MATRIX == 1088);

// The six keys the engine guard reads, with `blocks` blocks; `nextn` adds the count (nullptr: no key).
std::vector<fixture::Kv> arch_keys(uint64_t blocks, const fixture::Kv* nextn) {
    std::vector<fixture::Kv> kv{fixture::str("general.architecture", "qwen4exp"),
                                fixture::u32("qwen4exp.block_count", blocks),
                                fixture::u32("qwen4exp.embedding_length", 2560),
                                fixture::u32("qwen4exp.expert_count", 512),
                                fixture::u32("qwen4exp.expert_used_count", 10),
                                fixture::u32("qwen4exp.attention.head_count", 24),
                                fixture::u32("qwen4exp.attention.head_count_kv", 2)};
    if (nextn) kv.push_back(*nextn);
    return kv;
}

fixture::Tensor q8(const std::string& name, uint8_t seed) { return {name, {NE0, NE1}, 8, seed}; }

struct Model {
    std::vector<std::string> shards;
};

enum class Second { SplitKeysOnly, CopiesArchitecture };

// Shard 1: the metadata (with `kv`) and `first`; shard 2: only the split keys (ThinkingCap's shard 2), or the split
// keys plus a copied general.architecture, and `second`.
Model write_model(const fs::path& dir, const std::string& name, std::vector<fixture::Kv> kv,
                  const std::vector<fixture::Tensor>& first, const std::vector<fixture::Tensor>& second,
                  Second shape = Second::SplitKeysOnly, uint64_t cut = 0) {
    const uint64_t tensors = first.size() + second.size();
    for (const auto& k : fixture::split_keys(0, 2, tensors)) kv.push_back(k);
    std::vector<fixture::Kv> kv2 = fixture::split_keys(1, 2, tensors);
    if (shape == Second::CopiesArchitecture) kv2.insert(kv2.begin(), fixture::str("general.architecture", "qwen4exp"));
    Model m;
    m.shards = {(dir / (name + "-00001-of-00002.gguf")).string(), (dir / (name + "-00002-of-00002.gguf")).string()};
    fixture::write(m.shards[0], kv, first, cut);
    fixture::write(m.shards[1], kv2, second);
    return m;
}

// ThinkingCap's layout at fixture size: block 48 is the NextN layer, split across both shards.
Model thinkingcap(const fs::path& dir, const std::string& name, const fixture::Kv& nextn,
                  Second shape = Second::SplitKeysOnly) {
    return write_model(dir, name, arch_keys(49, &nextn), {q8(Q0, 1), q8(Q48, 2), q8(EH48, 3)},
                       {q8(Q47, 4), q8(K48, 5)}, shape);
}

// iq_pack's shape-only rows for the trunk projections the GGUF serves (the PLE test's row format); no blk.48 rows.
// The header pool is nonzero, as in the packs iq_pack writes.
void write_pack(const fs::path& dir, const std::vector<std::string>& names) {
    fs::create_directories(dir);
    std::string idx = "# align 256 pool 256 tensors " + std::to_string(names.size()) + "\n";
    for (const auto& n : names) idx += n + " 0 0 0 0 0 0 256 4 8 0 32 0 0 0 0 0 0 0\n";
    std::ofstream(dir / "index.txt", std::ios::binary) << idx;
    std::vector<char> bytes(16);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (char) fixture::pattern(9, i);
    std::ofstream(dir / "dense.bin", std::ios::binary).write(bytes.data(), (std::streamsize) bytes.size());
}

std::string names_of(const std::set<std::string>& s) {
    std::string r;
    for (const auto& n : s) r += (r.empty() ? "" : " ") + n;
    return r;
}

// served_names alone: no device, no arena.
int run_names(const fs::path& tmp) {
    std::printf("served_names (no device)\n");
    {
        const Model m = thinkingcap(tmp, "tc", fixture::u32(NEXTN, 1));
        std::set<std::string> out;
        std::string err;
        const bool ok = NativeDense::served_names(m.shards, true, out, err);
        check(ok && out == std::set<std::string>{Q0, Q47},
              "49 blocks, nextn 1: blocks 0 and 47 served, none of block 48 (" + names_of(out) + err + ")");
    }
    {
        const Model m = write_model(tmp, "past", arch_keys(49, nullptr), {q8(Q0, 1), q8(Q48, 2), q8(Q49, 3)},
                                    {q8(Q50, 4)});
        std::vector<fixture::Kv> kv = arch_keys(49, nullptr);
        kv.push_back(fixture::u32(NEXTN, 1));
        const Model n = write_model(tmp, "past1", kv, {q8(Q0, 1), q8(Q48, 2), q8(Q49, 3)}, {q8(Q50, 4)});
        std::set<std::string> out;
        std::string err;
        const bool ok = NativeDense::served_names(n.shards, true, out, err);
        check(ok && out == std::set<std::string>{Q0, Q49, Q50},
              "  undeclared blocks 49 and 50 are still served (" + names_of(out) + err + ")");
        out.clear();
        const bool before = NativeDense::served_names(m.shards, true, out, err);
        check(before && out == std::set<std::string>{Q0, Q48, Q49, Q50},
              "  the same file without the key serves block 48 too, as before (" + names_of(out) + err + ")");
    }
    {
        std::vector<fixture::Kv> kv = arch_keys(50, nullptr);
        kv.push_back(fixture::u32(NEXTN, 2));
        const Model m = write_model(tmp, "two", kv, {q8(Q47, 1), q8(Q48, 2), q8(Q49, 3)}, {q8(Q50, 4)});
        std::set<std::string> out;
        std::string err;
        const bool ok = NativeDense::served_names(m.shards, true, out, err);
        check(ok && out == std::set<std::string>{Q47, Q50},
              "50 blocks, nextn 2: 48 and 49 left out (" + names_of(out) + err + ")");
    }
    {
        const Model m = thinkingcap(tmp, "zero", fixture::u32(NEXTN, 0));
        std::set<std::string> out;
        std::string err;
        const bool ok = NativeDense::served_names(m.shards, true, out, err);
        check(ok && out == std::set<std::string>{Q0, Q47, Q48, K48},
              "nextn 0 is no tail: block 48 served, as without the key " + err);
    }
    {
        const Model m = thinkingcap(tmp, "bool", fixture::boolean(NEXTN, true));
        std::set<std::string> out;
        std::string err;
        const bool ok = NativeDense::served_names(m.shards, true, out, err);
        check(!ok && err == "native dense: qwen4exp.nextn_predict_layers is not an integer",
              "a bool count is refused: " + err);
    }
    {
        const Model m = thinkingcap(tmp, "copy", fixture::u32(NEXTN, 1), Second::CopiesArchitecture);
        std::set<std::string> out;
        std::string err;
        const bool ok = NativeDense::served_names(m.shards, true, out, err);
        check(ok && out == std::set<std::string>{Q0, Q47},
              "shard 2 copying general.architecture keeps the tail (" + names_of(out) + err + ")");
    }
    return 0;
}

struct Loaded {
    bool ok = false;
    std::string err;
    std::set<std::string> skip;
    WeightTable wt;
    NativeDense dense;
    void* arena = nullptr;
    ~Loaded() { if (arena) cudaFree(arena); }
};

// generate.cpp's order: served_names, pool_bytes, the arena, WeightTable::load (the table), then NativeDense::load
// when `dense` is set.  A setup failure leaves ok false with its error.
bool setup(const fs::path& pack, const Model& m, Loaded& l) {
    if (!NativeDense::served_names(m.shards, true, l.skip, l.err)) return false;
    if (!NativeDense::keep_unquantized_ple_key(pack.string(), l.skip, l.err)) return false;
    uint64_t pool = 0;
    if (!WeightTable::pool_bytes(pack.string(), pool, l.err, &l.skip)) return false;
    if (cudaMalloc(&l.arena, pool ? pool : 256) != cudaSuccess) { l.err = "cudaMalloc"; return false; }
    return l.wt.load(pack.string(), l.arena, pool ? pool : 256, l.err, &l.skip);
}
void load(const fs::path& pack, const Model& m, Loaded& l, int64_t lo = 0, int64_t hi = -1) {
    if (!setup(pack, m, l)) { l.err = "setup: " + l.err; return; }
    l.ok = l.dense.load(m.shards, l.wt, l.err, true, lo, hi);
}

bool attached(const Loaded& l, const std::string& name) {
    const auto* w = l.wt.find(name);
    return w && w->native_data && w->native_type == 8;
}

int run_load(const fs::path& tmp) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("native_dense_nextn_test: no CUDA device, skipped\n");
        return 77;
    }
    NativeDense::set_layer_range(-1, -1);
    const fs::path pack = tmp / "pack";
    write_pack(pack, {Q0, Q47});
    const Model tc = thinkingcap(tmp, "tc", fixture::u32(NEXTN, 1));

    std::printf("load: 49 blocks, nextn 1, a pack without block 48\n");
    uint64_t full_alloc = 0, half_alloc = 0;
    struct Bound { int64_t lo, hi; const char* what; };
    for (const Bound b : {Bound{0, -1, "no stage bound"}, Bound{0, 48, "[0, 48)"}, Bound{0, 49, "[0, 49)"}}) {
        auto l = std::make_unique<Loaded>();
        load(pack, tc, *l, b.lo, b.hi);
        check(l->ok && l->dense.tensor_count() == 2 && attached(*l, Q0) && attached(*l, Q47) && !l->wt.find(Q48) &&
                  !l->wt.find(K48),
              std::string(b.what) + ": loads, blocks 0 and 47 attached, no block 48 row " + l->err);
        if (b.hi == -1) full_alloc = l->dense.allocated_bytes();
        check(l->dense.allocated_bytes() == 2 * MATRIX, "  allocated " + std::to_string(l->dense.allocated_bytes()));
    }
    {
        auto l = std::make_unique<Loaded>();
        load(pack, tc, *l, 0, 24);
        check(l->ok && l->dense.tensor_count() == 1 && attached(*l, Q0) && !attached(*l, Q47),
              "[0, 24): block 0 attached, block 47 not " + l->err);
        half_alloc = l->dense.allocated_bytes();
        check(half_alloc == MATRIX, "  allocated " + std::to_string(half_alloc));
    }
    {
        NativeDense::set_layer_range(0, 24);   // the layer split's process-wide range (generate.cpp)
        auto l = std::make_unique<Loaded>();
        load(pack, tc, *l);
        NativeDense::set_layer_range(-1, -1);
        check(l->ok && l->dense.tensor_count() == 1 && attached(*l, Q0) && !attached(*l, Q47),
              "set_layer_range(0, 24), no explicit bound: block 0 attached, block 47 not " + l->err);
    }
    {
        const Model copy = thinkingcap(tmp, "copy", fixture::u32(NEXTN, 1), Second::CopiesArchitecture);
        auto l = std::make_unique<Loaded>();
        load(pack, copy, *l);
        check(l->ok && l->dense.tensor_count() == 2 && attached(*l, Q0) && attached(*l, Q47),
              "shard 2 copying general.architecture: loads, the tail kept (CUDA/HIP/Metal path) " + l->err);
    }

    std::printf("load: refusals that still apply\n");
    {
        std::vector<fixture::Kv> kv = arch_keys(49, nullptr);
        kv.push_back(fixture::u32(NEXTN, 1));
        const Model past = write_model(tmp, "past", kv, {q8(Q0, 1), q8(Q48, 2), q8(Q49, 3)}, {q8(Q47, 4)});
        for (const Bound b : {Bound{0, -1, "no stage bound"}, Bound{0, 50, "[0, 50)"}}) {
            auto l = std::make_unique<Loaded>();
            load(pack, past, *l, b.lo, b.hi);
            check(!l->ok && l->err == "native dense: tensor absent from canonical table: " + Q49,
                  std::string("an undeclared blk.49, ") + b.what + ": " + l->err);
        }
    }
    {
        // block 48's attn_q is the file's last payload; 16 bytes short
        std::vector<fixture::Kv> kv = arch_keys(49, nullptr);
        kv.push_back(fixture::u32(NEXTN, 1));
        const Model cut = write_model(tmp, "cut", kv, {q8(Q0, 1), q8(EH48, 3), q8(Q48, 2)}, {q8(Q47, 4)},
                                      Second::SplitKeysOnly, 16);
        auto l = std::make_unique<Loaded>();
        load(pack, cut, *l);
        check(!l->ok && l->err == "native dense: truncated payload " + Q48,
              "a truncated NextN payload is still refused: " + l->err);
    }

    std::printf("weight_bytes_for: what load allocates, without the device\n");
    {
        auto l = std::make_unique<Loaded>();
        const bool ready = setup(pack, tc, *l);
        check(ready, "the table loads " + l->err);
        uint64_t b48 = 0, b49 = 0, b24 = 0;
        std::string err;
        const bool ok = ready && NativeDense::weight_bytes_for(tc.shards, l->wt, true, 0, 48, b48, err) &&
                        NativeDense::weight_bytes_for(tc.shards, l->wt, true, 0, 49, b49, err) &&
                        NativeDense::weight_bytes_for(tc.shards, l->wt, true, 0, 24, b24, err);
        check(ok && b48 == 2 * MATRIX && b49 == b48 && b24 == MATRIX,
              "[0, 48) " + std::to_string(b48) + ", [0, 49) " + std::to_string(b49) + ", [0, 24) " +
                  std::to_string(b24) + " " + err);
        check(ok && b48 == full_alloc && b24 == half_alloc, "  equal to the loads' allocated_bytes");
        const Model bad = thinkingcap(tmp, "bool", fixture::boolean(NEXTN, true));
        uint64_t out = 1;
        err.clear();
        const bool refused = ready && !NativeDense::weight_bytes_for(bad.shards, l->wt, true, 0, 49, out, err);
        check(refused && err == "native dense: qwen4exp.nextn_predict_layers is not an integer",
              "a bool count is refused: " + err);
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    const bool names = argc > 1 && std::string(argv[1]) == "names";
    if (argc > 2 || (argc == 2 && !names)) {
        std::fprintf(stderr, "usage: native_dense_nextn_test [names]\n");
        return 2;
    }
    int rc = 0;
    try {
        TempDir tmp;
        rc = names ? run_names(tmp.path) : run_load(tmp.path);
    } catch (const std::exception& error) {
        std::printf("native_dense_nextn_test: setup failed: %s\n", error.what());
        return 1;
    }
    if (rc == 77) return 77;
    std::printf(g_fail ? "native_dense_nextn_test: %d FAILED\n" : "native_dense_nextn_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}

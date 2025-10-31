/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "tt.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "memory.h"
#include "misc.h"
#include "syzygy/tbprobe.h"
#include "thread.h"

namespace Stockfish {

template<std::size_t EntrySize, unsigned int LocalOffset>
struct TTExtraEntry {
    static constexpr std::size_t  kEntrySize   = EntrySize;  // in bits
    static constexpr unsigned int kLocalOffset = LocalOffset;

    struct Accessor;
    Accessor operator()() const;
};

struct TTEntry {

    // Convert internal bitfields to external types
    TTData read() const;

    // An "extra" entry is a small data field up to 5 bits that is stored
    // in the padding field of Cluster (which is 16 bits).
    static constexpr int      kExtraEntrySize = 1;
    static constexpr unsigned kExtraEntryMask = (1u << kExtraEntrySize) - 1;

    bool is_occupied() const;
    void save(
      Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, bool cut, uint8_t generation8);
    // The returned age is a multiple of TranspositionTable::GENERATION_DELTA
    uint8_t relative_age(const uint8_t generation8) const;

   private:
    friend class TranspositionTable;

    struct ExtraEntryAccessor {
        constexpr ExtraEntryAccessor(Cluster* c, int i) :
            cluster_(c),
            index_(i) {}

        operator int() const;

        void operator=(int val) const;

       private:
        Cluster* cluster_;
        int      index_;
    };

    uint16_t key16;
    uint8_t  depth8;
    uint8_t  genBound8;
    Move     move16;
    int16_t  value16;
    int16_t  eval16;

    std::pair<Cluster*, int> locate_in_cluster() const;

    // Below fields are "extra" entries, stored outside TTEntry and in the
    // padding of Cluster.

    // In C++20, all member function definitions can be removed by simply using
    // [[no_unique_address]] instead.

    TTExtraEntry<1, 0>::Accessor cutNode1() const;
};

// Cluster is a collection of TTEntry objects and designed to fit within
// a cache line, which is typically 64 bytes. The size of Cluster is
// explicitly defined outside the scope of the struct to ensure such
// property.
constexpr std::size_t kClusterSize = 0x20;

struct Cluster {
    static constexpr std::size_t kNumEntries = kClusterSize / sizeof(TTEntry);

    TTEntry       entry[kNumEntries];
    std::uint16_t extra;

    static constexpr std::size_t kExtraBitsPerEntry = (sizeof(extra) * 8) / kNumEntries;
} __attribute__((packed));

static_assert(sizeof(Cluster) == kClusterSize, "Wrong Cluster size");

// Because there are three entries per Cluster, it is possible to use ptr >> 3
// as an index even though the size of TTEntry is 10 bytes.
std::pair<Cluster*, int> TTEntry::locate_in_cluster() const {
    static_assert(Cluster::kNumEntries <= 4,
                  "Shift optimization is not valid for more than 4 entries.");

    Cluster* const cluster =
      reinterpret_cast<Cluster*>(reinterpret_cast<std::uintptr_t>(this) & ~(kClusterSize - 1));
    const int index = (reinterpret_cast<std::uintptr_t>(this) & (kClusterSize - 1)) >> 3;

    return {cluster, index};
}

template<std::size_t EntrySize, unsigned int LocalOffset>
struct TTExtraEntry<EntrySize, LocalOffset>::Accessor {
    static_assert(EntrySize > 0);
    static_assert(EntrySize * Cluster::kNumEntries <= sizeof(Cluster::extra) * 8);
    static_assert(EntrySize + LocalOffset < (sizeof(Cluster::extra) * 8) / Cluster::kNumEntries);

    constexpr Accessor(Cluster* c, int i) :
        cluster_(c),
        index_(i) {}

    constexpr operator int() const { return cluster_->extra >> offset() & mask(); }

    constexpr void operator=(int val) const {
        cluster_->extra = cluster_->extra & ~(mask() << offset())
                        | (static_cast<decltype(Cluster::extra)>(val) << offset());
    }

   private:
    constexpr int      offset() const { return LocalOffset + index_ * Cluster::kExtraBitsPerEntry; }
    constexpr unsigned mask() const { return (1u << kEntrySize) - 1; }

    Cluster* cluster_;
    int      index_;
};

TTExtraEntry<1, 0>::Accessor TTEntry::cutNode1() const {
    auto [cluster, index] = locate_in_cluster();
    return TTExtraEntry<1, 0>::Accessor(cluster, index);
}

TTData TTEntry::read() const {
    return TTData{Move(move16),           Value(value16),
                  Value(eval16),          Depth(depth8 + DEPTH_ENTRY_OFFSET),
                  Bound(genBound8 & 0x3), bool(genBound8 & 0x4),
                  bool(cutNode1())};
}

// `genBound8` is where most of the details are. We use the following constants to manipulate 5 leading generation bits
// and 3 trailing miscellaneous bits.

// These bits are reserved for other things.
static constexpr unsigned GENERATION_BITS = 3;
// increment for generation field
static constexpr int GENERATION_DELTA = (1 << GENERATION_BITS);
// cycle length
static constexpr int GENERATION_CYCLE = 255 + GENERATION_DELTA;
// mask to pull out generation number
static constexpr int GENERATION_MASK = (0xFF << GENERATION_BITS) & 0xFF;

// DEPTH_ENTRY_OFFSET exists because 1) we use `bool(depth8)` as the occupancy check, but
// 2) we need to store negative depths for QS. (`depth8` is the only field with "spare bits":
// we sacrifice the ability to store depths greater than 1<<8 less the offset, as asserted in `save`.)
bool TTEntry::is_occupied() const { return bool(depth8); }

// Populates the TTEntry with a new node's data, possibly
// overwriting an old position. The update is not atomic and can be racy.
void TTEntry::save(
  Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, bool cut, uint8_t generation8) {

    // Preserve the old ttmove if we don't have a new one
    if (m || uint16_t(k) != key16)
        move16 = m;

    // Overwrite less valuable entries (cheapest checks first)
    if (b == BOUND_EXACT || uint16_t(k) != key16 || d - DEPTH_ENTRY_OFFSET + 2 * pv > depth8 - 4
        || relative_age(generation8))
    {
        assert(d > DEPTH_ENTRY_OFFSET);
        assert(d < 256 + DEPTH_ENTRY_OFFSET);

        key16      = uint16_t(k);
        depth8     = uint8_t(d - DEPTH_ENTRY_OFFSET);
        genBound8  = uint8_t(generation8 | uint8_t(pv) << 2 | b);
        value16    = int16_t(v);
        eval16     = int16_t(ev);
        cutNode1() = cut;
    }
    else if (depth8 + DEPTH_ENTRY_OFFSET >= 5 && Bound(genBound8 & 0x3) != BOUND_EXACT)
        depth8--;
}


uint8_t TTEntry::relative_age(const uint8_t generation8) const {
    // Due to our packed storage format for generation and its cyclic
    // nature we add GENERATION_CYCLE (256 is the modulus, plus what
    // is needed to keep the unrelated lowest n bits from affecting
    // the result) to calculate the entry age correctly even after
    // generation8 overflows into the next cycle.
    return (GENERATION_CYCLE + generation8 - genBound8) & GENERATION_MASK;
}


// TTWriter is but a very thin wrapper around the pointer
TTWriter::TTWriter(TTEntry* tte) :
    entry(tte) {}

void TTWriter::write(
  Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, bool cut, uint8_t generation8) {
    entry->save(k, v, pv, b, d, m, ev, cut, generation8);
}


// Sets the size of the transposition table,
// measured in megabytes. Transposition table consists
// of clusters and each cluster consists of ClusterSize number of TTEntry.
void TranspositionTable::resize(size_t mbSize, ThreadPool& threads) {
    aligned_large_pages_free(table);

    clusterCount = mbSize * 1024 * 1024 / sizeof(Cluster);

    table = static_cast<Cluster*>(aligned_large_pages_alloc(clusterCount * sizeof(Cluster)));

    if (!table)
    {
        std::cerr << "Failed to allocate " << mbSize << "MB for transposition table." << std::endl;
        exit(EXIT_FAILURE);
    }

    clear(threads);
}


// Initializes the entire transposition table to zero,
// in a multi-threaded way.
void TranspositionTable::clear(ThreadPool& threads) {
    generation8              = 0;
    const size_t threadCount = threads.num_threads();

    for (size_t i = 0; i < threadCount; ++i)
    {
        threads.run_on_thread(i, [this, i, threadCount]() {
            // Each thread will zero its part of the hash table
            const size_t stride = clusterCount / threadCount;
            const size_t start  = stride * i;
            const size_t len    = i + 1 != threadCount ? stride : clusterCount - start;

            std::memset(&table[start], 0, len * sizeof(Cluster));
        });
    }

    for (size_t i = 0; i < threadCount; ++i)
        threads.wait_on_thread(i);
}


// Returns an approximation of the hashtable
// occupation during a search. The hash is x permill full, as per UCI protocol.
// Only counts entries which match the current generation.
int TranspositionTable::hashfull(int maxAge) const {
    int maxAgeInternal = maxAge << GENERATION_BITS;
    int cnt            = 0;
    for (int i = 0; i < 1000; ++i)
        for (int j = 0; j < int(Cluster::kNumEntries); ++j)
            cnt += table[i].entry[j].is_occupied()
                && table[i].entry[j].relative_age(generation8) <= maxAgeInternal;

    return cnt / Cluster::kNumEntries;
}


void TranspositionTable::new_search() {
    // increment by delta to keep lower bits as is
    generation8 += GENERATION_DELTA;
}


uint8_t TranspositionTable::generation() const { return generation8; }


// Looks up the current position in the transposition
// table. It returns true if the position is found.
// Otherwise, it returns false and a pointer to an empty or least valuable TTEntry
// to be replaced later. The replace value of an entry is calculated as its depth
// minus 8 times its relative age. TTEntry t1 is considered more valuable than
// TTEntry t2 if its replace value is greater than that of t2.
std::tuple<bool, TTData, TTWriter> TranspositionTable::probe(const Key key) const {

    TTEntry* const tte   = first_entry(key);
    const uint16_t key16 = uint16_t(key);  // Use the low 16 bits as key inside the cluster

    for (int i = 0; i < int(Cluster::kNumEntries); ++i)
        if (tte[i].key16 == key16)
            // This gap is the main place for read races.
            // After `read()` completes that copy is final, but may be self-inconsistent.
            return {tte[i].is_occupied(), tte[i].read(), TTWriter(&tte[i])};

    // Find an entry to be replaced according to the replacement strategy
    TTEntry* replace = tte;
    for (int i = 1; i < int(Cluster::kNumEntries); ++i)
        if (replace->depth8 - replace->relative_age(generation8)
            > tte[i].depth8 - tte[i].relative_age(generation8))
            replace = &tte[i];

    return {
      false,
      TTData{Move::none(), VALUE_NONE, VALUE_NONE, DEPTH_ENTRY_OFFSET, BOUND_NONE, false, false},
      TTWriter(replace)};
}


TTEntry* TranspositionTable::first_entry(const Key key) const {
    return &table[mul_hi64(key, clusterCount)].entry[0];
}

}  // namespace Stockfish

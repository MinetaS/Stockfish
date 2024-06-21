/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

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


// TTEntry struct is the 10 bytes transposition table entry, defined as below:
//
// key        16 bit
// depth       8 bit
// generation  5 bit
// pv node     1 bit
// bound type  2 bit
// move       16 bit
// value      16 bit
// evaluation 16 bit
//
// These fields are in the same order as accessed by TT::probe(), since memory is fastest sequentially.
// Equally, the store order in save() matches this order.

struct TTEntry {

    // Convert internal bitfields to external types
    TTData read() const {
        return TTData{Move(move16),           Value(value16),
                      Value(eval16),          Depth(depth8 + DEPTH_ENTRY_OFFSET),
                      Bound(genBound8 & 0x3), bool(genBound8 & 0x4)};
    }

    bool    is_occupied() const;
    void    save(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8);
    uint8_t relative_age(const uint8_t generation8) const;

   private:
    friend class TranspositionTable;

    Cluster* cluster() const {
        return reinterpret_cast<Cluster*>(reinterpret_cast<uintptr_t>(this) & ~uintptr_t(0x1F));
    }

    inline int cluster_index() const { return (reinterpret_cast<uintptr_t>(this) >> 3) & 0x3; }

    uint16_t extra() const;

    uint16_t key16;
    uint8_t  depth8;
    uint8_t  genBound8;
    Move     move16;
    int16_t  value16;
    int16_t  eval16;
};
static_assert(sizeof(TTEntry) == 10, "TTEntry is not 10 bytes");

static constexpr int ClusterSize = 3;

struct Cluster {
    TTEntry  entry[ClusterSize];
    uint16_t extra;

   private:
    friend class TranspositionTable;
    friend struct TTEntry;

    static constexpr uint16_t kExtraMask = 0x1F;

    constexpr uint16_t get_extra(int index) const { return extra >> (index * 5) & kExtraMask; }

    constexpr void set_extra(int index, uint16_t v) {
        extra = (extra & ~(kExtraMask << (index * 5))) | (v << (index * 5));
    }
};
static_assert(sizeof(Cluster) == 32, "Cluster is not 32 bytes");

uint16_t TTEntry::extra() const { return cluster()->get_extra(cluster_index()); }

static constexpr uint16_t get_extra_from_key(Key key) { return key >> 59; }

// `genBound8` is where most of the details are. We use the following constants to manipulate 5 leading generation bits
// and 3 trailing miscellaneous bits.

static constexpr unsigned GENERATION_BITS  = 3;
static constexpr int      GENERATION_DELTA = (1 << GENERATION_BITS);
static constexpr int      GENERATION_CYCLE = 255 + GENERATION_DELTA;
static constexpr int      GENERATION_MASK  = (0xFF << GENERATION_BITS) & 0xFF;

// DEPTH_ENTRY_OFFSET exists because 1) we use `bool(depth8)` as the occupancy check, but
// 2) we need to store negative depths for QS. (`depth8` is the only field with "spare bits":
// we sacrifice the ability to store depths greater than 1<<8 less the offset, as asserted in `save`.)
bool TTEntry::is_occupied() const { return bool(depth8); }

// Populates the TTEntry with a new node's data, possibly
// overwriting an old position. The update is not atomic and can be racy.
void TTEntry::save(
  Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8) {

    uint16_t ext    = get_extra_from_key(k);
    bool     hashOk = uint16_t(k) != key16 || ext != extra();

    // Preserve the old TT move if we don't have a new one
    if (m || hashOk)
        move16 = m;

    // Overwrite less valuable entries (cheapest checks first)
    if (hashOk || b == BOUND_EXACT || d - DEPTH_ENTRY_OFFSET + 2 * pv > depth8 - 4
        || relative_age(generation8))
    {
        assert(d > DEPTH_ENTRY_OFFSET);
        assert(d < 256 + DEPTH_ENTRY_OFFSET);

        key16     = uint16_t(k);
        depth8    = uint8_t(d - DEPTH_ENTRY_OFFSET);
        genBound8 = uint8_t(generation8 | uint8_t(pv) << 2 | b);
        value16   = int16_t(v);
        eval16    = int16_t(ev);
        cluster()->set_extra(cluster_index(), ext);
    }
}

uint8_t TTEntry::relative_age(const uint8_t generation8) const {
    // Due to our packed storage format for generation and its cyclic
    // nature we add GENERATION_CYCLE (256 is the modulus, plus what
    // is needed to keep the unrelated lowest n bits from affecting
    // the result) to calculate the entry age correctly even after
    // generation8 overflows into the next cycle.
    return (GENERATION_CYCLE + generation8 - genBound8) & GENERATION_MASK;
}


TTWriter::TTWriter(TTEntry* tte) :
    entry(tte) {}

void TTWriter::write(
  Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8) {
    entry->save(k, v, pv, b, d, m, ev, generation8);
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

// Returns an approximation of the hashtable occupation during a search. The
// hash is x permill full, as per UCI protocol. This counts entries which match
// the current generation only.
int TranspositionTable::hashfull() const {

    int cnt = 0;
    for (int i = 0; i < 1000; ++i)
        for (int j = 0; j < ClusterSize; ++j)
            cnt += table[i].entry[j].is_occupied()
                && (table[i].entry[j].genBound8 & GENERATION_MASK) == generation8;

    return cnt / ClusterSize;
}

void TranspositionTable::new_search() {
    // increment by delta to keep lower bits as is
    generation8 += GENERATION_DELTA;
}

uint8_t TranspositionTable::generation() const { return generation8; }

// Looks up the current position in the transposition table.
// If the matching entry is not found, it returns a pointer to an empty or
// least valuable TTEntry to be replaced later. The replacement strategy is
// based on depth and relative age of each entry.
TranspositionTable::ProbeResult TranspositionTable::probe(const Key key) const {

    uint16_t key16 = uint16_t(key);  // Use low 16 bits as a key inside the cluster
    Cluster* cl    = cluster(key);

    for (int i = 0; i < ClusterSize; ++i)
    {
        TTEntry* tte = &cl->entry[i];

        if (tte->key16 == key16 && tte->extra() == get_extra_from_key(key))
            return {tte->is_occupied(), tte->read(), TTWriter(tte)};
    }

    // Find an entry to be replaced.
    TTEntry* replace = &cl->entry[0];

    for (int i = 1; i < ClusterSize; ++i)
    {
        TTEntry* tte = &cl->entry[i];
        if (replace->depth8 - replace->relative_age(generation8) * 2
            > tte->depth8 - tte->relative_age(generation8) * 2)
            replace = tte;
    }

    return {false, replace->read(), TTWriter(replace)};
}


Cluster* TranspositionTable::cluster(const Key key) const {
    return &table[mul_hi64(key, clusterCount)];
}

}  // namespace Stockfish

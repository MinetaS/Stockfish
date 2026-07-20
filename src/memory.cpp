/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

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

#include "memory.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>  // std::cerr
#include <limits>

#if __has_include("features.h")
    #include <features.h>
#endif

#if defined(__linux__) && !defined(__ANDROID__)
    #include <errno.h>
    #include <sys/mman.h>
    #include <unistd.h>
    // IWYU pragma: no_include <bits/mman-map-flags-generic.h>
    #include <cstring>
    #include <mutex>
    #include <map>
#endif

#if defined(__APPLE__) || defined(__ANDROID__) || defined(__OpenBSD__) \
  || (defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE_ALIGNED_ALLOC) && !defined(_WIN32)) \
  || defined(__e2k__)
    #define POSIXALIGNEDALLOC
    #include <stdlib.h>
#endif

#ifdef _WIN32
    #if _WIN32_WINNT < 0x0601
        #undef _WIN32_WINNT
        #define _WIN32_WINNT 0x0601  // Force to include needed API prototypes
    #endif

    #ifndef NOMINMAX
        #define NOMINMAX
    #endif

    #include <ios>  // std::hex, std::dec
    #include <windows.h>

// The needed Windows API for processor groups could be missed from old Windows
// versions, so instead of calling them directly (forcing the linker to resolve
// the calls at compile time), try to load them at runtime. To do this we need
// first to define the corresponding function pointers.

#endif


namespace Stockfish {

// Wrappers for systems where the c++17 implementation does not guarantee the
// availability of aligned_alloc(). Memory allocated with std_aligned_alloc()
// must be freed with std_aligned_free().

void* std_aligned_alloc(usize alignment, usize size) {
#if defined(_ISOC11_SOURCE)
    return aligned_alloc(alignment, size);
#elif defined(POSIXALIGNEDALLOC)
    void* mem = nullptr;
    posix_memalign(&mem, alignment, size);
    return mem;
#elif defined(_WIN32) && !defined(_M_ARM) && !defined(_M_ARM64)
    return _mm_malloc(size, alignment);
#elif defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

void std_aligned_free(void* ptr) {

#if defined(POSIXALIGNEDALLOC)
    free(ptr);
#elif defined(_WIN32) && !defined(_M_ARM) && !defined(_M_ARM64)
    _mm_free(ptr);
#elif defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// aligned_large_pages_alloc() will return suitably aligned memory,
// if possible using large pages.

#if defined(_WIN32)

static void* aligned_large_pages_alloc_windows([[maybe_unused]] usize allocSize) {

    return windows_try_with_large_page_priviliges(
      [&](usize largePageSize) {
          // Round up size to full pages and allocate
          allocSize = (allocSize + largePageSize - 1) & ~usize(largePageSize - 1);
          return VirtualAlloc(nullptr, allocSize, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                              PAGE_READWRITE);
      },
      []() { return (void*) nullptr; });
}

void* aligned_large_pages_alloc_with_hint(usize allocSize, bool) {

    // Try to allocate large pages
    void* mem = aligned_large_pages_alloc_windows(allocSize);

    // Fall back to regular, page-aligned, allocation if necessary
    if (!mem)
        mem = VirtualAlloc(nullptr, allocSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

    return mem;
}

#else

    #if defined(__linux__) && !defined(__ANDROID__)

static std::map<void*, usize> mapped_pages;
static std::mutex             mapped_pages_mtx;

static void remember_mapping(void* mem, usize size) {
    std::lock_guard lg(mapped_pages_mtx);
    mapped_pages.emplace(mem, size);
}

// Anonymous mappings are initially unpopulated, so their pages are committed by
// the NUMA-bound thread which first touches them. Overallocate and trim the VMA
// to retain the alignment needed for transparent huge pages.
static void* aligned_mmap_alloc(usize allocSize) {
    constexpr usize MinAlignment = 2 * 1024 * 1024;

    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0)
        return nullptr;

    const usize alignment = std::max(MinAlignment, usize(pageSize));

    if (allocSize == 0 || allocSize > std::numeric_limits<usize>::max() - (alignment - 1))
        return nullptr;

    const usize size = ((allocSize + alignment - 1) / alignment) * alignment;
    if (size > std::numeric_limits<usize>::max() - alignment)
        return nullptr;

    const usize mappingSize = size + alignment;
    void*       mapping =
      mmap(nullptr, mappingSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (mapping == MAP_FAILED)
        return nullptr;

    const uintptr_t mappingAddress = reinterpret_cast<uintptr_t>(mapping);
    const usize     prefixSize     = (alignment - mappingAddress % alignment) % alignment;
    void* const     mem            = static_cast<char*>(mapping) + prefixSize;
    const usize     suffixSize     = mappingSize - prefixSize - size;

    if (prefixSize != 0 && munmap(mapping, prefixSize) != 0)
    {
        std::cerr << "munmap failed: " << strerror(errno) << std::endl;
        exit(EXIT_FAILURE);
    }

    if (suffixSize != 0 && munmap(static_cast<char*>(mem) + size, suffixSize) != 0)
    {
        std::cerr << "munmap failed: " << strerror(errno) << std::endl;
        exit(EXIT_FAILURE);
    }

        #if defined(MADV_HUGEPAGE)
    madvise(mem, size, MADV_HUGEPAGE);
        #endif

    remember_mapping(mem, size);
    return mem;
}

        #if defined(MAP_HUGE_SHIFT) && defined(__x86_64__)
            #define HAS_HUGE_PAGES

static void* try_huge_pages_alloc(usize allocSize) {
    usize size = ((allocSize + HugePageSize - 1) / HugePageSize) * HugePageSize;
    void* mem  = mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (30 << MAP_HUGE_SHIFT), -1, 0);

    if (mem == MAP_FAILED)
        return nullptr;

    remember_mapping(mem, size);
    return mem;
}
        #endif  // defined(MAP_HUGE_SHIFT) && defined(__x86_64__)

    #endif  // defined(__linux__) && !defined(__ANDROID__)

void* aligned_large_pages_alloc_with_hint(usize allocSize, [[maybe_unused]] bool hugePageHint) {
    #ifdef HAS_HUGE_PAGES
    if (hugePageHint && allocSize >= HugePageSize)
    {
        void* mem = try_huge_pages_alloc(allocSize);
        if (mem)
            return mem;
    }
    #endif

    #if defined(__linux__) && !defined(__ANDROID__)
    void* mem = aligned_mmap_alloc(allocSize);
    #else
        #if defined(__linux__)
    constexpr usize alignment = 2 * 1024 * 1024;  // 2MB page size assumed
        #else
    constexpr usize alignment = 4096;  // small page size assumed
        #endif

    usize size = ((allocSize + alignment - 1) / alignment) * alignment;
    void* mem  = std_aligned_alloc(alignment, size);
        #if defined(MADV_HUGEPAGE)
    madvise(mem, size, MADV_HUGEPAGE);
        #endif
    #endif
    return mem;
}

#endif

void* aligned_large_pages_alloc(usize size) {
    return aligned_large_pages_alloc_with_hint(size, false);
}

bool has_large_pages() {

#if defined(_WIN32)

    constexpr usize page_size = 2 * 1024 * 1024;  // 2MB page size assumed
    void*           mem       = aligned_large_pages_alloc_windows(page_size);
    if (mem == nullptr)
    {
        return false;
    }
    else
    {
        aligned_large_pages_free(mem);
        return true;
    }

#elif defined(__linux__)

    #if defined(MADV_HUGEPAGE)
    return true;
    #else
    return false;
    #endif

#else

    return false;

#endif
}


// aligned_large_pages_free() will free the previously memory allocated
// by aligned_large_pages_alloc(). The effect is a nop if mem == nullptr.

#if defined(_WIN32)

void aligned_large_pages_free(void* mem) {

    if (mem && !VirtualFree(mem, 0, MEM_RELEASE))
    {
        DWORD err = GetLastError();
        std::cerr << "Failed to free large page memory. Error code: 0x" << std::hex << err
                  << std::dec << std::endl;
        exit(EXIT_FAILURE);
    }
}

#else

void aligned_large_pages_free(void* mem) {
    if (!mem)
        return;

    #if defined(__linux__) && !defined(__ANDROID__)
    usize mappingSize = 0;
    {
        std::lock_guard lg(mapped_pages_mtx);
        if (auto it = mapped_pages.find(mem); it != mapped_pages.end())
        {
            mappingSize = it->second;
            mapped_pages.erase(it);
        }
    }

    if (mappingSize != 0)
    {
        if (munmap(mem, mappingSize) != 0)
        {
            std::cerr << "munmap failed: " << strerror(errno) << std::endl;
            exit(EXIT_FAILURE);
        }
        return;
    }

    std::cerr << "Attempted to free an unknown large-page mapping" << std::endl;
    exit(EXIT_FAILURE);
    #else
    std_aligned_free(mem);
    #endif
}

#endif
}  // namespace Stockfish

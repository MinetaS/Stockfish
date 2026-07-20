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

#ifndef THREAD_NATIVE_H_INCLUDED
#define THREAD_NATIVE_H_INCLUDED

#ifdef _MSC_VER
    #include <thread>
#else
    #include <cstdlib>
    #include <cstring>
    #include <functional>
    #include <iostream>
    #include <pthread.h>
    #include <utility>

    #include "misc.h"
    #include "process.h"

    #ifdef __linux__
        #include <sys/mman.h>
    #endif
#endif

namespace Stockfish {

#ifdef _MSC_VER

// MSVC-compatible toolchains use std::thread because they do not provide
// pthreads by default. On all other platforms, pthreads is required and used.

using NativeThread = std::thread;

#else

// On OSX threads other than the main thread are created with a reduced stack
// size of 512KB by default, this is too low for deep searches, which require
// somewhat more than 1MB stack, so adjust it to TH_STACK_SIZE.
// The implementation calls pthread_create() with the stack size parameter
// equal to the Linux 8MB default, on platforms that support it.

class NativeThread {
   public:
    template<class Function, class... Args>
    explicit NativeThread(Function&& fun, Args&&... args) {
        auto func = new std::function<void()>(
          std::bind(std::forward<Function>(fun), std::forward<Args>(args)...));

        pthread_attr_t attr;
        pthread_attr_init(&attr);

        // Caller-owned stack mappings ensure that a recreated thread cannot
        // inherit resident stack pages from glibc's process-wide pthread stack
        // cache. Guard pages are mapped explicitly because
        // pthread_attr_setguardsize() is ignored when pthread_attr_setstack()
        // supplies the stack storage.
        map_stack();

        if (is_stack_mapped())
        {
            pthread_attr_setguardsize(&attr, 0);
            pthread_attr_setstack(&attr, stack(), kStackSize);
        }
        else
        {
            pthread_attr_setstacksize(&attr, kStackSize);
        }

        auto start_routine = [](void* ptr) -> void* {
            auto f = reinterpret_cast<std::function<void()>*>(ptr);
            // Call the function
            (*f)();
            delete f;
            return nullptr;
        };

        const int error = pthread_create(&thread_, &attr, start_routine, func);
        if (error != 0)
        {
            delete func;
            unmap_stack();

            std::cerr << "Failed to create thread: " << std::strerror(error) << " (" << error << ")"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void join() {
        pthread_join(thread_, nullptr);
        unmap_stack();
    }

   private:
    static constexpr usize kStackSize = 8 * 1024 * 1024;

    void map_stack() {
    #ifdef __linux__
        const usize memSize = kStackSize + 2 * Process::gPageSize;

        // Assume Linux >= 2.6.27
        void* const m =
          mmap(nullptr, memSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
        if (m == MAP_FAILED)
            return;

        stack_memory_ = m;
        stack_size_   = memSize;

        if (mprotect(stack(), kStackSize, PROT_READ | PROT_WRITE) != 0)
        {
            unmap_stack();
            return;
        }
    #endif
    }

    void unmap_stack() {
        if (stack_memory_ == nullptr)
            return;

    #ifdef __linux__
        munmap(stack_memory_, stack_size_);
    #endif

        stack_memory_ = nullptr;
    }

    inline constexpr bool is_stack_mapped() const { return stack_memory_ != nullptr; }

    inline void* stack() const {
        assert(stack_memory_ != nullptr);
        return static_cast<char*>(stack_memory_) + Process::gPageSize;
    }

    pthread_t thread_;
    void*     stack_memory_ = nullptr;
    usize     stack_size_   = 0;
};

#endif  // _MSC_VER

}  // namespace Stockfish

#endif  // #ifndef THREAD_NATIVE_H_INCLUDED

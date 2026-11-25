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

#include "gzip.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <memory>
#include <optional>
#include <streambuf>

#include "miniz/miniz_tinfl.c"

namespace Stockfish::Gzip {
namespace {

constexpr auto CrcTable = [] {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t i = 0; i < table.size(); ++i)
    {
        auto crc = i;
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320u : 0);
        table[i] = crc;
    }
    return table;
}();

std::uint32_t update_crc(std::uint32_t crc, unsigned byte) {
    return (crc >> 8) ^ CrcTable[(crc ^ byte) & 0xff];
}

class GzipBuffer final: public std::streambuf {
   public:
    explicit GzipBuffer(std::istream& stream) :
        source(stream) { }

    bool complete() const { return state == State::End; }

   private:
    enum class State {
        Header,
        Body,
        Trailer,
        End,
        Error
    };

    std::istream&                        source;
    State                                state = State::Header;
    std::array<unsigned char, 32 * 1024> input;
    std::size_t                          next = 0, end = 0;

    tinfl_decompressor                            inflater{};
    std::array<unsigned char, TINFL_LZ_DICT_SIZE> window;
    std::size_t                                   offset;
    bool                                          firstWindow;
    std::uint32_t                                 crc, size;

    bool has_input() {
        if (next == end)
        {
            source.read(reinterpret_cast<char*>(input.data()), input.size());
            next = 0;
            end  = source.gcount();
        }
        return next != end && !source.bad();
    }

    int get_byte() { return has_input() ? input[next++] : traits_type::eof(); }

    std::optional<std::uint32_t> read_le(unsigned bytes) {
        std::uint32_t value = 0;
        for (unsigned i = 0; i < bytes; ++i)
        {
            int byte = get_byte();
            if (byte == traits_type::eof())
                return std::nullopt;
            value |= std::uint32_t(byte) << (8 * i);
        }
        return value;
    }

    bool read_header() {
        std::uint32_t headerCrc   = 0xffffffffu;
        auto          header_byte = [&] {
            int byte = get_byte();
            if (byte != traits_type::eof())
                headerCrc = update_crc(headerCrc, byte);
            return byte;
        };

        if (header_byte() != 0x1f || header_byte() != 0x8b || header_byte() != 8)
            return false;
        int flags = header_byte();
        if (flags == traits_type::eof() || (flags & 0xe0))
            return false;
        for (int i = 0; i < 6; ++i)  // MTIME, XFL and OS
            if (header_byte() == traits_type::eof())
                return false;

        if (flags & 4)  // FEXTRA
        {
            int lo = header_byte(), hi = header_byte();
            if (lo == traits_type::eof() || hi == traits_type::eof())
                return false;
            for (int i = lo + (hi << 8); i > 0; --i)
                if (header_byte() == traits_type::eof())
                    return false;
        }
        for (int flag : {8, 16})  // FNAME and FCOMMENT
            if (flags & flag)
                for (int byte; (byte = header_byte()) != 0;)
                    if (byte == traits_type::eof())
                        return false;

        return !(flags & 2) || read_le(2) == (~headerCrc & 0xffffu);  // FHCRC
    }

    int_type underflow() override {
        if (state == State::End || state == State::Error)
            return traits_type::eof();
        for (;;)
        {
            if (state == State::Header)
            {
                if (!read_header())
                    break;
                tinfl_init(&inflater);
                offset      = 0;
                firstWindow = true;
                crc         = 0xffffffffu;
                size        = 0;
                state       = State::Body;
            }
            if (state == State::Trailer)
            {
                if (read_le(4) != ~crc || read_le(4) != size)
                    break;
                if (has_input())
                    state = State::Header;  // Another gzip member follows.
                else if (source.eof() && !source.bad())
                {
                    state = State::End;
                    return traits_type::eof();
                }
                else
                    break;
                continue;
            }
            if (!has_input())
                break;

            auto*       out      = window.data() + offset;
            std::size_t written  = window.size() - offset;
            std::size_t consumed = end - next;
            // Until the dictionary is full, forbid references before the start of this member.
            unsigned flags = TINFL_FLAG_HAS_MORE_INPUT;
            if (firstWindow)
                flags |= TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
            auto status = tinfl_decompress(&inflater, input.data() + next, &consumed, window.data(),
                                           out, &written, flags);
            if (status < 0)
                break;
            next += consumed;
            if (status == TINFL_STATUS_DONE)
                state = State::Trailer;
            if (written)
            {
                for (std::size_t i = 0; i < written; ++i)
                    crc = update_crc(crc, out[i]);
                size += std::uint32_t(written);
                offset += written;
                if (offset == window.size())
                {
                    offset      = 0;
                    firstWindow = false;
                }
                auto* data = reinterpret_cast<char*>(out);
                setg(data, data, data + written);
                return traits_type::to_int_type(*gptr());
            }
        }
        state = State::Error;
        return traits_type::eof();
    }
};

}  // namespace

bool is_compressed(std::istream& source) {
    if (source.peek() != 0x1f)
        return false;

    source.get();
    bool gzip = source.peek() == 0x8b;
    source.unget();
    return gzip;
}

bool decompress(std::istream& source, const std::function<bool(std::istream&)>& readPayload) {
    auto         buffer = std::make_unique<GzipBuffer>(source);
    std::istream decoded(buffer.get());
    return readPayload(decoded) && buffer->complete();
}

}  // namespace Stockfish::Gzip

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

#include <iostream>

#include "bitboard.h"
#include "misc.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "tune.h"

namespace Stockfish {

Value PawnValue   = 208;
Value KnightValue = 781;
Value BishopValue = 825;
Value RookValue   = 1276;
Value QueenValue  = 2538;

Value PieceValue[PIECE_NB] = { 0 };

TUNE(PawnValue, KnightValue, BishopValue, RookValue, QueenValue);

} // namespace Stockfish

using namespace Stockfish;

int main(int argc, char* argv[]) {
    PieceValue[W_PAWN] = PieceValue[B_PAWN] = PawnValue;
    PieceValue[W_KNIGHT] = PieceValue[B_KNIGHT] = KnightValue;
    PieceValue[W_BISHOP] = PieceValue[B_BISHOP] = BishopValue;
    PieceValue[W_ROOK] = PieceValue[B_ROOK] = RookValue;
    PieceValue[W_QUEEN] = PieceValue[B_QUEEN] = QueenValue;

    std::cout << engine_info() << std::endl;

    Bitboards::init();
    Position::init();

    UCIEngine uci(argc, argv);

    Tune::init(uci.engine_options());

    uci.loop();

    return 0;
}

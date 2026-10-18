// Dumps every legal move of each board with the engine's policy index.
// stdin: one bughouse FEN per line ("<fenA>|<fenB>", as Board::set accepts).
// stdout TSV: pos  board  stm  uci  type  from  to  policy_index
#include <iostream>
#include <string>

#include "common/globals.h"
#include "common/utils.h"
#include "environment/board.h"

namespace {

const char* move_type_name(Stockfish::Move move) {
    if (move == Stockfish::MOVE_NONE) return "pass";
    switch (Stockfish::type_of(move)) {
        case Stockfish::NORMAL: return "normal";
        case Stockfish::PROMOTION: return "promotion";
        case Stockfish::EN_PASSANT: return "en_passant";
        case Stockfish::CASTLING: return "castling";
        case Stockfish::DROP: return "drop";
        default: return "other";
    }
}

}  // namespace

int main() {
    init_fairy_stockfish();
    init_policy_index();

    std::string line;
    int position = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;
        Board board;
        board.set(line);
        for (int boardNum : {BOARD_A, BOARD_B}) {
            const Stockfish::Color stm = board.side_to_move(boardNum);
            auto moves = board.legal_moves(boardNum);
            moves.push_back(Stockfish::MOVE_NONE);  // the engine's pass/sit action
            for (Stockfish::Move move : moves) {
                const bool real = move != Stockfish::MOVE_NONE;
                std::cout << position << '\t' << boardNum << '\t'
                          << (stm == Stockfish::WHITE ? 'w' : 'b') << '\t'
                          << board.uci_move(boardNum, move) << '\t'
                          << move_type_name(move) << '\t'
                          << (real ? static_cast<int>(Stockfish::from_sq(move)) : -1) << '\t'
                          << (real ? static_cast<int>(Stockfish::to_sq(move)) : -1) << '\t'
                          << get_fast_policy_index(move, stm) << '\n';
            }
        }
        ++position;
    }
}

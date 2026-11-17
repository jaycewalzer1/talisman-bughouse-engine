// Replays bughouse games and dumps the engine's input planes and network
// outputs, for comparison against the Python training encoder.
//
// usage: planes_dump <model.onnx> <out.bin>   (records on stdin)
// stdin, one record per line, tab separated:
//   team(w|b)  time_advantage(0|1)  <fenA>|<fenB>  moves  [clocks]  [elapsed]
// where moves is space separated "<board>:<uci>" (may be empty), the
// optional clocks field is "<aw> <ab> <bw> <bb>" in ms (Board::set_clocks,
// as UCI 'position ... clocks') and the optional elapsed field is "<ea> <eb>"
// in ms (Board::set_elapsed, as UCI '... elapsed'; only applied with clocks).
// Built against build-clock (planes_dump_clock) the planes have 43 channels
// per board (spec v2).
// out.bin, per record, float32: planes[NB_INPUT_VALUES], value, wdl[3],
// moves_left, pi_a[NB_POLICY_VALUES], pi_b[NB_POLICY_VALUES], and in the
// clock build two more floats: the think_gain logits for A and B (0 when the
// network has no think_gain output).
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common/globals.h"
#include "environment/board.h"
#include "environment/planes.h"
#include "nn/engine.h"
#include "Fairy-Stockfish/src/uci.h"

namespace {

std::vector<std::string> split(const std::string& line, char sep) {
    std::vector<std::string> parts;
    std::stringstream stream(line);
    std::string part;
    while (std::getline(stream, part, sep)) parts.push_back(part);
    return parts;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: planes_dump <model.onnx> <out.bin>\n";
        return 2;
    }
    init_fairy_stockfish();
    init_policy_index();

    Engine engine(0, 1);
    if (!engine.loadNetwork(argv[1], "")) {
        std::cerr << "failed to load " << argv[1] << "\n";
        return 1;
    }
    std::ofstream out(argv[2], std::ios::binary);

    std::vector<float> planes(NB_INPUT_VALUES());
    std::vector<float> piA(NB_POLICY_VALUES()), piB(NB_POLICY_VALUES());
    float value = 0.0f, wdl[3] = {}, movesLeft = 0.0f;

    std::string line;
    int record = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        const auto fields = split(line, '\t');
        if (fields.size() < 3) {
            std::cerr << "record " << record << ": expected >= 3 fields\n";
            return 1;
        }
        const Stockfish::Color team = fields[0] == "w" ? Stockfish::WHITE : Stockfish::BLACK;
        const bool timeAdvantage = fields[1] == "1";

        Board board;
        board.set(fields[2]);
        if (fields.size() > 3) {
            for (const std::string& token : split(fields[3], ' ')) {
                if (token.empty()) continue;
                const int boardNum = token[0] - '0';
                std::string uci = token.substr(2);
                const Stockfish::Move move = Stockfish::UCI::to_move(*board.pos[boardNum], uci);
                if (move == Stockfish::MOVE_NONE) {
                    std::cerr << "record " << record << ": illegal move " << token
                              << " in " << board.fen(boardNum) << "\n";
                    return 1;
                }
                board.push_move(boardNum, move);
            }
        }

        if (fields.size() > 4 && !fields[4].empty()) {
            std::stringstream clocks(fields[4]);
            int64_t aw, ab, bw, bb;
            if (!(clocks >> aw >> ab >> bw >> bb)) {
                std::cerr << "record " << record << ": bad clocks '" << fields[4] << "'\n";
                return 1;
            }
            board.set_clocks(aw, ab, bw, bb);
            if (fields.size() > 5 && !fields[5].empty()) {
                std::stringstream elapsed(fields[5]);
                int64_t ea, eb;
                if (!(elapsed >> ea >> eb)) {
                    std::cerr << "record " << record << ": bad elapsed '" << fields[5] << "'\n";
                    return 1;
                }
                board.set_elapsed(ea, eb);
            }
        }

        board_to_planes(board, planes.data(), team, timeAdvantage);
        if (!engine.runInference(planes.data(), &value, piA.data(), piB.data(), wdl, &movesLeft)) {
            std::cerr << "record " << record << ": inference failed\n";
            return 1;
        }
        out.write(reinterpret_cast<const char*>(planes.data()), planes.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(&value), sizeof(float));
        out.write(reinterpret_cast<const char*>(wdl), sizeof(wdl));
        out.write(reinterpret_cast<const char*>(&movesLeft), sizeof(float));
        out.write(reinterpret_cast<const char*>(piA.data()), piA.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(piB.data()), piB.size() * sizeof(float));
#ifdef HIVEMIND_CLOCK_PLANES
        float thinkGain[2] = {0.0f, 0.0f};
        if (engine.hasThinkGain() && !engine.evaluateThinkGain(planes.data(), thinkGain)) {
            std::cerr << "record " << record << ": think_gain failed\n";
            return 1;
        }
        out.write(reinterpret_cast<const char*>(thinkGain), sizeof(thinkGain));
#endif
        ++record;
    }
    std::cerr << "wrote " << record << " records\n";
}

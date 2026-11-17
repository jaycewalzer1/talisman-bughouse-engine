"""Writes tools/positions/sit_move.jsonl and drop_mates.jsonl (audit 2026-11-17 WP-H; theory 02 section 2.8, theory
04 section 2.4). The JSONL files are the source the tools read; this script records how they were made. Base FENs
were hand-built or found by a constrained random search and every claim is re-checked by
``tools/regression.py verify``. The puzzles' ``best`` lists are the solver's (``forced_mate_moves``) at the given depth.

    .venv/bin/python tools/positions/build_sets.py && .venv/bin/python tools/regression.py verify
"""
import sys, json; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from regression import *
def clk(us_att, opp_att, us_sup, opp_sup): return {"us_att":us_att,"opp_att":opp_att,"us_sup":us_sup,"opp_sup":opp_sup}
def diag(tag, d, expect):  # attacker diagonal d s: us_att 30+d/2, opp_sup 30-d/2
    h=int(d*500); return {"tag":tag,"clocks":clk(30000+h,30000,30000,30000-h),"expect":expect}
A_MINOR = "r5nk/pp5p/5P2/8/8/8/PP3PPP/6K1[] w - - 0 1"
A_SMOTH = "6rk/6pp/8/8/8/8/PP3PPP/6K1[] w - - 0 1"
A_ROOK = "1n4k1/pp3ppp/8/8/8/B7/PP3PPP/6K1[] w - - 0 1"
A_NC7 = "3rk3/3b1r2/8/2B5/8/2R5/PP3PPP/6K1[] w - - 0 1"
A_NC7G7 = "4kr2/3bBn1p/8/8/7B/8/PP3PPP/6K1[] w - - 0 1"
HANG = "6k1/5ppp/4p3/3{X}4/8/8/5PPP/6K1[] b - - 0 1"
PIN_N_E4 = "4r1k1/5ppp/8/3p4/4N3/8/PPP2PPP/4K3[] w - - 0 1"
PIN_N_D4 = "6k1/b4ppp/8/4p3/3N4/8/PPP3PP/6K1[] w - - 0 1"
PIN_B_E4 = "4r1k1/5ppp/8/3p4/4B3/8/PPP2PPP/4K3[] w - - 0 1"
QUIET_W = "6k1/5ppp/8/8/8/8/5PPP/6K1[] w - - 0 1"   # partner's opponent to move, nothing to capture
QUIET_B = "6k1/5ppp/8/8/8/8/5PPP/6K1[] b - - 0 1"   # partner to move, nothing to capture
OPEN_W = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R[] w KQkq - 2 3"
sit=[]
def S(id,fam,case,att,sup,claims,clocks,note):
    sit.append({"id":id,"set":"sit_move","family":fam,"case":case,"attack":att,"supply":sup,"claims":claims,"clocks":clocks,"note":note})
both_sit=[diag("diag+10",10,"sit"),diag("diag-10",-10,"sit")]
for X,att,drops in [("Q",A_MINOR,["Q@g7"]),("B",A_MINOR,["B@g7"]),("P",A_MINOR,["P@g7"]),("N",A_MINOR,["N@f7"]),
                    ("N",A_SMOTH,["N@f7"]),("R",A_ROOK,["R@f8"]),("N",A_NC7,["N@c7"]),("N",A_NC7G7,["N@c7","N@g7"])]:
    nm={A_MINOR:"minor",A_SMOTH:"smoth",A_ROOK:"rook",A_NC7:"nc7",A_NC7G7:"nc7g7"}[att]
    S(f"supply-{nm}-{X}","partner_supplies","02 2.8(2)",att,HANG.format(X=X),
      {"mating_drops":drops,"no_mate_now":True,"supply_capture":"e6d5","piece":X},both_sit,
      f"partner on turn wins a {X} with ...exd5; {drops[0]} then mates. Sit costs only absolute time (30 s).")
so=[diag("diag+10",10,"sit"),diag("diag+3",3,"sit"),diag("diag-3",-3,"move"),diag("diag-10",-10,"move")]
for nm,att,drop,sup,X in [("minor-pinNe4",A_MINOR,"N@f7",PIN_N_E4,"N"),("smoth-pinNd4",A_SMOTH,"N@f7",PIN_N_D4,"N"),
                          ("minor-pinBe4",A_MINOR,"B@g7",PIN_B_E4,"B"),("smoth-pinNe4",A_SMOTH,"N@f7",PIN_N_E4,"N"),
                          ("nc7-pinNd4",A_NC7,"N@c7",PIN_N_D4,"N")]:
    S(f"sitoff-{nm}","sitoff_for_piece","02 2.7/2.8",att,sup,
      {"mating_drops":[drop],"no_mate_now":True,"pinned_supply":True,"piece":X},so,
      f"partner's opponent to move; any move of his loses the pinned {X}, so he sits too: the sit-off is decided by "
      f"the attacker's diagonal. Ahead: sit; behind: move.")
F3=["6k1/5ppp/8/8/8/7p/PP2n1PP/7K[q] w - - 0 1","6k1/5ppp/8/8/8/4b3/PP4PP/7K[qn] w - - 0 1",
    "6k1/5ppp/8/8/8/8/PP2nPPP/7K[qn] w - - 0 1","6k1/5ppp/8/8/8/5n2/PP3pPP/7K[rp] w - - 0 1"]
for i,att in enumerate(F3):
    S(f"lose-{i+1}","every_move_loses","02 2.8(3)",att,QUIET_B,{"every_move_allows_mate":True},both_sit,
      "every move on the attack board allows a drop mate; sitting is strictly better whatever the diagonal")
for nm,att,drop,X in [("minor",A_MINOR,"N@f7","N"),("smoth",A_SMOTH,"N@f7","N"),("rook",A_ROOK,"R@f8","R"),("nc7",A_NC7,"N@c7","N")]:
    S(f"nosource-{nm}","no_source","02 2.7 (don't sit)",att,QUIET_W,{"mating_drops":[drop],"no_mate_now":True,"no_supply":True,"piece":X},
      [diag("diag+10",10,"move"),diag("diag-10",-10,"move")],
      f"{drop} would mate but no {X} can come: the partner's opponent has none. Sitting only burns time.")
eq=[{"tag":"eq","clocks":clk(30000,30000,30000,30000)},{"tag":"lead+2","clocks":clk(31000,30000,30000,29000)}]
puz=[]
def P(id,fam,att,depth,note):
    best=forced_mate_moves(chess.variant.CrazyhouseBoard(att),depth)
    puz.append({"id":id,"set":"drop_mates","family":fam,"attack":att,"supply":OPEN_W,"claims":{"forced_mate":depth},
                "best":best,"value_sign":1,"clocks":eq,"note":note})
P("smothered-1","smothered","6rk/6pp/8/8/8/8/PP3PPP/6K1[N] w - - 0 1",1,"N@f7#")
P("smothered-2","smothered","5r1k/6pp/7N/8/8/8/PP3PPP/6K1[R] w - - 0 1",2,"R@g8+ Rxg8 Nf7#")
P("smothered-3","smothered","5rk1/7p/5Pp1/8/8/8/PP3PPP/6K1[NR] w - - 0 1",3,"N@h6+ Kh8 R@g8+ Rxg8 Nf7#")
P("yoyo-1","yoyo","5rk1/5ppp/8/8/8/8/PP3PPP/6K1[NNNPQ] w - - 0 1",4,"hook check N@h6+ (theory 04 2.4, mirrored)")
P("f7-1","f7","r1bqkb1r/pppp2pp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R[Q] w KQkq - 0 1",1,"Q@f7#")
P("f7-2","f7","r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R[N] w KQkq - 0 1",2,"Bxf7+ Ke7 N@d5#")
P("f7-3","f7","r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R[NQ] w KQkq - 0 1",2,"Bxf7+ Ke7 N@d5# with a queen in hand too")
P("clamp-g7","clamp","r5nk/pp5p/5P2/8/8/8/PP3PPP/6K1[Q] w - - 0 1",1,"Q@g7# (f6 pawn clamp)")
P("contact-f8","contact","1n4k1/pp3ppp/8/8/8/B7/PP3PPP/6K1[R] w - - 0 1",1,"R@f8# guarded by Ba3")
P("smothered-4","smothered","6rk/6pp/8/8/8/8/PP3PPP/6K1[NQ] w - - 0 1",1,"N@f7# with a queen in hand (Q@ drops do not mate)")
for path,rows in ((SIT_MOVE_FILE,sit),(DROP_MATE_FILE,puz)):
    with open(path,"w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print(path,len(rows))

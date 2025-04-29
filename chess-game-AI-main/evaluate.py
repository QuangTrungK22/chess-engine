import time
from chess_engine import GameState
import algorithm_utils

def evaluate_agents(num_games = 1000):
    minimax_wins = 0
    random_wins = 0
    draws = 0
    start_total_time = time.time()

    for i in range(num_games):
        gs = GameState()
        game_start_time = time.time()
        print(f"Starting game {i + 1} of {num_games}...")

        #luan phien doi mau cho moi agent
        minimax_is_white = i % 2 == 0

        while not gs.check_mate and not gs.state_mate:
            valid_moves = gs.get_valid_moves()
            if not valid_moves:
                break
            move = None
            if (gs.white_to_move  and minimax_is_white) or (not gs.white_to_move and not minimax_is_white):
                move = algorithm_utils.minimax(gs, valid_moves)
            else:
                move = algorithm_utils.random_move(valid_moves)
            
            if move is None:
                 print(f"Warning: Agent could not find move in game {i+1}. State: Checkmate={gs.check_mate}, Stalemate={gs.stale_mate}")
            break

            gs.make_move(move)
 # Ghi nhận kết quả
        winner = ""
        if gs.check_mate:
            if gs.white_to_move: # Người đi lượt tiếp theo là Trắng -> Đen vừa chiếu hết
                if minimax_is_white:
                    random_wins += 1
                    winner = "Random (Black)"
                else:
                    minimax_wins += 1
                    winner = "Minimax (Black)"
            else: # Người đi lượt tiếp theo là Đen -> Trắng vừa chiếu hết
                if minimax_is_white:
                    minimax_wins += 1
                    winner = "Minimax (White)"
                else:
                    random_wins += 1
                    winner = "Random (White)"
        elif gs.stale_mate:
            draws += 1
            winner = "Draw"
        else:
             # Trường hợp vòng lặp kết thúc mà không phải checkmate/stalemate (ví dụ do lỗi)
             print(f"Warning: Game {i+1} ended unexpectedly.")
             draws +=1 # Coi như hòa trong trường hợp này
             winner = "Draw (Unexpected)"


        game_end_time = time.time()
        print(f"Game {i+1} finished. Winner: {winner}. Duration: {game_end_time - game_start_time:.2f} sec")

    end_total_time = time.time()
    print("\n--- Evaluation Results ---")
    print(f"Total Games: {num_games}")
    print(f"Minimax Wins: {minimax_wins} ({minimax_wins/num_games:.1%})")
    print(f"Random Wins: {random_wins} ({random_wins/num_games:.1%})")
    print(f"Draws: {draws} ({draws/num_games:.1%})")
    print(f"Total Evaluation Time: {end_total_time - start_total_time:.2f} sec")

if __name__ == "__main__":
  algorithm_utils.MAX_DEPTH = 2 # Giảm độ sâu để test nhanh hơn
  print(f"Running evaluation with Minimax depth: {algorithm_utils.MAX_DEPTH}")
  evaluate_agents(num_games=10)
        

            

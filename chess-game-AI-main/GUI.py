import tkinter as tk
from tkinter import ttk
from main import start_game, undo_move, reset_game, play_white, play_black  # Import các hàm từ main.py

class ChessGameGUI:
    def __init__(self, master):
        self.master = master
        p.display.set_caption("Chess Game with Minimax AI")  # Thay thế master.title() bằng pygame.display.set_caption()

        # --- CHẾ ĐỘ CHƠI ---
        ttk.Label(master, text="Chế độ chơi").grid(row=0, column=0, padx=10, pady=5)
        self.mode_var = tk.StringVar(value="Người vs Người")
        self.mode_menu = ttk.Combobox(master, textvariable=self.mode_var, values=["Người vs Người", "Người vs AI", "AI vs AI"])
        self.mode_menu.grid(row=0, column=1)

        # --- QUẢN LÝ NƯỚC ĐI ---
        self.undo_button = ttk.Button(master, text="Hoàn tác nước đi (Z)", command=self.undo_move)
        self.undo_button.grid(row=1, column=0, pady=5)

        self.reset_button = ttk.Button(master, text="Khởi tạo lại (R)", command=self.reset_game)
        self.reset_button.grid(row=1, column=1, pady=5)

        self.white_button = ttk.Button(master, text="Chơi bên trắng (E)", command=self.play_white)
        self.white_button.grid(row=2, column=0, pady=5)

        self.black_button = ttk.Button(master, text="Chơi bên đen (Q)", command=self.play_black)
        self.black_button.grid(row=2, column=1, pady=5)

        # --- NÚT CHẠY --- (Chọn chế độ và bắt đầu chơi)
        self.start_button = ttk.Button(master, text="Bắt đầu trò chơi", command=self.run_game)
        self.start_button.grid(row=3, column=0, columnspan=2, pady=10)

        # --- KẾT QUẢ ---
        self.result_label = ttk.Label(master, text="Kết quả sẽ hiển thị ở đây")
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)

    def run_game(self):
        mode = self.mode_var.get()
        self.game_state = start_game(mode)  # Khởi động trò chơi với chế độ đã chọn
        self.result_label.config(text=f"Chế độ: {mode} đang chạy...")

    def undo_move(self):
        self.game_state = undo_move(self.game_state)  # Thực hiện undo move
        self.result_label.config(text="Đã hoàn tác nước đi.")

    def reset_game(self):
        self.game_state = reset_game()  # Reset trò chơi
        self.result_label.config(text="Trò chơi đã được reset.")

    def play_white(self):
        self.game_state = play_white(self.game_state)  # Chơi với AI bên trắng
        self.result_label.config(text="Chơi với AI bên trắng.")

    def play_black(self):
        self.game_state = play_black(self.game_state)  # Chơi với AI bên đen
        self.result_label.config(text="Chơi với AI bên đen.")


if __name__ == "__main__":
    root = tk.Tk()
    gui = ChessGameGUI(root)
    root.mainloop()

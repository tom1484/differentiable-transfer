from libtmux import Server, Window
from libtmux.constants import PaneDirection


def create_grid_window(name: str, grid_num: int, rc_ratio: int = 2) -> Window:
    tmux_svr = Server()
    session = tmux_svr.new_session(session_name=name, window_name="main")
    window = session.active_window

    col = grid_num // (rc_ratio + 1)
    col = col + 1 if col == 0 else col

    row = grid_num // col
    if col * row < grid_num:
        row += 1

    for i in range(row):
        p = window.split(direction=PaneDirection.Below)
        for j in range(col - 1):
            p.split(direction=PaneDirection.Right)

    window.cmd("kill-pane")
    window.select_layout("tiled")

    return window

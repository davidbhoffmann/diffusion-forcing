from collections import deque
from omegaconf.omegaconf import open_dict
import numpy as np


def get_solutions_tree(maze, goal_pos: tuple, length:int) -> tuple[dict, list]:
    """
    Compute a shortest paths tree from all valid nodes in the maze to the 
    specified goal position.

    Args:
    maze SolvedMaze: from maze_dataset.
    goal_pos Tuple[int, int]:  with the index position of the goal only 
        counting valid nodes not intermediate steps.
    min_length Int: specifies the minimum shortest path length to the goal

    Return:
    tuple
        paths_tree dict[tuple[int,int], tuple[int,int]] which gives us the 
            next step toward the goal for any valid position in the maze.
        start_nodes list of start nodes with a shortest path length longer 
            than min_length

    """

    q = deque([(goal_pos, 0)])
    # paths_tree: dict[tuple[int, int], dict] = {goal_pos: {"next": None, "goal_dist": 0}}
    paths_tree: dict[tuple[int, int], tuple[int, int]|None] = {goal_pos: None}
    start_nodes = []

    while q:
        (r, c), goal_distance = q.popleft()
        goal_distance +=1
        
        neighbors = []
        # Note: connection_list[0, r, c] tells us if there is a connection downward. Hence the 
        # following two line checks if we can go down and up respectively
        if r < maze.grid_n - 1 and maze.connection_list[0, r, c]: neighbors.append((r + 1, c))
        if r > 0 and maze.connection_list[0, r - 1, c]: neighbors.append((r - 1, c))
        # connection_list[1, r, c] tells us if we can go rightward. Hence the following two lines 
        # check if we can go right or left respectively.
        if c < maze.grid_n - 1 and maze.connection_list[1, r, c]: neighbors.append((r, c + 1))
        if c > 0 and maze.connection_list[1, r, c - 1]: neighbors.append((r, c - 1))
        
        for n in neighbors:
            if n not in paths_tree:
                # paths_tree[n] = {"next": (r, c), "goal_dist": goal_distance}
                paths_tree[n] = (r, c)
                q.append((n, goal_distance))
                if goal_distance <= length: start_nodes.append(n)

    return paths_tree, start_nodes

def get_action(curr: tuple[int, int], nxt: tuple[int, int]| None) -> int:
    """
    Derives the action necessary to transition from curr to next position.
    Actions: 1 = up, 2 = down, 3 = left, 4 = right.
    """
    if nxt==curr or nxt==None:
        a = 0 # Do nothing
    elif nxt[0] < curr[0]:
        a = 1  # Up
    elif nxt[0] > curr[0]:
        a = 2  # Down
    elif nxt[1] < curr[1]:
        a = 3  # Left
    else:
        a = 4  # Right
    return a 

        
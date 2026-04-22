
import numpy as np
from typing import List

U, L, F, R, B, D = 0, 1, 2, 3, 4, 5
FACE_NAMES = ['U', 'L', 'F', 'x`R', 'B', 'D']
COLORS = {
    0: 'W',
    1: 'O',
    2: 'G',
    3: 'R',
    4: 'B',
    5: 'Y'
}

class RubikCube:
    def __init__(self, state: np.ndarray = None):
        if state is not None:
            self.state = state.copy().astype(np.uint8)
        else:
            self.state = self.get_solved_state()

    @staticmethod
    def get_solved_state() -> np.ndarray:
        state = np.zeros(54, dtype=np.uint8)
        for face in range(6):
            state[face*9 : (face+1)*9] = face
        return state

    def copy(self) -> 'RubikCube':
        return RubikCube(self.state.copy())

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def set_state(self, state: np.ndarray):
        self.state = state.copy().astype(np.uint8)

    def is_solved(self) -> bool:
        solved = self.get_solved_state()
        return np.array_equal(self.state, solved)

    def apply_move(self, move: str):
        if move == 'U':
            self._rotate_U()
        elif move == "U'":
            self._rotate_U_prime()
        elif move == 'D':
            self._rotate_D()
        elif move == "D'":
            self._rotate_D_prime()
        elif move == 'L':
            self._rotate_L()
        elif move == "L'":
            self._rotate_L_prime()
        elif move == 'R':
            self._rotate_R()
        elif move == "R'":
            self._rotate_R_prime()
        elif move == 'F':
            self._rotate_F()
        elif move == "F'":
            self._rotate_F_prime()
        elif move == 'B':
            self._rotate_B()
        elif move == "B'":
            self._rotate_B_prime()
        else:
            raise ValueError(f"Invalid move: {move}")

    def apply_moves(self, moves: List[str]):
        for move in moves:
            self.apply_move(move)

    def _rotate_face_clockwise(self, face_idx: int):
        start = face_idx * 9
        face = self.state[start:start+9].reshape(3, 3)
        rotated = np.rot90(face, -1)
        self.state[start:start+9] = rotated.flatten()

    def _rotate_U(self):
        self._rotate_face_clockwise(U)
        temp = self.state[F*9 : F*9+3].copy()
        self.state[F*9 : F*9+3] = self.state[R*9 : R*9+3]
        self.state[R*9 : R*9+3] = self.state[B*9 : B*9+3]
        self.state[B*9 : B*9+3] = self.state[L*9 : L*9+3]
        self.state[L*9 : L*9+3] = temp

    def _rotate_U_prime(self):
        for _ in range(3):
            self._rotate_U()

    def _rotate_D(self):
        self._rotate_face_clockwise(D)
        temp = self.state[F*9+6 : F*9+9].copy()
        self.state[F*9+6 : F*9+9] = self.state[L*9+6 : L*9+9]
        self.state[L*9+6 : L*9+9] = self.state[B*9+6 : B*9+9]
        self.state[B*9+6 : B*9+9] = self.state[R*9+6 : R*9+9]
        self.state[R*9+6 : R*9+9] = temp

    def _rotate_D_prime(self):
        for _ in range(3):
            self._rotate_D()

    def _rotate_L(self):
        self._rotate_face_clockwise(L)
        temp = self.state[U*9 : U*9+7:3].copy()
        self.state[U*9 : U*9+7:3] = self.state[B*9+2 : B*9+9:3][::-1]
        self.state[B*9+2 : B*9+9:3] = self.state[D*9 : D*9+7:3][::-1]
        self.state[D*9 : D*9+7:3] = self.state[F*9 : F*9+7:3]
        self.state[F*9 : F*9+7:3] = temp

    def _rotate_L_prime(self):
        for _ in range(3):
            self._rotate_L()

    def _rotate_R(self):
        self._rotate_face_clockwise(R)
        temp = self.state[U*9+2 : U*9+9:3].copy()
        self.state[U*9+2 : U*9+9:3] = self.state[F*9+2 : F*9+9:3]
        self.state[F*9+2 : F*9+9:3] = self.state[D*9+2 : D*9+9:3]
        self.state[D*9+2 : D*9+9:3] = self.state[B*9 : B*9+7:3][::-1]
        self.state[B*9 : B*9+7:3] = temp[::-1]

    def _rotate_R_prime(self):
        for _ in range(3):
            self._rotate_R()

    def _rotate_F(self):
        self._rotate_face_clockwise(F)
        temp = self.state[U*9+6 : U*9+9].copy()
        self.state[U*9+6 : U*9+9] = self.state[L*9+2 : L*9+9:3][::-1]
        self.state[L*9+2 : L*9+9:3] = self.state[D*9 : D*9+3]
        self.state[D*9 : D*9+3] = self.state[R*9 : R*9+7:3][::-1]
        self.state[R*9 : R*9+7:3] = temp

    def _rotate_F_prime(self):
        for _ in range(3):
            self._rotate_F()

    def _rotate_B(self):
        self._rotate_face_clockwise(B)
        temp = self.state[U*9 : U*9+3].copy()
        self.state[U*9 : U*9+3] = self.state[R*9+2 : R*9+9:3]
        self.state[R*9+2 : R*9+9:3] = self.state[D*9+6 : D*9+9][::-1]
        self.state[D*9+6 : D*9+9] = self.state[L*9 : L*9+7:3]
        self.state[L*9 : L*9+7:3] = temp[::-1]

    def _rotate_B_prime(self):
        for _ in range(3):
            self._rotate_B()

    def hash_state(self) -> bytes:
        return self.state.tobytes()

    def __repr__(self) -> str:
        return f"RubikCube(state={self.state.tolist()})"

ALL_MOVES = [
    'U', "U'", 'D', "D'",
    'L', "L'", 'R', "R'",
    'F', "F'", 'B', "B'"
]

def scramble(cube: RubikCube, num_moves: int) -> List[str]:
    moves = []
    for _ in range(num_moves):
        move = np.random.choice(ALL_MOVES)
        cube.apply_move(move)
        moves.append(move)
    return moves
if __name__ == "__main__":
    cube = RubikCube()
    print("Solved state:", cube.get_state())
    cube.apply_move("U")
    cube.apply_move("R")
    print("After U, R:", cube.get_state())
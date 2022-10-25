from state_representation import *
from search import Problem


class EnergyProblem(Problem):
    def __init__(self, initial_state: StateRepresentation, used_actions: List[int], used_heuristic: int):
        self.used_actions   = used_actions
        self.used_heuristic = used_heuristic
        super().__init__(initial_state)

    def actions(self, state: StateRepresentation) -> Generator[Operator, None, None]:
        return state.generate_actions(self.used_actions)

    def result(self, state: StateRepresentation, action: Operator) -> StateRepresentation:
        return state.apply_action(action)

    def value(self, state: StateRepresentation) -> float:
        if self.used_heuristic == GAIN_HEURISTIC:
            return state.gain_heuristic()

        if self.used_heuristic == ENTROPY_HEURISTIC:
            return state.entropy_heuristic()

    def goal_test(self, state: StateRepresentation) -> bool:
        return False
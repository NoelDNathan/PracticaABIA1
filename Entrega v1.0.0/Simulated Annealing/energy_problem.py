from xmlrpc.client import Boolean
from state_representation import *
from search import Problem


class EnergyProblem(Problem):
    def __init__(self, initial_state: StateRepresentation, used_actions: List[int], used_heuristic: int, use_one_action: Boolean = False):
        self.used_actions   = used_actions
        self.used_heuristic = used_heuristic
        self.use_one_action = use_one_action
        super().__init__(initial_state)

    def actions(self, state: StateRepresentation) -> Generator[Operator, None, None]:
        if self.use_one_action:
            return state.generate_one_action()
        
        else:
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
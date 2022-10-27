from math import trunc, log
from typing import List, Generator, Set

from operators import *
from problem_parameters import *

NONPOWERPLANT = -1

GAIN_HEURISTIC = 0
ENTROPY_HEURISTIC = 1
COMBINED_HEURISTIC = 2
FIX_STATE_HEURISTIC = 3

MOVE_CLIENT = 0
SWAP_CLIENTS = 1
TURN_ON_POWER_PLANT = 2
SUPER_MOVE_CLIENT = 3


class StateRepresentation(object):

    def __init__(self,
                 params: ProblemParameters,
                 c_pp: List[int],
                 remain: List[float],
                 consum: List[List[float]],
                 gain: float,
                 prices: List[List[float]],
                 last_action=None,
                 count_actions=[0, 0, 0, 0, 0],
                 misplaced_clients=0) -> None:

        self.params = params
        self.c_pp = c_pp
        self.remain = remain
        self.consum = consum
        self.prices = prices
        self.gain = gain
        self.last_action = last_action
        self.count_actions = count_actions
        self.misplaced_clients = misplaced_clients

    def copy(self):
        return StateRepresentation(self.params,
                                   self.c_pp.copy(),
                                   self.remain.copy(),
                                   self.consum,
                                   self.gain,
                                   self.prices,
                                   self.last_action,
                                   self.count_actions.copy(),
                                   self.misplaced_clients
                                   )

    def served_clients(self) -> int:
        s_c = 0
        for i in self.c_pp:
            if i != NONPOWERPLANT:
                s_c += 1
        return s_c

    def __repr__(self) -> str:
        return f"Served clients = {self.served_clients()} of {len(self.c_pp)} \nGains: {self.gain_heuristic()} \n"

    def generate_actions(self, used_actions) -> Generator[Operator, None, None]:
        num_c = len(self.params.clients_vector)
        num_pp = len(self.params.power_plants_vector)

        if isinstance(self.last_action, MoveClient):
            self.count_actions[MOVE_CLIENT] += 1
        elif isinstance(self.last_action, SwapClients):
            self.count_actions[SWAP_CLIENTS] += 1
        elif isinstance(self.last_action, TurnOnPowerPlant):
            self.count_actions[TURN_ON_POWER_PLANT] += 1
        elif isinstance(self.last_action, SuperMoveClient):
            self.count_actions[SUPER_MOVE_CLIENT] += 1

        for id_pp in range(num_pp):
            if self.remain[id_pp] == self.params.power_plants_vector[id_pp].Produccion:
                if TURN_ON_POWER_PLANT in used_actions:
                    yield TurnOnPowerPlant(id_pp)

            else:
                if MOVE_CLIENT in used_actions:
                    for id_c in range(num_c):
                        if id_pp == self.c_pp[id_c]:
                            continue

                        c_consum = self.consum[id_pp][id_c]
                        if c_consum < self.remain[id_pp]:
                            yield MoveClient(id_c, id_pp)

        if SWAP_CLIENTS in used_actions:
            for id_c1 in range(num_c):
                if id_c1 != num_c - 1:
                    for id_c2 in range(id_c1 + 1, num_c):
                        id_pp1 = self.c_pp[id_c1]
                        id_pp2 = self.c_pp[id_c2]

                        if id_pp1 == id_pp2 or id_pp1 == NONPOWERPLANT or id_pp2 == NONPOWERPLANT:
                            continue

                        consum_c1_pp1 = self.consum[id_pp1][id_c1]
                        consum_c1_pp2 = self.consum[id_pp2][id_c1]
                        consum_c2_pp1 = self.consum[id_pp1][id_c2]
                        consum_c2_pp2 = self.consum[id_pp2][id_c2]

                        remain_pp1 = self.remain[id_pp1]
                        remain_pp2 = self.remain[id_pp2]

                        if consum_c2_pp1 - consum_c1_pp1 < remain_pp1 and consum_c1_pp2 - consum_c2_pp2 < remain_pp2:
                            yield SwapClients(id_c1, id_c2)

        if SUPER_MOVE_CLIENT in used_actions:
            for id_c in range(num_c):
                for id_pp in range(num_pp):
                    if id_pp == self.c_pp[id_c]:
                        continue

                    c_consum = self.consum[id_pp][id_c]
                    if c_consum < self.remain[id_pp]:
                        yield SuperMoveClient(id_c, id_pp)

    def generate_one_action(self, used_actions) -> Generator[Operator, None, None]:
        move_client_combinations = set()
        swap_client_combinations = set()
        super_move_client_combinations = set()

        numClients = len(self.params.clients_vector)
        numCentrals = len(self.params.power_plants_vector)

        for id_client1 in range(numClients):

            # MoveClient
            if MOVE_CLIENT in used_actions:
                for id_PwP in range(numCentrals):
                    if self.c_pp[id_client1] == id_PwP:
                        continue

                    csm_client = self.consum[id_PwP][id_client1]

                    if csm_client < self.remain[id_PwP]:
                        move_client_combinations.add((id_client1, id_PwP))

            # Super Move Client
            if SUPER_MOVE_CLIENT in used_actions:
                for id_c in range(numClients):
                    for id_pp in range(numCentrals):
                        if id_pp == self.c_pp[id_c]:
                            continue

                        c_consum = self.consum[id_pp][id_c]
                        if c_consum < self.remain[id_pp]:
                            super_move_client_combinations.add((id_c, id_pp))

        if MOVE_CLIENT in used_actions:
            combination = random.choice(list(move_client_combinations))
            yield MoveClient(combination[0], combination[1])  
        elif SUPER_MOVE_CLIENT in used_actions:  
            combination = random.choice(list(super_move_client_combinations))
            yield SuperMoveClient(combination[0], combination[1])

    def apply_action(self, action: Operator):
        new_state = self.copy()
        new_state.last_action = action

        def apply_move_client(new_state, action):

            id_c = action.id_client
            id_pp1 = self.c_pp[id_c]
            pp1 = self.params.power_plants_vector[id_pp1]
            id_pp2 = action.id_destination_PwP

            new_state.c_pp[id_c] = id_pp2

            if id_pp1 != NONPOWERPLANT:
                new_state.remain[id_pp1] += self.consum[id_pp1][id_c]

                if new_state.remain[id_pp1] == pp1.Produccion:
                    new_state.gain += self.prices[1][id_pp1] - \
                        self.prices[0][id_pp1]
            else:
                new_state.gain += self.prices[2][id_c]

            new_state.remain[id_pp2] -= self.consum[id_pp2][id_c]
            return new_state

        def apply_turn_on_power_plant(new_state, id_pp):
            new_state.gain -= self.prices[1][id_pp] + \
                self.prices[0][id_pp]
            return new_state

        if isinstance(action, MoveClient):
            new_state = apply_move_client(new_state, action)

        elif isinstance(action, SwapClients):
            id_c1 = action.id_client1
            id_c2 = action.id_client2
            id_pp1 = self.c_pp[id_c1]
            id_pp2 = self.c_pp[id_c2]

            new_state.c_pp[id_c1] = id_pp2
            new_state.c_pp[id_c2] = id_pp1

            new_state.remain[id_pp1] += self.consum[id_pp1][id_c1] - \
                self.consum[id_pp1][id_c2]
            new_state.remain[id_pp2] += self.consum[id_pp2][id_c2] - \
                self.consum[id_pp2][id_c1]

        elif isinstance(action, TurnOnPowerPlant):
            new_state = apply_turn_on_power_plant(new_state, action.id_pp)

        elif isinstance(action, SuperMoveClient):
            id_c = action.id_client
            client = self.params.clients_vector[id_c]
            if self.c_pp[id_c] == NONPOWERPLANT:
                if client.Contrato == GARANTIZADO:
                    new_state.gain += client.Consumo * \
                        VEnergia.tarifa_cliente_garantizada(client.Tipo)
                    new_state.misplaced_clients -= 1
                else:
                    new_state.gain += self.prices[2][id_c]
            new_state = apply_move_client(new_state, action)
            id_pp = action.id_destination_PwP
            if self.remain[id_pp] == self.params.power_plants_vector[id_pp].Produccion:
                new_state = apply_turn_on_power_plant(new_state, id_pp)

        return new_state

    def gain_heuristic(self) -> float:
        return self.gain

    def entropy_heuristic(self) -> float:
        total_entropy = 0
        for id, remain in enumerate(self.remain):
            max_prod = self.params.power_plants_vector[id].Produccion
            occupancy = 1 - (remain/max_prod)
            if occupancy > 0:
                total_entropy += - (occupancy * log(occupancy))
        return -total_entropy

    def combined_heuristic(self) -> float:
        return self.gain_heuristic() + self.entropy_heuristic()

    def fix_state_heuristic(self) -> float:
        k = 7000
        return -self.misplaced_clients * k + self.gain_heuristic()

from symbol import power
from typing import List, Generator, Set
from abia_energia import *

# Problem params


class ProblemParameters():
    def __init__(self, clients_vector: Clientes, power_plants_vector: Centrales) -> None:
        self.clients_vector = clients_vector
        self.power_plants_vector = power_plants_vector
# Operators


class Operator():
    pass


class MoveClient(Operator):
    def __init__(self, client, destination_PwP):
        self.client = client
        self.destination_PwP = destination_PwP

    def __repr__(self) -> str:
        return f"Client {self.client} has been moved to power plant {self.destination_PwP}"


class SwapClient(Operator):
    def __init__(self, client1, client2):
        self.client1 = client1
        self.client2 = client2

    def __repr__(self) -> str:
        return f"Swap between Client {self.client1} and Client {self.client2}"


class remove_non_guaranteed_client(Operator):
    def __init__(self, p_i: int, p_j: int):
        pass

    def __repr__(self) -> str:
        pass

# Generators initial states


def generate_simple_initial_state(params: ProblemParameters):
    power_plant_vector = params.power_plants_vector.copy()
    current_power_plant = power_plant_vector.pop()
    for client in params.clients_vector:
        if client.Contrato == NOGARANTIZADO:
            continue
        consum = client.consumption2PowerPlant(current_power_plant)
        while True:
            remain = current_power_plant.remain_energy

            if consum < remain:
                client.power_plant = current_power_plant
                current_power_plant.remain_energy -= consum
                break

            current_power_plant = power_plant_vector.pop()


def generate_complex_initial_state():
    pass


class StateRepresentation():
    def __init__(self, params: ProblemParameters, client_assignation, remaining_energy):
        self.params = params
        self.client_assignation = client_assignation
        self.remaining_energy = remaining_energy

    def copy(self):
        pass

    def __repr__(self) -> str:
        pass

    def generate_actions(self) -> Generator[Operator, None, None]:
        for id_client1, client1 in enumerate(self.params.clients_vector):
            for id_power_plant, power_plant in enumerate(self.params.power_plants_vector):

                # Podríamos ahorrarnos calcular esto?
                consum_client = client1.consumption2PowerPlant(power_plant)

                if consum_client < power_plant.remaining_energy:
                    if client1.power_plant is not power_plant:
                        yield MoveClient(id_client1, id_power_plant)

            for id_client2, client2 in enumerate(self.params.clients_vector):
                if id_client1 == id_client2:
                    continue

                # Podríamos ahorrarnos calcular esto?
                consum_client1 = client1.consumption2PowerPlant()

                consum_client2 = client2.consumption2PowerPlant()

                PwP1 = client1.power_plant
                PwP2 = client2.power_plant

                if PwP1 is PwP2:
                    continue

                if (PwP1.remaining_energy - consum_client1 + consum_client2 < PwP1.Produccion
                        and PwP2.remaining_energy - consum_client2 + consum_client1 < PwP2.Produccion):
                    yield SwapClient(id_client1, id_client2)

    def apply_actions(self, action: Operator) -> StateRepresentation:
        new_state = self.copy()

        if isintance(action, MoveClient):
            client = action.client

            PwP_1 = client.power_plant
            PwP_2 = action.destination_PwP

            client.power_plant.remaining_energy += client.consumption2PowerPlant()
            new_state.destination_PwP.remaining_energy -= client.consumption2PowerPlant()

        elif isintance(action, SwapClient):
            client1 = action.client1
            client2 = action.client2

        return new_state

    def heuristic():
        pass

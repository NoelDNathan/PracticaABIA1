from typing import List, Generator, Set
from abia_energia import *


# Our Methods

def distance(obj1, obj2):
    x = obj1.CoordX - obj2.CoordX
    y = obj1.CoordY - obj2.CoordY
    return (x**2 + y ** 2)**(1/2)


def electry_supplied_to_client(client: Cliente, power_plant: Central) -> float:
    dist = distance(client, power_plant)
    return client.Consumo * (1 + VEnergia.loss(dist))
    # return client.Consumo / (1 - VEnergia.loss(dist))

# Problem params


class ProblemParameters(object):
    def __init__(self, clients_vector: Clientes, power_plants_vector: Centrales) -> None:
        self.clients_vector = clients_vector
        self.power_plants_vector = power_plants_vector
# Operators


class Operator(object):
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


class RemoveNGClient(Operator):
    def __init__(self, p_i: int, p_j: int):
        pass

    def __repr__(self) -> str:
        pass

# Generators initial states


def generate_simple_initial_state(params: ProblemParameters) -> StateRepresentation:
    remaining_energies = list()
    real_consumption = list()
    client_power_plant = list()

    ids_power_plants = list()

    for id_PwP, power_plant in enumerate(params.power_plants_vector):
        ids_power_plants.append(id_PwP)
        remaining_energies.append(power_plant.Produccion)
        row = list()

        for client in params.clients_vector:
            consumtion = electry_supplied_to_client(client, power_plant)
            row.append(consumtion)
        real_consumption.append(row)

    for id_client, client in enumerate(params.clients_vector):
        if client.Contrato == NOGARANTIZADO:
            continue

        id_PwP = ids_power_plants.pop()
        consum = real_consumption[id_PwP][id_client]
        while True:
            if consum < remaining_energies[id_PwP]:
                client_power_plant.append(id_PwP)
                remaining_energies[id_PwP] -= consum
                break

            id_PwP = ids_power_plants.pop()
        # Que pasaria si no hubieran más PwP

    return StateRepresentation(params, client_power_plant, remaining_energies, real_consumption)


def generate_complex_initial_state():
    pass


class StateRepresentation():
    def __init__(self, params: ProblemParameters, client_power_plant: List[int], remaining_energies: List[float], real_consumption: List[List[float]]):
        self.params = params
        self.client_power_plant = client_power_plant
        self.remaining_energies = remaining_energies
        self.real_consumption = real_consumption

    def copy(self):
        return StateRepresentation(self.params, self.client_power_plant.copy(), self.remaining_energies.copy())

    def __repr__(self) -> str:
        pass

    def generate_actions(self) -> Generator[Operator, None, None]:
        for id_client1, client1 in enumerate(self.params.clients_vector):
            for id_PwP in range(self.params.power_plants_vector):
                if self.client_power_plant[id_client1] == id_PwP:
                    continue

                csm_client = self.real_consumption[id_PwP][id_client1]

                if csm_client < self.remaining_energies[id_PwP]:
                    yield MoveClient(id_client1, id_PwP)

            for id_client2, client2 in enumerate(self.params.clients_vector):
                if id_client1 == id_client2:
                    continue

                id_PwP1 = self.client_power_plant[client1]
                id_PwP2 = self.client_power_plant[client2]

                if id_PwP1 == id_PwP2:
                    continue

                csm_pwp1_cli1 = self.real_consumption[id_PwP1][id_client1]
                csm_pwp2_cli1 = self.real_consumption[id_PwP2][id_client1]

                csm_pwp1_cli2 = self.real_consumption[id_PwP1][id_client2]
                csm_pwp2_cli2 = self.real_consumption[id_PwP2][id_client2]

                remain1 = self.remaining_energies[id_PwP1]
                remain2 = self.remaining_energies[id_PwP2]

                if (csm_pwp1_cli2 - csm_pwp1_cli1 < remain1 and csm_pwp2_cli1 - csm_pwp2_cli2 < remain2):
                    yield SwapClient(id_client1, id_client2)

    def apply_actions(self, action: Operator) -> StateRepresentation:
        new_state = self.copy()

        if isinstance(action, MoveClient):
            client = action.client

            PwP_1 = client.power_plant
            PwP_2 = action.destination_PwP

        elif isinstance(action, SwapClient):
            client1 = action.client1
            client2 = action.client2

        return new_state

    def heuristic():
        # HOLA
        pass

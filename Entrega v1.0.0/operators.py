class Operator(object):
    pass


class MoveClient(Operator):
    def __init__(self, id_client, id_destination_PwP):
        self.id_client = id_client
        self.id_destination_PwP = id_destination_PwP

    def __repr__(self) -> str:
        return f"Client {self.id_client} has been moved to power plant {self.id_destination_PwP}"


class SwapClients(Operator):
    def __init__(self, id_client1, id_client2):
        self.id_client1 = id_client1
        self.id_client2 = id_client2

    def __repr__(self) -> str:
        return f"Swap between Client {self.id_client1} and Client {self.id_client2}"


class TurnOnPowerPlant(Operator):
    def __init__(self, id_pp: int):
        self.id_pp = id_pp

    def __repr__(self) -> str:
        return f"Power Plant {self.id_pp} has been turned on"


class SuperMoveClient(Operator):
    def __init__(self, id_client, id_destination_PwP):
        self.id_client = id_client
        self.id_destination_PwP = id_destination_PwP

    def __repr__(self) -> str:
        return f"Client {self.id_client} has been moved to power plant {self.id_destination_PwP}"

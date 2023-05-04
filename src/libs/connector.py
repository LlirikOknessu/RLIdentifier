import socket
import struct
from numpy import random
import datetime
import logging

logger = logging.getLogger()


class AbstractConnector:
    def __init__(self, address: tuple[str, int]):
        print('Start')
        logger.info(f"Initialize connector at address: {address}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(address)
        self.socket.listen(1)
        self.connection, self.client_address = self.socket.accept()
        logger.info(f"Connector is initialized at: {address}\n"
                    f"Client address: {self.client_address}")

    def receive(self):
        pass

    def step(self, k_predict, t_predict):
        pass

    def close(self):
        pass

    def reset_simulation(self, simulation_transfer: int):
        pass


class IdentificationConnectorConnector(AbstractConnector):
    def __init__(self, address: tuple[str, int]):
        super().__init__(address)
        self.bytes_to_receive = 8
        self.structure_to_unpack = 'ff'

    def receive(self):
        data = None
        try:
            data = self.connection.recv(self.bytes_to_receive)
        except Exception:
            self.connection.close()
        x_true, y_true, k_target, t_target = struct.unpack(self.structure_to_unpack, data)
        state = [x_true, y_true, k_target, t_target]
        return state

    def step(self, k_predict, t_predict):
        # print('send ', float(action))
        try:
            # print(action)
            self.connection.send(struct.pack('ff', float(action), float(flag), float(y_target)))
        except Exception:
            self.connection.close()

    def close(self):
        self.connection.close()

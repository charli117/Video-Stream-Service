import pyaudio
import logging

class Microphone:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.logger = logging.getLogger('Microphone')

    def start(self, device_index=None):
        try:
            self.logger.info("Starting microphone...")  # Pab50
            if device_index is None:
                device_index = self.get_default_device_index()

            self.stream = self.audio.open(format=pyaudio.paInt16,
                                          channels=1,
                                          rate=44100,
                                          input=True,
                                          input_device_index=device_index,
                                          frames_per_buffer=1024)
            self.logger.info("Microphone started successfully.")  # Pc632
        except Exception as e:
            self.logger.error(f"Failed to start microphone: {str(e)}")  # Pd0d2
            raise

    def read(self):
        if self.stream is not None:
            return self.stream.read(1024)
        return None

    def release(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def get_default_device_index(self):
        return self.audio.get_default_input_device_info()['index']

    @staticmethod
    def list_devices():
        audio = pyaudio.PyAudio()
        device_count = audio.get_device_count()
        devices = []
        for i in range(device_count):
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': device_info['name']
                })
        audio.terminate()
        return devices

    @classmethod
    def update_device_names(cls, device_names):
        cls._device_names = device_names

    @classmethod
    def get_device_name(cls, index):
        return cls._device_names.get(str(index), f'Microphone {index}')

import ms5837
import time
import numpy as np


class Bar30:
    def __init__(self):
        # We must initialize the sensor before reading it
        sensor = ms5837.MS5837_30BA() # Default I2C bus is 1 (Raspberry Pi 3)
        sensor.setFluidDensity(ms5837.DENSITY_SALTWATER) # Use predefined saltwater density
        self.ret = sensor.init()
        self.output = [0,0]
    def read(self) -> np.ndarray:
           if self.ret:                
                if self.sensor.read():
                      # Get the most recent depth measurement in meters.
                    d = self.sensor.depth() 
                    # Default is degrees C (no arguments)
                    t = self.sensor.temperature()
                    self.output = [d, t]
                    
                    return self.output


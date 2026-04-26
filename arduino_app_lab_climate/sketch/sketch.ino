// SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
//
// SPDX-License-Identifier: MPL-2.0

#include <Arduino_Modulino.h>
#include <Arduino_RouterBridge.h>

// Create object instance
ModulinoThermo thermo;

unsigned long previousMillis = 0; 	// Stores last time values were updated
const long interval = 1000; 		// Every second

void setup() {
  Bridge.begin();

  // Initialize Modulino I2C communication
  Modulino.begin(Wire1);
  // Detect and connect to temperature/humidity sensor module
  thermo.begin();
}

void loop() {
  unsigned long currentMillis = millis(); // Get the current time
  if (currentMillis - previousMillis >= interval) {
    // Save the last time you updated the values
    previousMillis = currentMillis;

    // Read temperature and humidity
    float celsius = thermo.getTemperature();
    float humidity = thermo.getHumidity();
    
    // Read light level from analog pin A0 (Standard on UNO Q for the light sensor)
    float light = (float)analogRead(A0);

    // Send all 3 values to Python
    Bridge.notify("record_sensor_samples", celsius, humidity, light);
  }
}

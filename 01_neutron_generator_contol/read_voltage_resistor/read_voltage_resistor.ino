// ***********************************
// *** Reading voltage over resistor behind resistor circuit 
// ***********************************
int sensorValue = analogRead(A0);
float voltage = 0.0;


const int numReadings = 30;
int readings[numReadings];      // the readings from the analog input
int readIndex = 0;              // the index of the current reading
int total = 0;                  // the running total
float average = 0;              // the average
// ***********************************
// *** Time
// ***********************************
unsigned long previousMillis = 0; // previous time
const long interval = 1000;           // interval at which to execute

void setup()
{
  Serial.begin (9600);
}

// reads the analog inputs numReadings1 times and returns the average
void readAnalog(){
  // subtract the last reading:
  total = total - readings[readIndex];

  // read from the sensor:
  readings[readIndex] = analogRead(sensorValue);

  // add the reading to the total:
  total = total + readings[readIndex];

  // advance to the next position in the array:
  readIndex = readIndex + 1;

  // if we're at the end of the array...
  if (readIndex >= numReadings) {
    // ...wrap around to the beginning:
    readIndex = 0;
  }

  // calculate the average:
  average = total / numReadings;

  // send it to the computer as ASCII digits
  voltage = average * (5.0 / 1023.0); // uncalibrated

  delay(5);        // delay in between reads for stability  
}

void loop()
{
  readAnalog();
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    // save the last time executed
    Serial.print("Voltage: ");  // voltage
    Serial.println(voltage);
    previousMillis = currentMillis;
    // print: V1 V_out
    delay(50);
  }
}

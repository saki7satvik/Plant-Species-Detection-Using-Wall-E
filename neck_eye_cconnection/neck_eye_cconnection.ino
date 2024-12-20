#include <Servo.h>

// Define servo objects
Servo a;  // Eye servo 1
Servo b;  // Eye servo 2
Servo c;  // Neck servo top
Servo d;  // Neck servo middle
Servo e;  // Neck servo bottom
Servo f;  // Right hand servo
Servo g;  // Left hand servo

// Pin assignments for servos
const int aPin = 13;
const int bPin = 12;
const int cPin = 11;
const int dPin = 10;
const int ePin = 9;
const int fPin = 8;
const int gPin = 7;

void setup() {
  // Start serial communication
  Serial.begin(9600);
  
  // Attach servos to their respective pins
  a.attach(aPin);
  b.attach(bPin);
  c.attach(cPin); // limit - 0(right), 90(center), 180(left)
  d.attach(dPin);
  e.attach(ePin);
  f.attach(fPin);
  g.attach(gPin);

  // Set initial positions for all servos
  a.write(0);  // Eye servo 1 to 0 degrees
  b.write(0);  // Eye servo 2 to 0 degrees
  c.write(90);  // Neck top servo to 0 degrees
  d.write(0);  // Neck middle servo to 0 degrees
  e.write(0);  // Neck bottom servo to 0 degrees
  f.write(0);  // Right hand servo to 0 degrees
  g.write(0);  // Left hand servo to 0 degrees

  Serial.println("Servos initialized and ready.");
}

void loop() {
  // Check if data is available on the serial port
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // Read the incoming command string

    // Check and execute commands
    if (input.startsWith("init")) {
      initialize_all();  // Initialize all servos with a test sequence
      Serial.println("Initialization complete.");
    } 
    else if (input.startsWith("look_up")) {
      look_up();  // Move servos to look up
      Serial.println("Wall-E looking up.");
    } 
    else if (input.startsWith("look_down")) {
      look_down();  // Move servos to look down
      Serial.println("Wall-E looking down.");
    } 
    else if (input.startsWith("turn_left")) {
      look_left();  // Move servos to look left
      Serial.println("Wall-E looking left.");
    } 
    else if (input.startsWith("turn_right")) {
      look_right();  // Move servos to look right
      Serial.println("Wall-E looking right.");
    } 
    else if (input.startsWith("go_left")) {
      go_left();  // Move Wall-E left
      Serial.println("Wall-E moving left.");
    } 
    else if (input.startsWith("go_right")) {
      go_right();  // Move Wall-E right
      Serial.println("Wall-E moving right.");
    } 
    else if (input.startsWith("go_front")) {
      go_front();  // Move Wall-E forward
      Serial.println("Wall-E moving forward.");
    } 
    else if (input.startsWith("go_back")) {
      go_back();  // Move Wall-E backward
      Serial.println("Wall-E moving backward.");
    }
    else if (input.startsWith("dance")) {
      while (input.startsWith("stop")){
        dance();
      }
    }
  }
}

// Function to initialize all servos with a test movement sequence
void initialize_all() {
  c.write(135);  // Move neck top servo to 135 degrees
  d.write(135);  // Move neck middle servo to 135 degrees
  delay(500);    // Wait for 500 ms
  c.write(45);   // Move neck top servo to 45 degrees
  d.write(45);   // Move neck middle servo to 45 degrees
  delay(500);    // Wait for 500 ms
  c.write(90);   // Move neck top servo to neutral 90 degrees
  d.write(90);   // Move neck middle servo to neutral 90 degrees
  delay(500);    // Wait for 500 ms
}

// Function to make Wall-E look up
void look_up() {
  d.write(60);  // Move neck top servo upward
  e.write(120);
}

// Function to make Wall-E look down
void look_down() {
  d.write(150);  // Move neck top servo downward
  e.write(30);
}

// Function to make Wall-E look left
void look_left() {
  c.write(150);  // Move neck middle servo to the left
}

// Function to make Wall-E look right
void look_right() {
  c.write(30);  // Move neck middle servo to the right
}

// Function to make Wall-E move left
void go_left() {
  f.write(45);  // Adjust right hand servo to move left
  g.write(135); // Adjust left hand servo to move left
}

// Function to make Wall-E move right
void go_right() {
  f.write(135); // Adjust right hand servo to move right
  g.write(45);  // Adjust left hand servo to move right
}

// Function to make Wall-E move forward
void go_front() {
  f.write(90);  // Both hand servos in a neutral forward-moving position
  g.write(90);  
}

// Function to make Wall-E move backward
void go_back() {
  f.write(180); // Adjust right hand servo for backward movement
  g.write(0);   // Adjust left hand servo for backward movement
}

void dance(){
  d.write(30);
  e.write(30);
  f.write(45);
  g.write(135);
  delay(500);
  d.write(150);
  e.write(150);
  f.write(135);
  g.write(45);
  delay(500);
  // move front an go back

}

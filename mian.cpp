
#include <Arduino.h>

#define RXD2 16  // 定义Serial2的RX引脚
#define TXD2 17  // 定义Serial2的TX引脚

void setup() {
  Serial.begin(115200);  // 初始化主串口用于调试，波特率115200
  Serial2.begin(9600, SERIAL_8N1, RXD2, TXD2);  // 初始化Serial2用于接收树莓派数据
  Serial.println("ESP32 ready, waiting for fall detection...");
}

void loop() {
  if (Serial2.available()) {
    String message = Serial2.readStringUntil('\n');
    Serial.println("Received: " + message);  // 通过主串口输出接收到的消息
    if (message == "FALL") {
      Serial.println("Fall detected! Triggering alert...");
      // 这里可以添加警报逻辑，例如点亮LED或蜂鸣器
    }
  }
  delay(100);  // 小延迟避免CPU占用
}



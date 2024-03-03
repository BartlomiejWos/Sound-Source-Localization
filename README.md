# Microphone Array Speech Source Localization using SRP-PHAT and MUSIC

## The purpose of this project was to implement sound source direction estimation algorithm using Raspberry PI microphone array.

Parts of the project shows how do the following:
* Data parser for microphone array and stream reader 
* VAD - Voice Activity Detector
* LPC - Linear Predicting Coding (envelope spectrum analysis and detection of main frequencies)
* MUSIC - Multiple Signal Classification algorithm for signal direction finding 
* SRP-PHAT - Steered-response power with phase transform algorithm for signal direction finding 

## Block Diagram:
![alg](https://github.com/BartlomiejWos/Sound-Source-Localization/assets/161388878/02ef84a9-a495-4e54-8a9f-d8eb0f695951)

## VAD results:
![VAD](https://github.com/BartlomiejWos/Sound-Source-Localization/assets/161388878/9c1f4510-7404-469a-b3a9-3c26a394e2b3)

## Simulation results based on Room Impulse Response (RIR) generator:
![sim_results](https://github.com/BartlomiejWos/Sound-Source-Localization/assets/161388878/75a05712-d4cb-4a5f-a9e4-87b14c5724d6)

## Real-Time Performance:
![real_time_res](https://github.com/BartlomiejWos/Sound-Source-Localization/assets/161388878/7e02112d-07ad-4221-b39f-459ca1161e2a)

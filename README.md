# Mechanical Properties Prediction

### Problem Statement
•	Currently there are no precise theoretical methods to predict mechanical properties of steels.<br>
•	All the methods available are by backed by statistics and extensive physical testing of the materials. <br>
•	Since testing each material with different composition is a highly tedious task this project tries to predict the mechanical properties of the alloys.<br>
•	The dataset had weight percentages of alloying metals (like Aluminium, copper, manganese, nitrogen, nickel, cobalt, carbon, etc) and the temperature (in Celsius) for each test as features and mechanical properties (tensile strength, yield strength, elongation and reduction in area) as output variable.<br>

### Solution Proposed 
This project is a perfect candidate for machine learning as there are to established formulas to calculate material properties according to their composition. In this project we are trying to predict the mechanical properties of alloy materials according to their composition and temperature using regression algorithms. this will in turn reduce the cost of casting and testing the various alloy materials 

## Tech Stack Used
1. Python 
2. Flask
3. Machine learning algorithms
4. Docker
5. MongoDB

## Infrastructure Required.

1. AWS S3
2. AWS EC2
3. AWS ECR
4. Git Actions

## How to run?
Before we run the project, make sure that you are having MongoDB in your local system, with Compass since we are using MongoDB for data storage. You also need AWS account to access the service like S3, ECR and EC2 instances.

## Training pipeline
![training pipeline](https://github.com/siddhantbhagat/ML_pipeline/assets/41586492/1dfaf4f9-ff97-4532-971b-0b63102f75d2)


## Project Architecture
![image](https://user-images.githubusercontent.com/57321948/193536768-ae704adc-32d9-4c6c-b234-79c152f756c5.png)


## Deployment Architecture
![image](https://user-images.githubusercontent.com/57321948/193536973-4530fe7d-5509-4609-bfd2-cd702fc82423.png)


### Step 1: Clone the repository
```bash
[git clone https://github.com/sethusaim/Sensor-Fault-Detection.git](https://github.com/siddhantbhagat/ML_pipeline.git)
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -n properties python=3.7 -y
```

```bash
conda activate properties
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Run the application server
```bash
python app.py
```

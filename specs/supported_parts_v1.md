# Supported Parts – Version 1 (Aligned with Label Studio)

This document defines the **official list of vehicle parts** supported in Phase 1 of the damage detection pipeline.

These parts must be annotated using **RectangleLabels** in Label Studio.
Every annotated region must have:
- One **part**
- One **damage type**
- One **severity level**

## 🚗 Supported Parts (V1)

- front bumper  
- rear bumper  
- hood  
- trunk  
- windshield  
- front left door  
- front right door  
- rear left door  
- rear right door  
- left fender 
- right fender
- rear left fender
- rear right fender
- headlights  
- taillights  

## 🛑 Removed parts for V1 (not annotated)
To reduce complexity and stay aligned with V1 requirements:
- wheel/rim  
- side door (unspecified)

These can be added in V2 or V1.1 if needed.

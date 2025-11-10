# Supported Views – Version 1

This document defines the set of camera views (angles) that the pipeline should recognize from the input photos of a damaged vehicle. These views are used later to (1) detect missing photos and (2) give evidence for damaged parts.

## Views

1. **front**  
   Photo taken from the front of the car, roughly centered. The front bumper, grille, headlights, and windshield base should be visible.

2. **rear**  
   Photo taken from the rear of the car, roughly centered. The rear bumper, trunk/tailgate, and taillights should be visible.

3. **left**  
   Photo taken from the left side of the car, showing the left doors, left fender, and part of the front/rear.

4. **right**  
   Photo taken from the right side of the car, showing the right doors, right fender, and part of the front/rear.

5. **front-left**  
   3/4 view taken from the front-left corner. We should see the front bumper + left fender + part of the left door. Useful when the front-left area is impacted.

6. **front-right**  
   3/4 view taken from the front-right corner. We should see the front bumper + right fender + part of the right door. Useful when the front-right area is impacted.

7. **close-up**
    zoomed in photos about the damages/
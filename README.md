# Lucas_Kanade_Tracker    

## For running LK tracker only
First, change the directory to Code:  
```bash
cd Code  
```
and run: 
```bash
python3 project4.py  
```
Ensure that line 104 to 106 look like:
```bash
dataset = "Bolt2"
#dataset = "Car4"
# dataset = "DragonBaby"
```
You'll see a video of usain bolt running and being detected using the Lucas-Kanade tracker.  


To see car's video, just comment line 104 and uncomment line 105 of project4.py so it looks like:  
```bash
#dataset = "Bolt2"
dataset = "Car4"
# dataset = "DragonBaby"
```

Running the code again will output car's video with the car being detected using the Lucas-Kanade tracker.  

To see dragon baby's video, just comment line 104, 105 and uncomment line 106 of project4.py so it looks like:   
```bash
#dataset = "Bolt2"
#dataset = "Car4"
dataset = "DragonBaby"   
```
Running the code again will output dragon baby's video with the car being detected using the Lucas-Kanade tracker.  

## For running robust LK tracker
First, change the directory to Code:  
```bash
cd Code  
```
and run: 
```bash
python3 project4_robust.py  
```
Ensure that line 127 to 129 look like:
```bash
dataset = "Bolt2"
#dataset = "Car4"
# dataset = "DragonBaby"
```
You'll see a video of usain bolt running and being detected using the robust Lucas-Kanade tracker.  


To see car's video, just comment line 127 and uncomment line 128 of project4.py so it looks like:  
```bash
#dataset = "Bolt2"
dataset = "Car4"
# dataset = "DragonBaby"
```

Running the code again will output car's video with the car being detected using the robust Lucas-Kanade tracker.  

To see dragon baby's video, just comment line 127,128 and uncomment line 129 of project4.py so it looks like:   
```bash
#dataset = "Bolt2"
#dataset = "Car4"
dataset = "DragonBaby"   
```
Running the code again will output dragon baby's video with the car being detected using the robust Lucas-Kanade tracker.  


 
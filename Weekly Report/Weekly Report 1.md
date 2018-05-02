## Weekly Report

> Zhen Zhao
>
> May 2nd, 2018

### Conclussion of Past Week

Have mainly been working on the simulator development

- [x] analyze the old code by HUANG, generate the **funciton scheduling** graph
- [x] review the popular GUI solution in python. Eventually, still stick to **tkinter**
- [x] read the **Doc** of SPAS, to summary more about the constraints (`still under working`)
- [x] design the project structure (detail avaible at [github_wiki_design](https://github.com/ZhenZHAO/EAVNSIM/wiki/Roughly-Design))
- [x] re-coding subfucntions into the following two series of functions
      
      - transformation related
        - [trans_unit.py](https://github.com/ZhenZHAO/EAVNSIM/blob/master/trans_unit.py)
        - [trans_time.py](https://github.com/ZhenZHAO/EAVNSIM/blob/master/trans_time.py)
        - [trans_coordinate.py](https://github.com/ZhenZHAO/EAVNSIM/blob/master/trans_coordinate.py)
      
      - various model formulation
        - [model_satellite.py](https://github.com/ZhenZHAO/EAVNSIM/blob/master/model_satellite.py)
        - [model_effect.py](https://github.com/ZhenZHAO/EAVNSIM/blob/master/model_effect.py)
        - [model_obs_ability.py](https://github.com/ZhenZHAO/EAVNSIM/blob/master/model_obs_ability.py)

PS: Things are always harder than it looks at first...

### What To Do 

Intent to be away for my wedding ceremony. Surely, keep working on the developement,

- [ ] To read the source code of [APSYNSIM](https://launchpad.net/apsynsim) , extract the following info
      
      - Strechable GUI
      - Real-time refreshing module
      - Various plot modules
- [ ] To finish the telescope visibility functions
- [ ] To finish the designs of SQLite Database module

- The **constraints** is not well implemented in the OLD VERSION, which is still the most challenging part!



### To Track my development process

- The developement is currently hosted on [My Github](https://github.com/ZhenZHAO/EAVNSIM)

- Development plan, available at [My Github_Project_Plan](https://github.com/ZhenZHAO/EAVNSIM/projects/4) 
- Also, a wiki is maintained at [My Project Wiki](https://github.com/ZhenZHAO/EAVNSIM/wiki) 

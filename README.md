 ---Folder objective 1:
 
 objective 1 have some basic contraint and include total work hours of all employees in the aim function
 The maximum function is maximize(10*total_number_of_assignments-0.1*total_work)
 
  .workforce_Python API.py is written in python using docplex.mp
  
  .https://www.ibm.com/docs/en/icos/20.1.0?topic=cplex-setting-up-python-api(setup the cplex python API)
  
  .workforce_google_colab.ipynb is the file written in google colab. we do not use it beacuse the it has operation size limitation.
 
 
 ---Folder objective 2:
 
 objective 2 construct different time windows for each task according to allowable time deviation. only picking a time window for each task.
 
 
 ---Folder objective 3 & 4:
 
 objective 3:Categorize the uncompleted tasks from Ojective 2 into two different groups
 
              Group 1 : tasks that could potentially get done if the employees worked longer hours
              
              Group 2: tasks that cannot be done, even if employees worked longer hours.
              
 objective 4: have a new variable room limitation. new constrain: each room only allow one taks at each time.
 
              draw plot for employee and task assigned to them, plot for room and task assigned to the room.
 
 
---Folder objective 5:
 objective 5 have a new varible task predecessors. new constraint: if the predecessors of the task was not asssigned and completed, the task can not be assigned.
 
 update the unassigned task classification.
 
 
---Folder objective 6:
 
objective 6 have a new variable employee study variability: how many study is the employee involved in.
 
the maximize function: mdl.maximize(10*total_number_of_assignments-0.1*total_work-total_study_varibility)
 
 workforce_study_varibility_version2 is the version we use. the running speed of version 1 is too slow.
 
 
 
---Folder demo:

new function: pop out window asking use to select the file for production need and employee schedual.

new function: store the output file and plots at the location of cplex.

Production Needs.csv: delet some useless column. make the time for task and its predecessor more reasonable. make the task and its predecessor in the same project. change the name of column. change the name of task id and employee id.

Employee-Task(colored by project).png is per employee : sequence of tasks color coded by project (label with skill), shows total # of projects, total working time


---Folder #of task+work hour

the maximize function is maximize(10*total_number_of_assignments-0.1*total_work).

you can compare the plot with the Employee-Task(colored by project).png in demo folder. you can see by the color that each employee involve in more projects.


---Folder # of task+study variability/

the maximize function is maximize(10*total_number_of_assignments-total_study_varibility)

you can compare the plot with the Employee-Task(colored by project).png in demo folder. you can see by the color that each employee involve in less projects.
 
 
 
 
  

# Info: this section imports packages 
from ast import increment_lineno
from asyncio import tasks
from fileinput import filename
from re import X
from sqlite3 import Row
import sys
import docplex.mp
from collections import namedtuple
import pandas as pd
import numpy as np
import math
import csv
import tkinter
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# Note: please ensure all data files are in csv format. (Production Need, Employee schedual, Employee Skillset)
# This workforce allocation.py file and Employee Skillset file should be stored in the location of Cplex.
# Production Need & Employee schedual file shoud keep the same column name with sample files. (the program assumes these column titles will not change)
# output file and plot are stored at the location of cplex by default. 
# You can just pre-pend the path to filename if u want change the location, for example:savefig('C:\Test\test.fig')
root = tkinter.Tk() #p op-up window for asking select file.
root.withdraw()
#Folderpath=filedialog.askdirectory() # ask for where you store the csv for employee's skill
Needfile = filedialog.askopenfilename(title='Choose your file for production need')# ask for selecting production need document in csv form.
Timefile = filedialog.askopenfilename(title='Choose your file for employee availability')# ask for selecting employee availability in csv form.
#os.chdir(Folderpath)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Info: this section reads .csv files to memory
tasksTable =pd.read_csv(Needfile,sep=',',header=0)
avilibilityTable = pd.read_csv(Timefile, sep=",", header=0)
eskillsTable = pd.read_csv('Employee Skillset.csv', sep=",", header=0)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
o_tasklist =[tasksTable["Need ID"][i] for i in range(len(tasksTable))] #Info: list of the original tasks
nb_task=len(tasksTable)# Info: number of the original tasks
timelist=[x/2 for x in list(range(48))]


def myround(x):
  if x==0:
    return 0
  else:
    return int(round(x)+1)

#Info: This section creates variable EmployeeTime :2D array outlines availability for each employee in integer format along with hour format
def EmployeeTime(x):
  x.insert(1,"Time","")
  for i in range(len(x)):
    start_str=x.loc[i,'Start Time']
    end_str=x.loc[i,'End Time']
    lunchs_str=x.loc[i,'Lunch Start']
    lunche_str=x.loc[i,'Lunch End']
    # transfer the time(00:00) into integer which ranged from 0 to 48 （1 unit indicte half hour).
    starttime=int(start_str[:len(start_str)-3])*2+myround(int(start_str[-2:])/60)
    lunchs=int(lunchs_str[:len(lunchs_str)-3])*2+myround(int(lunchs_str[-2:])/60)
    lunche=int(lunche_str[:len(lunche_str)-3])*2+myround(int(lunche_str[-2:])/60)
    endtime=int(end_str[:len(end_str)-3])*2+myround(int(end_str[-2:])/60)
    x.at[i,'Time']=timelist[starttime:lunchs]+timelist[lunche:endtime]
  employee_time=x.set_index('Employee ID').to_dict()['Time']
  return(employee_time) 

# Info: this section creates a a dictionary (employee_time) of employee and their availability in integer format (ie a subset of EmployeeTime).
# eg:key:10001;value:[18,19] whcih means employee are avilible from 9-10am.
employee_time=EmployeeTime(avilibilityTable)

# Info: This section updates the taskTable - it has all the info related to production tasks (ie ID, room, project, start_time, predecessors...)
# Note: to handle positive and negative variabilities of a possible task time, the software creates muliple copies of the task (one copy for each 30 min increment of variability)
#   For example if a task had +/- 60 min time variability, there would be a total of 5 instances of that task (1 original + 2 negative variability copies + 2 positive variability copies) 
def UpdateTask(x):
  x.insert(1,"Time","")
  nb_task=len(x)
  for i in range(nb_task):
      taskstart_str=x.loc[i,'Start Time']
      taskend_str=x.loc[i,'End Time']
      minus=x.loc[i,'Allowable Negative Deviation'] # Info: negative deviation means task is done earlier than schedualed
      plus=x.loc[i,'Allowable Positive Deviation'] # Info: positive deviation means task is done later than schedualed
      taskstart=int(taskstart_str[:len(taskstart_str)-3])*2+myround(int(taskstart_str[-2:])/60)
      taskend=int(taskend_str[:len(taskend_str)-3])*2+myround(int(taskend_str[-2:])/60)
      minus_num=int(minus[:len(minus)-3])*2+myround(int(minus[-2:])/60)
      plus_num=int(plus[:len(plus)-3])*2+myround(int(plus[-2:])/60)
      x.at[i,'Time']=timelist[taskstart:taskend]
      
      # we adjust the entire task time forward or backward according in units of half an hour.
      # and add it into table as a new task.
      if minus_num !=0:
        for j in range(1,minus_num+1):
          x_new_row=pd.DataFrame({'Need ID':[(x.loc[i,'Need ID'],"-",j)],'Skillset ID':[x.loc[i,'Skillset ID']],
          'Time':[timelist[taskstart-j:taskend-j]],'Room':[x.loc[i,'Room']],
          'Predecessors':[x.loc[i,'Predecessors']],'Project':[x.loc[i,'Project']]})
          x=pd.concat([x,x_new_row],ignore_index=True)
      if plus_num !=0:
        for j in range(1,plus_num+1):
          x_new_row=pd.DataFrame({'Need ID':[(x.loc[i,'Need ID'],"+",j)],'Skillset ID':[x.loc[i,'Skillset ID']],
          'Time':[timelist[taskstart+j:taskend+j]],'Room':[x.loc[i,'Room']],
          'Predecessors':[x.loc[i,'Predecessors']],'Project':[x.loc[i,'Project']]})
        x=pd.concat([x,x_new_row],ignore_index=True)
      x.at[i,'Need ID']=(x.loc[i,'Need ID'],"0")
  return(x)
tasksTable=UpdateTask(tasksTable)

# Info : this section creates employee_skills which is a dictionary of employee and the skill set they have.
def EmployeeSkill(x):
  TeSkill = namedtuple("TeSkill", ["Employee", "skillset"])
  employee_skills = {}
  for esk in x.itertuples(index=False):
    eskt= TeSkill(*esk)
    employee_skills.setdefault(eskt.Employee, []).append(eskt.skillset)
  return(employee_skills)
employee_skills=EmployeeSkill(eskillsTable) 


task_time=tasksTable.set_index('Need ID').to_dict()['Time'] # Info : this section create task time: a dictionary of task and its scheduled time.
task_room=tasksTable.set_index('Need ID').to_dict()['Room'] # Info : this section creates a dictionary of task and the room required for the task.
task_study=tasksTable.set_index('Need ID').to_dict()['Project'] # Info : this section creates dictionary  of task and which study/project it belongs to.
task_skill=tasksTable.set_index('Need ID').to_dict()['Skillset ID'] # Info : this section creates task_skill whichi is a a dictionary of task and its required skill set.

# This section creates a list for all the skills that we have, all the rooms that we have, all the employees that we have  
skillslist = eskillsTable['Skillset ID'].unique()
roomlist=tasksTable['Room'].unique()
employeelist = avilibilityTable['Employee ID'].unique()
tasklist = tasksTable["Need ID"].unique()
studylist= tasksTable['Project'].unique()



### Import the model - this is the start of the doCplex
from docplex.mp.model import Model
mdl = Model("employee")

### Define the decision variables for doCplex model
# binary variable, 1 indicte task T is assigned to employee E. employee_task_vars is a 2D variable
employee_task_vars = mdl.binary_var_matrix(employeelist, tasklist)

# integer variable, represents for the total number of assigned task.
total_number_of_assignments = mdl.sum(employee_task_vars[e,t] for e in employeelist for t in tasklist)

### Create variables
# Info: create model variables. 
# These variables will serve to create employee_work_time_vars which will contain the the actual optimized working time decided on by the model. 
# Employee_work_time_vars is subsequently used to calculated Total_work variable which is the sum of work for all employees used in the optimization function.  
work_end_var=mdl.continuous_var_dict(employeelist) # end time of the last task of employee E.
work_start_var=mdl.continuous_var_dict(employeelist) # 1/(start time of the first task) of employee E.
employee_work_time_vars = mdl.continuous_var_dict(employeelist, lb=0, name='EmployeeWorkTime') # actural working time for employee E.

for e in employeelist:
    mdl.add_constraint(work_end_var[e]==mdl.max(employee_task_vars[e,t]*task_time[t][-1] for t in tasklist))
    mdl.add_constraint(work_start_var[e]==mdl.max(employee_task_vars[e,t]/task_time[t][0] for t in tasklist))
    mdl.add_constraint(employee_work_time_vars[e]== work_end_var[e]+work_start_var[e])
total_work=mdl.sum(employee_work_time_vars[e] for e in employeelist)

# Info: create model variables and constraints. 
# These variables will serve to create employee_study_varibility which will contain the the actual optimized projects assigned to employees decided on by the model. 
# employee_study_varibility is subsequently used to calculate total_study_varibility variable which is the total number of projects assigned to employees used in the optimization function.  
study_number=mdl.integer_var_matrix(employeelist,studylist) # number of task assigned to employee E and belongs to study S.
number=mdl.binary_var_matrix(employeelist,studylist) # a binary varible, 1 if the employee E is assigned a task belongs to study S.
employee_study_varibility=mdl.integer_var_dict(employeelist) # number of study that employee E involved in.

# Constraint 0: 
# To minimize the number of projects two variables were created : -
#       - Study_number : the number of tasks associated to a given study for each employee (2D variable)
#       - number : integer indicating if a study has been associated to a given employee (0 or 1) (2D variabl)
#   Constraint : study_number >= number for a given study for a given employee
#   Constraint : number * the number of all assigned task for all projects >= study_number for that employee for that project

nb_tasks = len(tasksTable)
for e in employeelist:
  for s in studylist:
    for t in tasksTable[tasksTable["Project"]== s]["Need ID"]:
      study_number[e,s]+=mdl.sum(employee_task_vars[e,t])
    mdl.add_constraint(number[e,s]<=study_number[e,s])
    mdl.add_constraint(study_number[e,s]<=number[e,s]*nb_tasks)
  employee_study_varibility[e]=mdl.sum(number[e,s] for s in studylist)
total_study_varibility=mdl.sum(employee_study_varibility[e] for e in employeelist)


# Info: Constraint 1: when employee are unavaibele, he can not be assigned any tasks starting that time. (ie set employee_task_vars to 0 when employee is not available to work)
for e in employeelist:
  for t in tasklist:
    # all():check if employee_time contain all element of task_time.
    # == False: if not, we don't assign task T to employeee E.
    if all(elem in employee_time[e] for elem in task_time[t])==False:
      mdl.add_constraint(employee_task_vars[e, t] == 0)

#  Info: Constraint 2: can not assign overlap task; employee perform one task at each t. (it looks at any tasks which overlaps at the same time, and then ensures that these are not assigned to a given employee)
for i1 in range(nb_tasks):
    for i2 in range(i1 + 1, nb_tasks):
      s1=tasksTable.loc[i1,'Need ID']
      s2=tasksTable.loc[i2,'Need ID']
      # check if the list of shcedualed time for two tasks have common elements
      if len(set(tasksTable.iloc[i1,1]) & set(tasksTable.iloc[i2,1]))>0:
        for n in employeelist:
          mdl.add_constraint(employee_task_vars[n, s1] + employee_task_vars[n, s2] <= 1)

#  Info: Constraint 3: enforce skill requirements for selected task.
for task in tasklist:
  for employee in employeelist:
    if len(set(employee_skills[employee])& set(task_skill[task]))==0:# check if two lists have common elements.
      mdl.add_constraint(employee_task_vars[employee, task] == 0)

#  Info: Constraint 4: one task only need one employee.
for s in tasklist:
    total_assigned= mdl.sum(employee_task_vars[n, s] for n in employeelist) # number of task is assigned to task T.
    mdl.add_constraint(total_assigned<= 1)

#  Info: Constraint 5: only pick one time window for each task. (this is constructed very similar to constraint 2). 
# This constraint to select only 1 instance of a taskID from the tasksTable - which can contain multiple instances of a NeedID (when there are positive and/or negative variabilities associated to that task) 
for i1 in range(nb_tasks):
    s1=tasksTable.loc[i1,'Need ID']
    total_assigned_s1= mdl.sum(employee_task_vars[n, s1] for n in employeelist)
    for i2 in range(i1+1, nb_tasks):     
      s2=tasksTable.loc[i2,'Need ID']
      total_assigned_s2= mdl.sum(employee_task_vars[n, s2] for n in employeelist)
      if s1[0]==s2[0]:
          mdl.add_constraint(total_assigned_s1 + total_assigned_s2<= 1)

#  Info: Constraint 6: job room limitation. 
# Current this ensures that there is no more than one task being performed in a room at a given time. 
# Note you can set the maximium number of concurent tasks in a room at a given time using this constraint
for h in timelist:
  for r in roomlist:
    total_assign=0
    for t in tasksTable[tasksTable["Room"]== r]["Need ID"]:
      if h in task_time[t]:
        total_assign+=mdl.sum(employee_task_vars[e,t] for e in employeelist)
    mdl.add_constraint(total_assign<=1)

#  Info: Constraint 7: tasks precessors
# The purpose of this section is to ensure that the task predecessors are completed prior to the start of a given task
# this is accomplished by:
#     Step 1 : the total number of assigned task T should always less or equal than total number of assigned predecessor tasks.
#     Step 2 : if start time of the task < end time predecessor then [...]
#     note: a deeper dive is needed to better understand this constraint
for i in range(nb_tasks):
  t=tasksTable.loc[i,'Need ID']
  if pd.isna(tasksTable.loc[i,'Predecessors'])==False:
    p=tasksTable.loc[i,'Predecessors']
    timeB=task_time[t][0] # time B is the starting time of the task T
    for j in range(nb_tasks):
      if tasksTable.loc[j,'Need ID'][0]==p:
        timeA=task_time[tasksTable.loc[j,'Need ID']][-1] # time A is the ending time of predecessor
        total_assign_t=mdl.sum(employee_task_vars[e,t] for e in employeelist) # total number of assigned task for all employees
        total_assign_j=mdl.sum(employee_task_vars[e,tasksTable.loc[j,'Need ID']] for e in employeelist) # total number of assigned predecessors task for all employees
        # number of assigned task T should always less or equal than number of assigned predecessor.
        mdl.add_constraint(total_assign_t<=total_assign_j) 
        
        # if predecessor is late than task T, we only assign one or none of two tasks.
        if timeB<=timeA:
          mdl.add_constraint(total_assign_j+total_assign_t<=1)


###Aim function and slove
mdl.maximize(10*total_number_of_assignments-0.1*total_work-total_study_varibility)
s = mdl.solve()



### Output of the model

## employee actual working time & table for employee and room schedual
# find the actual startind and ending time for each employee 
# work_time is a list (1D) of the total hours worked by each employee in the day
work_time=[] 
employee_end_time={}
for e in employeelist:
  start_time= 24
  end_time=0
  for t in tasklist:
    if employee_task_vars[e,t].solution_value >0:
      start_time = min(task_time[t][0],start_time)
      end_time = max(task_time[t][-1],end_time)
  employee_end_time[e]=end_time
  work_time.append(max(end_time-start_time+0.5,0))
total_work_hours=sum(work_time)
print("The total work hours is",total_work_hours,"hours")


# shared code : creates labels for figures
def createLabels(ax, title, xlabel, ylabel):
  tmin=int(min(employee_time.values())[0])
  tmax=math.ceil(int(max(employee_time.values())[-1])+0.5)
  plt.xticks(np.arange(tmin,tmax),["{}".format(str(s)) for s in timelist[tmin:tmax]])
  plt.xlim([tmin,tmax])
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.grid()
  ax.set_title(title)

# assign a color to each skill set to be used in figures
colorBySkills = {}
for d, c in zip(skillslist, ['r', 'm', 'b', 'g', 'y', 'c']):
    colorBySkills[d] = c
colors=['k','r', 'm', 'b', 'g', 'y', 'c']
labels=['avilibility','A','B','C','D','E','F']
red_patch = [mpatches.Patch(color=colors[i], label=labels[i])for i in range(len(labels))]

# plot the task assignment for each employee and save the plot
# plot support formats: eps, jpeg, jpg, pdf, pgf, png....
def displayTasksAssignmentsGantt(ax):
  ylabels, tickloc = [], []
  for i in range(len(employeelist)):
    n=employeelist[i]
    lunchs_str=avilibilityTable["Lunch Start"][i]
    lunchs=(int(lunchs_str[:len(lunchs_str)-3])*2+myround(int(lunchs_str[-2:])/60))/2
    ax.bar(employee_time[n][0],0.3,width=lunchs-employee_time[n][0],bottom=i+0.5,color='k',align='edge')
    ax.bar(lunchs+1,0.3,employee_time[n][-1]-lunchs-0.5,bottom=i+0.5,color='k',align='edge')
    for s in tasklist:
      if employee_task_vars[n,s].solution_value > 0:
        ax.bar(task_time[s][0], 0.4,
        width=task_time[s][-1] - task_time[s][0]+0.455, bottom=i + 0.1,
        color=colorBySkills[task_skill[s]],align='edge')
        ax.text(task_time[s][0],i+0.2,s[0],fontsize=12)
                  
    ylabels.append("{}:{}".format(str(n),employee_skills[n])) # (3)number of total hours works
    tickloc.append(i + 0.5)

  #configure plot paramaters
  plt.ylim(0, len(employeelist))
  plt.yticks(tickloc, ylabels)
  plt.legend(handles=[red_patch][0])#[0] beacuse error:'list' object has no attribute 'get_label'
  plt.savefig('Employee-Task plot.png')
  plt.close()
  createLabels(ax, 'Taks Assignments', 'Hours Of The Day', 'Employee ID & Skill')

# plot the task assignment for each room and save the plot
def displayTasksRoomGantt(ax):
  ylabels, tickloc = [], []
  for i in range(len(roomlist)):
    n=roomlist[i]
    room_tasklist=[k for k,v in task_room.items() if v == n]
    for s in room_tasklist:
      for e in employeelist:
        if employee_task_vars[e,s].solution_value > 0:
          ax.bar(task_time[s][0], 0.2,width=task_time[s][-1] - task_time[s][0]+0.455, bottom=i + 0.2,
          color=colorBySkills[task_skill[s]],align='edge')
          ax.text(task_time[s][0],i+0.2,s[0],fontsize=12)              
    ylabels.append("{}:{}".format('Room',str(n))) # (3)number of total hours works
    tickloc.append(i + 0.5)
  plt.ylim(0, len(roomlist))
  plt.yticks(tickloc, ylabels)
  plt.savefig('Room-Task plot.png')
  plt.close()
  createLabels(ax, 'Taks Assignments', 'Hours Of The Day', 'Task Room')

#display figures
fig = plt.figure(figsize=[14,12])
ax = plt.subplot(111)
displayTasksAssignmentsGantt(ax)

fig = plt.figure(figsize=[14,6])
bx = plt.subplot(111)
displayTasksRoomGantt(bx)


## list and classify the unassigned task, and provide corresponding reason.
#print the list of unassigned list and its scheduled time, required skill set.
def unassignTask(assignlist,o_tasklist):
  for e in employeelist:
    for t in tasklist:
      if employee_task_vars[e,t].solution_value > 0:
        if (t[0] in o_tasklist):
          assignlist.append(t[0])
  return(list(set(o_tasklist)-set(assignlist)))
unassignlist=unassignTask([],o_tasklist)
print("Unassigned Task:",unassignlist)
for i in unassignlist:
  print(i,": from",task_time[(i,"0")][0],"to",task_time[(i,"0")][-1]+0.5,",required skillset:",task_skill[(i,"0")])

#classify the unassigned task and output it as csv file.
header = ['Task ID', 'Type', 'Detail','Employee/Skill/Room']
with open('UnassignTaskType.csv', 'w', encoding='UTF8') as f:
  writer = csv.writer(f)
  writer.writerow(header)
  
  # find tasks which were not completed because predecessor was not completed, write the info to the file and remove those tasks from the unassigned list.
  deletelist=[]
  for i in unassignlist:
    Predecessor=tasksTable.loc[tasksTable["Need ID"]==(i,"0")]['Predecessors'].tolist()[0]
    if (np.isnan(Predecessor)==False):
      if int(Predecessor) in unassignlist:
        writer.writerow([i,'I','The task can not be done due to its Predecessor',Predecessor])
        deletelist.append(i)
  newunassignlist=set(unassignlist)^set(deletelist)
  deletelist.clear()  
  
  # classify remaining tasks which have been unassigned due to room limitations, write to file and remove the task from the unassigned list.
  for i in unassignlist:
    Skill=task_skill[(i,"0")]
    Room=task_room[(i,"0")]
    EmployeeList=eskillsTable[eskillsTable['Skillset ID']== Skill]['Employee ID'].tolist()
    Deviation_n=tasksTable.loc[tasksTable["Need ID"]==(i,"0")]["Allowable Negative Deviation"].to_string(index=False)
    Deviation_p=tasksTable.loc[tasksTable["Need ID"]==(i,"0")]["Allowable Positive Deviation"].to_string(index=False)
    nb_negative=myround(int(Deviation_n[-2:])/60)+float(Deviation_n[:len(Deviation_n)-3])*2
    nb_positive=myround(int(Deviation_p[-2:])/60)+float(Deviation_p[:len(Deviation_p)-3])*2   
    case=0
    
    for j in range(-1*int(nb_negative),int(nb_positive)+1): # each allowable time deviation of the unassigned task:
      time_window=(np.array(task_time[(i,"0")])+j*0.5).tolist()
      # chck if any employee have free time to do this assigned task at the specific time window
      if any(len(list(set(time_window).intersection(employee_time[e])))==len(task_time[(i,"0")]) for e in EmployeeList):
        # if any task are assigned and have the same room with the unassigned task 
        # at any time of the time window, the unassigned task can not be done due to room limitation
        for t in time_window:
          TaskList=[k for k,v in task_time.items() if t in v]
          for k in TaskList:
            if ((any(employee_task_vars[e,k].solution_value>0 for e in employeelist)) and (task_room[k]==Room)):
              case+=1
    if case>0:
      writer.writerow([i,'II','The task can not be done due to room limitation',Room])
      deletelist.append(i)
  newunassignlist=set(unassignlist)^set(deletelist)

  # classify remaining tasks which have been unassigned into need overtime or need more skills, write to file and remove the task from the unassigned list.
  # can be resolved by overtime : means that if an employee is able to work additional hours, then the task can be completed
  # can be resolved by more skills : means that even if the employee was available to work additional hours (after their last task), the task could not be completed.
  # note : overtime is defined as working additional hours after the end time, NOT work additional hours before the start time
  
  for i in newunassignlist:
    Skill=task_skill[(i,"0")]
    EmployeeList=eskillsTable[eskillsTable['Skillset ID']== Skill]['Employee ID'].tolist()
    Deviation=tasksTable.loc[tasksTable["Need ID"]==(i,"0")]["Allowable Positive Deviation"].to_string(index=False)
    nb=myround(int(Deviation[-2:])/60)+float(Deviation[:len(Deviation)-3])*2
    index=[]
    for j in EmployeeList:
      case=0
      # if ending time of task is earlier than the ending time of employee's avibility, it can not be solved by work overtime.
      # if ...is later..., however ending time of task in earlier than the ending time of last task of this employee,...
      # other than that, it can be solved by work overtime.
      if task_time[(i,"0")][-1]+nb*0.5> employee_time[j][-1]:
        if task_time[(i,"0")][0]+nb*0.5< employee_time[j][-1]:
          if employee_end_time[j]<task_time[(i,"0")][0]+nb*0.5:
            case=1
        else: case=1
      else:case=0
      index.append(case)
    if sum(index)>0:
      writer.writerow([i,'III','Task can be done if employee work longer time'])
    else:
      writer.writerow([i,'IV','We should train more employee for skill:',Skill])

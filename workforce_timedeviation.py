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

## This file and employee-skill's table should be store in the location of Cplex.
# Note: please ensure all data files are in csv format
root = tkinter.Tk()
root.withdraw()
#Folderpath=filedialog.askdirectory() # ask for where you store the csv for employee's skill, employee's avilibility
Needfile = filedialog.askopenfilename(title='Choose your file for production need')#a sk for selecting production need document in csv form.
Timefile = filedialog.askopenfilename(title='Choose your file for employee availability')#a sk for selecting employee availability in csv form.
#os.chdir(Folderpath)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Info: this section reads .csv files to memory
tasksTable =pd.read_csv(Needfile,sep=',',header=0)
avilibilityTable = pd.read_csv(Timefile, sep=",", header=0)
eskillsTable = pd.read_csv('Employee Skillset1.csv', sep=",", header=0)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
o_tasklist =[tasksTable["Need ID"][i] for i in range(len(tasksTable))]
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
    start_str=x.iloc[i,3]
    end_str=x.iloc[i,4]
    lunchs_str=x.iloc[i,5]
    lunche_str=x.iloc[i,6]
    # transfer the time(00:00) into integer which ranged from 0 to 48 （1 unit indicte half hour).
    starttime=int(start_str[:len(start_str)-3])*2+myround(int(start_str[-2:])/60)
    lunchs=int(lunchs_str[:len(lunchs_str)-3])*2+myround(int(lunchs_str[-2:])/60)
    lunche=int(lunche_str[:len(lunche_str)-3])*2+myround(int(lunche_str[-2:])/60)
    endtime=int(end_str[:len(end_str)-3])*2+myround(int(end_str[-2:])/60)
    x.iat[i,1]=timelist[starttime:lunchs]+timelist[lunche:endtime]
  employee_time={}
  TeTime = namedtuple("TeTime", ["Employee", "timeset"])
  for et in x.iloc[:,0:2].itertuples(index=False):
    ett=TeTime(*et)
    employee_time.setdefault(ett.Employee,ett.timeset)
  return(employee_time) 

# Info: this section creates a a dictionary of employee and their availability in integer format (ie a subset of EmployeeTime).
# eg:key:10001;value:[18,19] whcih means employee are avilible from 9-10am.
employee_time=EmployeeTime(avilibilityTable)

# Info: This section updates the taskTable - it has all the info related to production tasks (ie ID, room, project, start_time, predecessors...)
def UpdateTask(x):
  x.insert(1,"Time","")
  nb_task=len(x)
  selected_columns = x[["Need ID","Time","SkillsetID1"]]
  assignTable=selected_columns.copy()
  for i in range(nb_task):
      taskstart_str=x.iloc[i,4]
      taskend_str=x.iloc[i,6]
      minus=x.iloc[i,7]
      plus=x.iloc[i,8]
      taskstart=int(taskstart_str[:len(taskstart_str)-3])*2+myround(int(taskstart_str[-2:])/60)
      taskend=int(taskend_str[:len(taskend_str)-3])*2+myround(int(taskend_str[-2:])/60)
      minus_num=int(minus[:len(minus)-3])*2+myround(int(minus[-2:])/60)
      plus_num=int(plus[:len(plus)-3])*2+myround(int(plus[-2:])/60)
      x.iat[i,1]=timelist[taskstart:taskend]
      assignTable.iat[i,1]=timelist[taskstart:taskend]
      # we adjust the entire task time forward or backward according to its allowable deviation in units of half an hour.
      # and add it into table as a new task.
      for j in range(1,minus_num+1):
        x_new_row=pd.DataFrame({'Need ID':[(x.iloc[i,0],"-",j)],'SkillsetID1':[x.iloc[i,2]],
        'Time':[timelist[taskstart-j:taskend-j]],'Room':[x.iloc[i,9]],'Predecessors':[x.iloc[i,10]]})
        x=pd.concat([x,x_new_row],ignore_index=True)
      for j in range(1,plus_num+1):
        x_new_row=pd.DataFrame({'Need ID':[(x.iloc[i,0],"+",j)],'SkillsetID1':[x.iloc[i,2]],
        'Time':[timelist[taskstart+j:taskend+j]],'Room':[x.iloc[i,9]],'Predecessors':[x.iloc[i,10]]})
        x=pd.concat([x,x_new_row],ignore_index=True)
      x.iat[i,0]=(x.iloc[i,0],"0")
  return(x)
tasksTable=UpdateTask(tasksTable)

# Info : this section creates employee_skills whichi is a dictionary of employee and the skill set they have.
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
task_study=tasksTable.set_index('Need ID').to_dict()['Study Number'] # Info : this section creates dictionary  of task and which study/project it belongs to.
task_skill=tasksTable.set_index('Need ID').to_dict()['SkillsetID1'] # Info : this section creates task_skill whichi is a a dictionary of task and its required skill set.

# This section creates a list for all the skills that we have, all the rooms that we have, all the employees that we have  
skillslist = eskillsTable['Skillset ID'].unique()
roomlist=['A','B','C']
employeelist = [avilibilityTable["Employee ID"][i] for i in range(len(avilibilityTable))]
tasklist = [tasksTable["Need ID"][i] for i in range(len(tasksTable))]
studylist=['A','B','C']


### Import the model - this is the start of the doCplex
from docplex.mp.model import Model
mdl = Model("employee")

### Define the decision variables for doCplex model
# binary variable, 1 indicte task T is assigned to employee E.
employee_task_vars = mdl.binary_var_matrix(employeelist, tasklist, 'EmployeeAssigned')
# integer variable, represents for the total number of assigned task.
total_number_of_assignments = mdl.sum(employee_task_vars[e,t] for e in employeelist for t in tasklist)

### Create variables
# Info: create model variables. These variables will serve to create employee_work_time_vars which will contain the the actual optimized working time decided on by the model. Employee_work_time_vars is subsequently used to calculated Total_work variable which is the sum of work for all employees used in the optimization function.  
work_end_var=mdl.continuous_var_dict(employeelist) # end time of the last task of employee E.
work_start_var=mdl.continuous_var_dict(employeelist) # 1/(start time of the first task) of employee E.
employee_work_time_vars = mdl.continuous_var_dict(employeelist, lb=0, name='EmployeeWorkTime') # actural working time for employee E.

for e in employeelist:
    mdl.add_constraint(work_end_var[e]==mdl.max(employee_task_vars[e,t]*task_time[t][-1] for t in tasklist))
    mdl.add_constraint(work_start_var[e]==mdl.max(employee_task_vars[e,t]/task_time[t][0] for t in tasklist))
    mdl.add_constraint(employee_work_time_vars[e]== work_end_var[e]+work_start_var[e])
total_work=mdl.sum(employee_work_time_vars[e] for e in employeelist)

# Info: create model variables. These variables will serve to create employee_study_varibility which will contain the the actual optimized projects assigned to employees decided on by the model. employee_study_varibility is subsequently used to calculate total_study_varibility variable which is the total number of projects assigned to employees used in the optimization function.  
study_number=mdl.integer_var_matrix(employeelist,studylist) # number of task assigned to employee E and belongs to study S.
number=mdl.binary_var_matrix(employeelist,studylist) # a binary varible, 1 if the employee E is assigned a task belongs to study S.
employee_study_varibility=mdl.integer_var_dict(employeelist) # number of study that employee E involved in.

nb_tasks = len(tasksTable)
for e in employeelist:
  for s in studylist:
    for t in tasksTable[tasksTable["Study Number"]== s]["Need ID"]:
      study_number[e,s]+=mdl.sum(employee_task_vars[e,t])
    mdl.add_constraint(number[e,s]<=study_number[e,s])
    mdl.add_constraint(study_number[e,s]<=number[e,s]*nb_tasks)
  employee_study_varibility[e]=mdl.sum(number[e,s] for s in studylist)+1
total_study_varibility=mdl.sum(employee_study_varibility[e] for e in employeelist)


### Info: Create constraints
# Info: Constraint 1: when employee are unavaibele, he can not be assigned any tasks starting that time.
for e in employeelist:
  for t in tasklist:
    # check if the list of employee's avariblity has common elements with the list of task scheduled time.
    if all(elem in employee_time[e] for elem in task_time[t])==False:
      mdl.add_constraint(employee_task_vars[e, t] == 0)

#  Info: Constraint 2: can not assign overlap task; employee perform one task at each t.
for i1 in range(nb_tasks):
    for i2 in range(i1 + 1, nb_tasks):
      s1=tasksTable.iloc[i1,0]
      s2=tasksTable.iloc[i2,0]
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

#  Info: Constraint 5: only pick one time window for each task.
for i1 in range(nb_tasks):
    s1=tasksTable.iloc[i1,0]
    total_assigned_s1= mdl.sum(employee_task_vars[n, s1] for n in employeelist)
    for i2 in range(i1 + 1, nb_tasks):     
      s2=tasksTable.iloc[i2,0]
      total_assigned_s2= mdl.sum(employee_task_vars[n, s2] for n in employeelist)
      if s1[0]==s2[0]:
          mdl.add_constraint(total_assigned_s1 + total_assigned_s2<= 1)

#  Info: Constraint 6: job room limitation.
for h in timelist:
  for r in roomlist:
    total_assign=0
    for t in tasksTable[tasksTable["Room"]== r]["Need ID"]:
      if h in task_time[t]:
        total_assign+=mdl.sum(employee_task_vars[e,t] for e in employeelist)
    mdl.add_constraint(total_assign<=1)

#  Info: Constraint 7: tasks precessors
for i in range(nb_tasks):
  t=tasksTable.iloc[i,0]
  if np.isnan(tasksTable.iloc[i,10])==False:
    p=tasksTable.iloc[i,10]
    timeB=task_time[t][0] # time B is the starting time of the task T
    for j in range(nb_tasks):
      if tasksTable.iloc[j,0][0]==p:
        timeA=task_time[tasksTable.iloc[j,0]][-1] # time A is the ending time of predecessor
        total_assign_t=mdl.sum(employee_task_vars[e,t] for e in employeelist)
        total_assign_j=mdl.sum(employee_task_vars[e,tasksTable.iloc[j,0]] for e in employeelist)
        # number of assigned task T should always less or equal than number of assigned predecessor.
        mdl.add_constraint(total_assign_t<=total_assign_j) 
        # if predecessor is late than task T, we only assign one or none of two tasks.
        if timeB<=timeA:
          mdl.add_constraint(total_assign_j+total_assign_t<=1)

##i#nformations of model (for troubleshooting purposes)
mdl.add_kpi(total_number_of_assignments,"number of assigned task")
mdl.add_kpi(total_study_varibility,'study_varibility')
mdl.print_information()


###Aim function and slove
mdl.maximize(10*total_number_of_assignments-0.1*total_work-total_study_varibility)
s = mdl.solve()
mdl.report()


### Output of the model

## employee actural working time & table for employee and room schedual
# find the actural startind and ending time for each employee 
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

# shared code
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
  plt.ylim(0, len(employeelist))
  plt.yticks(tickloc, ylabels)
  plt.legend(handles=[red_patch][0])#[0] beacuse error:'list' object has no attribute 'get_label'
  plt.savefig('figure1.png')
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
  plt.savefig('figure2.png')
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
  deletelist=[]
  for i in unassignlist:
    Predecessor=tasksTable.loc[tasksTable["Need ID"]==(i,"0")]['Predecessors'].tolist()[0]
    if (np.isnan(Predecessor)==False):
      if int(Predecessor) in unassignlist:
        writer.writerow([i,'I','The task can not be done due to its Predecessor',Predecessor])
        deletelist.append(i)
  newunassignlist=set(unassignlist)^set(deletelist)
  deletelist.clear()  
  for i in unassignlist:
    Skill=task_skill[(i,"0")]
    Room=task_room[(i,"0")]
    EmployeeList=eskillsTable[eskillsTable['Skillset ID']== Skill]['Employee ID'].tolist()
    Deviation_n=tasksTable.loc[tasksTable["Need ID"]==(i,"0")]["Allowble Negative Deviation (hrs)"].to_string(index=False)
    Deviation_p=tasksTable.loc[tasksTable["Need ID"]==(i,"0")]["Allowable Positive Deviation (hrs)"].to_string(index=False)
    nb_negative=myround(int(Deviation_n[-2:])/60)+float(Deviation_n[:len(Deviation_n)-3])*2
    nb_positive=myround(int(Deviation_p[-2:])/60)+float(Deviation_p[:len(Deviation_p)-3])*2   
    case=0
    for j in range(-1*int(nb_positive),int(nb_negative)+1): # each allowable time deviation of the unassigned task:
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

  #when we speak of work overtime, we only refer to leave off later.
  for i in newunassignlist:
    Skill=task_skill[(i,"0")]
    EmployeeList=eskillsTable[eskillsTable['Skillset ID']== Skill]['Employee ID'].tolist()
    Deviation=tasksTable.loc[tasksTable["Need ID"]==(i,"0")]["Allowble Negative Deviation (hrs)"].to_string(index=False)
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
      writer.writerow([i,'IV','We should tran more employee',Skill])

